import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import keyboard
import argparse
import editdistance

# Global flags for training control
PAUSE_TRAINING = False
STOP_TRAINING = False
END_EPOCH = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 16
learning_rate_task = 1e-4
learning_rate_afdm = 1e-3
num_epochs = 100
image_height = 64  # Fixed height, maintain aspect ratio
hidden_size = 256
num_layers = 2

# Dataset class with aspect ratio preservation
class CharacterDataset(Dataset):
    def __init__(self, image_paths, labels, target_height=64, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.target_height = target_height
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Preserve aspect ratio
        width, height = image.size
        new_width = int(width * (self.target_height / height))
        image = image.resize((new_width, self.target_height))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data preparation
def prepare_data(data_dir):
    image_paths = []
    labels = []
    label_to_idx = {}
    idx = 0
    
    for character_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, character_folder)
        if os.path.isdir(folder_path):
            if character_folder not in label_to_idx:
                label_to_idx[character_folder] = idx
                idx += 1
            
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, image_file))
                    labels.append(label_to_idx[character_folder])
    
    return image_paths, labels, label_to_idx

# TPS Grid Generation
class TPSGridGen(nn.Module):
    def __init__(self, H, W, K=4):
        super(TPSGridGen, self).__init__()
        self.H, self.W = H, W
        self.K = K
        
        # Generate fixed base control points in normalized coordinates
        self.base_control_points = self._generate_base_control_points()
        
        # Generate target grid in normalized coordinates
        self.target_grid = self._generate_target_grid()
        
    def _generate_base_control_points(self):
        # Create a grid of K x K control points (normalized to [-1, 1])
        k_sqrt = int(math.sqrt(self.K))
        x = torch.linspace(-1, 1, steps=k_sqrt)
        y = torch.linspace(-1, 1, steps=k_sqrt)
        grid_x, grid_y = torch.meshgrid(x, y)
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return points
    
    def _generate_target_grid(self):
        # Create a regular grid for target coordinates
        x = torch.linspace(-1, 1, steps=self.W)
        y = torch.linspace(-1, 1, steps=self.H)
        grid_y, grid_x = torch.meshgrid(y, x)
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return grid
    
    def _compute_delta_matrix(self, control_points):
        # Compute the delta matrix for TPS transformation
        K = control_points.shape[0]
        delta = torch.zeros(K+3, K+3, device=control_points.device)
        
        # Fill the upper left block with radial basis function values
        for i in range(K):
            for j in range(K):
                if i != j:
                    delta[i, j] = self._U(torch.norm(control_points[i] - control_points[j]))
        
        # Fill the right block with control points coordinates
        delta[:K, K] = 1
        delta[:K, K+1:] = control_points
        
        # Fill the bottom block with zeros and control points
        delta[K, :K] = 1
        delta[K+1:, :K] = control_points.t()
        
        return delta
    
    def _U(self, r):
        # TPS basis function U(r) = r²log(r²)
        return r**2 * torch.log(r**2 + 1e-6)
    
    def compute_tps_transformation(self, source_points):
        # Compute the TPS transformation matrix
        batch_size = source_points.size(0)
        
        # Compute delta matrix for base control points
        delta_P = self._compute_delta_matrix(self.base_control_points.to(source_points.device))
        
        # Compute inverse of delta matrix
        inv_delta_P = torch.inverse(delta_P)
        
        # Batch processing for source points
        batch_transformed_grid = []
        
        for b in range(batch_size):
            # Compute parameters for TPS transformation
            param_matrix = torch.zeros(self.K+3, 2, device=source_points.device)
            param_matrix[:self.K, :] = source_points[b]
            
            # Compute the transformation coefficients
            coefficients = torch.matmul(inv_delta_P, param_matrix)
            
            # Apply transformation to the target grid
            batch_grid = self.target_grid.to(source_points.device)
            
            # Compute radial basis function values for target grid
            U = torch.zeros(batch_grid.size(0), self.K, device=source_points.device)
            for i in range(self.K):
                U[:, i] = self._U(torch.norm(batch_grid - self.base_control_points[i].to(source_points.device), dim=1))
            
            # Compute transformed grid
            P = torch.cat([torch.ones(batch_grid.size(0), 1, device=source_points.device), batch_grid], dim=1)
            transformed_grid = torch.matmul(U, coefficients[:self.K, :])
            transformed_grid += torch.matmul(P, coefficients[self.K:, :])
            
            # Reshape to match the image dimensions [H, W, 2]
            transformed_grid = transformed_grid.view(self.H, self.W, 2)
            batch_transformed_grid.append(transformed_grid)
        
        # Stack batch dimension [B, H, W, 2]
        return torch.stack(batch_transformed_grid, dim=0)

# Adversarial Feature Deformation Module (AFDM)
class AFDM(nn.Module):
    def __init__(self, input_channels, k=4, K=16):
        super(AFDM, self).__init__()
        self.k = k  # Number of feature sub-maps
        self.K = K  # Number of control points
        self.channels_per_submap = input_channels // k
        
        # Localization network for predicting control points
        self.localization_net = nn.Sequential(
            nn.Conv2d(self.channels_per_submap, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * K),  # Predict 2D coordinates for K control points
            nn.Tanh()  # Normalize coordinates to [-1, 1]
        )
        
    def forward(self, x, apply_deformation=True):
        # Split input into k sub-maps
        batch_size, C, H, W = x.size()
        sub_maps = torch.split(x, self.channels_per_submap, dim=1)
        
        # Process each sub-map
        deformed_sub_maps = []
        for i, sub_map in enumerate(sub_maps):
            if apply_deformation:
                # Predict control points
                control_points = self.localization_net(sub_map)
                control_points = control_points.view(batch_size, self.K, 2)
                
                # Generate TPS transformation grid
                tps_grid_gen = TPSGridGen(H, W, K=self.K)
                grid = tps_grid_gen.compute_tps_transformation(control_points)
                
                # Normalize grid to [-1, 1] for grid_sample
                grid = grid * 2 - 1
                
                # Apply transformation with bilinear sampling
                deformed_sub_map = F.grid_sample(sub_map, grid, align_corners=True)
                deformed_sub_maps.append(deformed_sub_map)
            else:
                # Skip deformation for some samples
                deformed_sub_maps.append(sub_map)
        
        # Concatenate deformed sub-maps
        deformed_feature_map = torch.cat(deformed_sub_maps, dim=1)
        
        return deformed_feature_map

# Main CRNN model with AFDM
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # 8-layer CNN as per requirements
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pool after conv1
        )
        
        self.cnn_stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pool after conv2
        )
        
        self.cnn_stage3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.cnn_stage3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Pool after conv3_2
        )
        
        # AFDM module to be inserted after conv4_1
        self.cnn_stage4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # AFDM module
        self.afdm = AFDM(input_channels=512, k=4, K=16)
        
        self.cnn_stage4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Pool after conv4_2
        )
        
        self.cnn_stage5_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.cnn_stage5_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),  # 2x2 kernel for final conv
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Pool after conv5_2
        )
        
        # Map-to-Sequence
        self.map_to_seq = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None))  # Collapse height dimension
        )
        
        # Bidirectional LSTM layers
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # Transcription layer (CTC)
        self.classifier = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for blank in CTC
        
    def forward(self, x, apply_deformation=True):
        # CNN feature extraction
        x = self.cnn_stage1(x)
        x = self.cnn_stage2(x)
        x = self.cnn_stage3_1(x)
        x = self.cnn_stage3_2(x)
        x = self.cnn_stage4_1(x)
        
        # Apply AFDM
        x = self.afdm(x, apply_deformation)
        
        # Continue with CNN
        x = self.cnn_stage4_2(x)
        x = self.cnn_stage5_1(x)
        x = self.cnn_stage5_2(x)
        
        # Map to sequence
        x = self.map_to_seq(x)  # [batch_size, channels, 1, width]
        x = x.squeeze(2)  # [batch_size, channels, width]
        x = x.permute(0, 2, 1)  # [batch_size, width, channels]
        
        # RNN
        x, _ = self.rnn(x)
        
        # Transcription
        output = self.classifier(x)
        
        return output

# CTC Loss
class CTCLoss(nn.Module):
    def __init__(self, blank=0):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.criterion(log_probs, targets, input_lengths, target_lengths)

# Handler for keyboard events
def check_for_pause():
    global PAUSE_TRAINING, STOP_TRAINING, END_EPOCH
    if keyboard.is_pressed('p'):
        PAUSE_TRAINING = not PAUSE_TRAINING
        if PAUSE_TRAINING:
            print("\nTraining paused. Press 'p' to resume, 's' to stop, or 'e' to end current epoch")
        else:
            print("\nTraining resumed")
        # Small delay to prevent multiple toggles from a single press
        time.sleep(0.5)
    
    if keyboard.is_pressed('s'):
        STOP_TRAINING = True
        print("\nStopping training after this batch...")
        time.sleep(0.5)
    
    if keyboard.is_pressed('e'):
        END_EPOCH = True
        print("\nEnding current epoch...")
        time.sleep(0.5)
        return True
    
    return False

# Function to save checkpoint
def save_checkpoint(model, task_optimizer, afdm_optimizer, scheduler, epoch, 
                   train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers,
                   best_valid_wer, label_to_idx, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'task_optimizer_state_dict': task_optimizer.state_dict(),
        'afdm_optimizer_state_dict': afdm_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_wers': train_wers,
        'valid_wers': valid_wers,
        'train_cers': train_cers,
        'valid_cers': valid_cers,
        'best_valid_wer': best_valid_wer,
        'label_to_idx': label_to_idx
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

# Function to load checkpoint
def load_checkpoint(filepath, model, task_optimizer, afdm_optimizer, scheduler=None):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return model, task_optimizer, afdm_optimizer, scheduler, 0, [], [], [], [], [], [], float('inf'), {}
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    task_optimizer.load_state_dict(checkpoint['task_optimizer_state_dict'])
    afdm_optimizer.load_state_dict(checkpoint['afdm_optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (model, task_optimizer, afdm_optimizer, scheduler, checkpoint['epoch'],
            checkpoint['train_losses'], checkpoint['valid_losses'],
            checkpoint['train_wers'], checkpoint['valid_wers'],
            checkpoint['train_cers'], checkpoint['valid_cers'],
            checkpoint['best_valid_wer'], checkpoint['label_to_idx'])

# Calculate Word Error Rate (WER) and Character Error Rate (CER)
def calculate_error_rates(predictions, targets, idx_to_label):
    batch_wer = []
    batch_cer = []

    for pred, target in zip(predictions, targets):
        # Convert predicted indices to characters
        pred_str = ''.join([idx_to_label.get(idx, '') for idx in pred if idx > 0])  # Skip blank (0)

        # Ensure target is iterable (handle single-label cases)
        if not hasattr(target, '__iter__'):
            target_seq = [int(target)]
        else:
            target_seq = target

        target_str = ''.join([idx_to_label.get(idx, '') for idx in target_seq])

        # Character Error Rate (CER)
        cer = editdistance.eval(pred_str, target_str) / max(len(target_str), 1)
        batch_cer.append(cer)

        # Word Error Rate (WER)
        pred_words = pred_str.split()
        target_words = target_str.split()
        wer = editdistance.eval(pred_words, target_words) / max(len(target_words), 1)
        batch_wer.append(wer)

    return np.mean(batch_wer), np.mean(batch_cer)
def decode_ctc(output, blank=0):
    """Greedy CTC decoder"""
    # Get the most likely class at each timestep
    _, max_indices = torch.max(output, 2)
    max_indices = max_indices.cpu().numpy()
    
    # Remove duplicates and blanks
    decoded = []
    for batch_idx, indices in enumerate(max_indices):
        current_decoded = []
        prev_idx = -1
        
        for idx in indices:
            if idx != blank and idx != prev_idx:
                current_decoded.append(idx)
            prev_idx = idx
        
        decoded.append(current_decoded)
    
    return decoded

# Function to train model with alternating optimization
def train_model(model, train_loader, valid_loader, criterion, task_optimizer, afdm_optimizer, 
                scheduler, num_epochs, num_classes, label_to_idx, start_epoch=0, 
                train_losses=None, valid_losses=None, train_wers=None, valid_wers=None,
                train_cers=None, valid_cers=None, best_valid_wer=float('inf'),
                early_stop_patience=5, pretrain_iterations=10000, afdm_init_iterations=500):
    
    global PAUSE_TRAINING, STOP_TRAINING, END_EPOCH
    
    # Create idx_to_label mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Initialize lists if not provided
    if train_losses is None:
        train_losses = []
    if valid_losses is None:
        valid_losses = []
    if train_wers is None:
        train_wers = []
    if valid_wers is None:
        valid_wers = []
    if train_cers is None:
        train_cers = []
    if valid_cers is None:
        valid_cers = []
    
    early_stop_counter = 0
    iteration_count = 0
    
    print("\nTraining controls:")
    print("  Press 'p' to pause/resume training")
    print("  Press 's' to stop training completely")
    print("  Press 'e' to end the current epoch\n")
    
    try:
        # Pre-training phase if starting from scratch
        if start_epoch == 0 and pretrain_iterations > 0:
            print(f"Starting pre-training for {pretrain_iterations} iterations...")
            model.train()
            pretrain_loss = 0.0
            
            # Temporarily disable AFDM by setting apply_deformation=False
            for i, (images, labels) in enumerate(tqdm(train_loader, desc="Pre-training task network")):
                if i >= pretrain_iterations:
                    break
                    
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass without deformation
                outputs = model(images, apply_deformation=False)
                
                # For simplicity, use all time steps and full sequence length
                batch_size, seq_len, _ = outputs.size()
                inputs_length = torch.full((batch_size,), seq_len, dtype=torch.long)
                targets_length = torch.full((batch_size,), 1, dtype=torch.long)  # Assuming single character labels
                
                # Apply log softmax for CTC loss
                log_probs = F.log_softmax(outputs, dim=2)
                
                # Calculate loss
                loss = criterion(log_probs.transpose(0, 1), labels, inputs_length, targets_length)
                
                # Backward and optimize (task network only)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
                
                pretrain_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    print(f"Pre-training iteration {i+1}/{pretrain_iterations}, Loss: {pretrain_loss / (i+1):.4f}")
            
            print("Pre-training completed.")
            
            # Initialize AFDM if specified
            if afdm_init_iterations > 0:
                print(f"Initializing AFDM for {afdm_init_iterations} iterations...")
                model.train()
                
                for i, (images, labels) in enumerate(tqdm(train_loader, desc="Initializing AFDM")):
                    if i >= afdm_init_iterations:
                        break
                        
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass with deformation
                    outputs = model(images, apply_deformation=True)
                    
                    # For simplicity, use all time steps and full sequence length
                    batch_size, seq_len, _ = outputs.size()
                    inputs_length = torch.full((batch_size,), seq_len, dtype=torch.long)
                    targets_length = torch.full((batch_size,), 1, dtype=torch.long)  # Assuming single character labels
                    
                    # Apply log softmax for CTC loss
                    log_probs = F.log_softmax(outputs, dim=2)
                    
                    # Calculate loss (we want to maximize this for AFDM initialization)
                    loss = -criterion(log_probs.transpose(0, 1), labels, inputs_length, targets_length)
                    
                    # Backward and optimize (AFDM only)
                    afdm_optimizer.zero_grad()
                    loss.backward()
                    afdm_optimizer.step()
                    
                    if (i + 1) % 50 == 0:
                        print(f"AFDM init iteration {i+1}/{afdm_init_iterations}, Loss: {-loss.item():.4f}")
                
                print("AFDM initialization completed.")
        
        # Main training loop
        for epoch in range(start_epoch, num_epochs):
            if STOP_TRAINING:
                print("Training stopped by user.")
                break
                
            start_time = time.time()
            END_EPOCH = False
            
            # Training
            model.train()
            train_loss = 0.0
            epoch_decoded_predictions = []
            epoch_labels = []
            
            for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
                # Check if we should pause
                while PAUSE_TRAINING and not STOP_TRAINING:
                    time.sleep(0.1)  # Sleep to reduce CPU usage while paused
                    check_for_pause()  # This also handles the resume action
                
                if STOP_TRAINING or END_EPOCH:
                    break
                
                check_for_pause()  # Check for keyboard input
                
                images = images.to(device)
                labels_tensor = labels.to(device)
                
                # Randomly choose whether to apply deformation (50% of batches)
                apply_deformation = (torch.rand(1).item() > 0.5)
                
                # Alternating optimization between task network and AFDM
                # First update the task network (minimize recognition loss)
                task_optimizer.zero_grad()
                outputs = model(images, apply_deformation=apply_deformation)
                
                # For CTC loss
                batch_size, seq_len, _ = outputs.size()
                inputs_length = torch.full((batch_size,), seq_len, dtype=torch.long)
                targets_length = torch.full((batch_size,), 1, dtype=torch.long)  # Assuming single character labels
                
                # Apply log softmax for CTC loss
                log_probs = F.log_softmax(outputs, dim=2)
                
                # Calculate task loss
                task_loss = criterion(log_probs.transpose(0, 1), labels_tensor, inputs_length, targets_length)
                
                # Backward and optimize task network
                task_loss.backward()
                task_optimizer.step()
                
                # Then update the AFDM (maximize recognition loss)
                if apply_deformation:  # Only update AFDM when it was applied
                    afdm_optimizer.zero_grad()
                    outputs = model(images, apply_deformation=True)
                    
                    # Apply log softmax for CTC loss
                    log_probs = F.log_softmax(outputs, dim=2)
                    
                    # Calculate adversarial loss (negative task loss)
                    adv_loss = -criterion(log_probs.transpose(0, 1), labels_tensor, inputs_length, targets_length)
                    
                    # Backward and optimize AFDM
                    adv_loss.backward()
                    afdm_optimizer.step()
                
                # Track total loss (using task loss)
                train_loss += task_loss.item() * images.size(0)
                
                # Decode outputs for WER/CER calculation
                decoded_predictions = decode_ctc(log_probs.detach())
                epoch_decoded_predictions.extend(decoded_predictions)
                epoch_labels.extend(labels.numpy())
                
                iteration_count += 1
            
            if not STOP_TRAINING:  # Only process if we didn't stop mid-epoch
                # Calculate training metrics
                train_loss = train_loss / len(train_loader.dataset)
                train_wer, train_cer = calculate_error_rates(epoch_decoded_predictions, epoch_labels, idx_to_label)
                
                train_losses.append(train_loss)
                train_wers.append(train_wer)
                train_cers.append(train_cer)
                
                # Validation
                model.eval()
                valid_loss = 0.0
                epoch_decoded_predictions = []
                epoch_labels = []
                
                with torch.no_grad():
                    for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]"):
                        images = images.to(device)
                        labels_tensor = labels.to(device)
                        
                        # Forward pass without deformation for validation
                        outputs = model(images, apply_deformation=False)
                        
                        # Apply log softmax for CTC loss
                        log_probs = F.log_softmax(outputs, dim=2)
                        
                        # Calculate validation loss
                        valid_batch_loss = criterion(log_probs.transpose(0, 1), labels_tensor, inputs_length, targets_length)
                        valid_loss += valid_batch_loss.item() * images.size(0)
                        
                        # Decode outputs for WER/CER calculation
                        decoded_predictions = decode_ctc(log_probs)
                        epoch_decoded_predictions.extend(decoded_predictions)
                        epoch_labels.extend(labels.numpy())
                
                # Calculate validation metrics
                valid_loss = valid_loss / len(valid_loader.dataset)
                valid_wer, valid_cer = calculate_error_rates(epoch_decoded_predictions, epoch_labels, idx_to_label)
                
                valid_losses.append(valid_loss)
                valid_wers.append(valid_wer)
                valid_cers.append(valid_cer)
                
                # Update scheduler if provided
                if scheduler:
                    scheduler.step(valid_wer)  # Assuming we want to optimize for WER
                
                # Print metrics
                epoch_time = time.time() - start_time
                print(f'\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s')
                print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
                print(f'Train WER: {train_wer:.4f}, Valid WER: {valid_wer:.4f}')
                print(f'Train CER: {train_cer:.4f}, Valid CER: {valid_cer:.4f}')
                
                # Save checkpoint if this is the best model so far
                if valid_wer < best_valid_wer:
                    print(f'Validation WER improved from {best_valid_wer:.4f} to {valid_wer:.4f}')
                    best_valid_wer = valid_wer
                    save_checkpoint(model, task_optimizer, afdm_optimizer, scheduler, epoch, 
                                 train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers,
                                 best_valid_wer, label_to_idx, 'best_model.pth')
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f'Validation WER did not improve. Early stopping counter: {early_stop_counter}/{early_stop_patience}')
                
                # Regular checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    save_checkpoint(model, task_optimizer, afdm_optimizer, scheduler, epoch, 
                                 train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers,
                                 best_valid_wer, label_to_idx, f'checkpoint_epoch_{epoch+1}.pth')
                
                # Early stopping
                if early_stop_counter >= early_stop_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
        # Plot training curves
        if len(train_losses) > 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(valid_losses, label='Valid Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(train_wers, label='Train WER')
            plt.plot(valid_wers, label='Valid WER')
            plt.title('Word Error Rate')
            plt.xlabel('Epoch')
            plt.ylabel('WER')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(train_cers, label='Train CER')
            plt.plot(valid_cers, label='Valid CER')
            plt.title('Character Error Rate')
            plt.xlabel('Epoch')
            plt.ylabel('CER')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_curves.png')
            plt.close()
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
        
    except Exception as e:
        print(f"Error during training: {e}")
        
    finally:
        # Save final model regardless of how training ended
        save_checkpoint(model, task_optimizer, afdm_optimizer, scheduler, epoch, 
                     train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers,
                     best_valid_wer, label_to_idx, 'final_model.pth')
        
        return model, train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers

# Collate function for variable width images
def collate_fn(batch):
    # Sort batch by decreasing width
    batch.sort(key=lambda x: x[0].shape[2], reverse=True)
    images, labels = zip(*batch)
    
    # Stack images with padding
    max_width = max(img.shape[2] for img in images)
    padded_images = []
    
    for img in images:
        c, h, w = img.shape
        padded_img = torch.zeros(c, h, max_width)
        padded_img[:, :, :w] = img
        padded_images.append(padded_img)
    
    images = torch.stack(padded_images)
    
    return images, torch.tensor(labels)



def main(args):
    global device, STOP_TRAINING
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup for training mode
    STOP_TRAINING = False
    
    # Data preparation
    image_paths, labels, label_to_idx = prepare_data(args.data_dir)
    num_classes = len(label_to_idx)
    
    print(f"Found {len(image_paths)} images with {num_classes} unique classes")
    
    # Split data
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=args.seed, stratify=labels)
    
    # Data augmentation and transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CharacterDataset(
        train_paths, train_labels, target_height=image_height, transform=transform)
    valid_dataset = CharacterDataset(
        valid_paths, valid_labels, target_height=image_height, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    # Initialize model
    model = CRNN(num_classes=num_classes + 1).to(device)  # +1 for blank in CTC
    
    # Initialize optimizers
    task_optimizer = optim.Adam([
        {'params': model.cnn_stage1.parameters()},
        {'params': model.cnn_stage2.parameters()},
        {'params': model.cnn_stage3_1.parameters()},
        {'params': model.cnn_stage3_2.parameters()},
        {'params': model.cnn_stage4_1.parameters()},
        {'params': model.cnn_stage4_2.parameters()},
        {'params': model.cnn_stage5_1.parameters()},
        {'params': model.cnn_stage5_2.parameters()},
        {'params': model.map_to_seq.parameters()},
        {'params': model.rnn.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=learning_rate_task)
    
    afdm_optimizer = optim.Adam(model.afdm.parameters(), lr=learning_rate_afdm)
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        task_optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Initialize criterion
    criterion = CTCLoss(blank=0)
    
    # Load checkpoint if available
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model, task_optimizer, afdm_optimizer, scheduler, start_epoch, \
        train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers, \
        best_valid_wer, loaded_label_to_idx = load_checkpoint(
            args.checkpoint, model, task_optimizer, afdm_optimizer, scheduler)
        
        # Use loaded label_to_idx if available
        if loaded_label_to_idx:
            label_to_idx = loaded_label_to_idx
    else:
        start_epoch = 0
        train_losses, valid_losses = [], []
        train_wers, valid_wers = [], []
        train_cers, valid_cers = [], []
        best_valid_wer = float('inf')
    
    # Train model
    model, train_losses, valid_losses, train_wers, valid_wers, train_cers, valid_cers = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        task_optimizer=task_optimizer,
        afdm_optimizer=afdm_optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        num_classes=num_classes,
        label_to_idx=label_to_idx,
        start_epoch=start_epoch,
        train_losses=train_losses,
        valid_losses=valid_losses,
        train_wers=train_wers,
        valid_wers=valid_wers,
        train_cers=train_cers,
        valid_cers=valid_cers,
        best_valid_wer=best_valid_wer,
        early_stop_patience=args.patience,
        pretrain_iterations=args.pretrain_iters,
        afdm_init_iterations=args.afdm_init_iters
    )
    
    print("Training completed!")
    
 

# Test function
def test_model(model_path, test_dir, label_to_idx=None):
    # Load model and label_to_idx
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get label_to_idx from checkpoint if not provided
    if label_to_idx is None:
        label_to_idx = checkpoint['label_to_idx']
    
    num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Initialize model
    model = CRNN(num_classes=num_classes + 1).to(device)  # +1 for blank in CTC
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare test data
    image_paths, labels, _ = prepare_data(test_dir)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    test_dataset = CharacterDataset(
        image_paths, labels, target_height=image_height, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Evaluate
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    criterion = CTCLoss(blank=0)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels_tensor = labels.to(device)
            
            # Forward pass without deformation for testing
            outputs = model(images, apply_deformation=False)
            
            # For CTC loss
            batch_size, seq_len, _ = outputs.size()
            inputs_length = torch.full((batch_size,), seq_len, dtype=torch.long)
            targets_length = torch.full((batch_size,), 1, dtype=torch.long)  # Assuming single character labels
            
            # Apply log softmax for CTC loss
            log_probs = F.log_softmax(outputs, dim=2)
            
            # Calculate test loss
            batch_loss = criterion(log_probs.transpose(0, 1), labels_tensor, inputs_length, targets_length)
            test_loss += batch_loss.item() * images.size(0)
            
            # Decode outputs for WER/CER calculation
            decoded_predictions = decode_ctc(log_probs)
            all_predictions.extend(decoded_predictions)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_wer, test_cer = calculate_error_rates(all_predictions, all_labels, idx_to_label)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test WER: {test_wer:.4f}")
    print(f"Test CER: {test_cer:.4f}")
    
    return test_loss, test_wer, test_cer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRNN with AFDM for Character Recognition")
    parser.add_argument("--data_dir", type=str, default="./Test Chars", help="Directory containing the dataset")
    parser.add_argument("--epochs", type=int, default=num_epochs, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--pretrain_iters", type=int, default=10000, help="Number of iterations for pre-training")
    parser.add_argument("--afdm_init_iters", type=int, default=500, help="Number of iterations for AFDM initialization")
    parser.add_argument("--test", action="store_true", help="Test model after training")
    parser.add_argument("--test_dir", type=str, default="./test_data", help="Directory containing test data")
    
    args = parser.parse_args()
    main(args)
    
    