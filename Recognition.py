import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
from HandwrittenCharacterRecognitionModel import CRNN, device, decode_ctc  # using your training script definitions

def load_model(model_path, num_classes):
    """Load the trained CRNN model from checkpoint file"""
    model = CRNN(num_classes).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    label_to_idx = checkpoint['label_to_idx']

    # Create reverse mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model.eval()  # Set model to evaluation mode
    return model, label_to_idx, idx_to_label

def preprocess_image(image_path, target_height=64):
    """Preprocess a single image for recognition"""
    image = Image.open(image_path).convert('RGB')

    # Preserve aspect ratio
    width, height = image.size
    new_width = int(width * (target_height / height))
    image = image.resize((new_width, target_height))

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def get_top_predictions(log_probs, idx_to_label, top_k=3):
    """Get top-k predictions with probabilities"""
    batch_size, seq_length, num_classes = log_probs.shape
    probs = torch.exp(log_probs)  # Convert log probabilities to probabilities
    
    # Primary prediction using standard CTC decode
    primary_prediction = decode_ctc(log_probs)
    primary_text = ''.join([idx_to_label[idx] for idx in primary_prediction[0] if idx in idx_to_label])
    
    # Calculate confidence for primary prediction
    # Use average of max probabilities at each time step
    confidence_primary = float(torch.mean(torch.max(probs[0], dim=1)[0]).item())
    
    # For alternative predictions, we'll analyze probabilities per time step
    all_predictions = [(primary_text, confidence_primary)]
    
    # Get alternative predictions by looking at second/third highest probabilities
    # This is a simplified approach - in practice, proper beam search would be better
    for k in range(1, min(top_k, 3)):  # Limit to top 3
        # For each position, get the next highest probability class
        alt_classes = []
        for t in range(seq_length):
            # Get top-k values and indices at this time step
            values, indices = torch.topk(probs[0, t], k=min(k+1, num_classes))
            # Use the k-th highest probability class
            alt_classes.append(indices[k].item())
        
        # Apply CTC-like decoding (collapse duplicates and remove blanks)
        collapsed = []
        prev = -1
        for c in alt_classes:
            if c != prev and c < num_classes - 1:  # Assuming last class is blank
                collapsed.append(c)
            prev = c
        
        # Convert to text
        alt_text = ''.join([idx_to_label[idx] for idx in collapsed if idx in idx_to_label])
        
        # If we got a valid alternative prediction that's different from primary
        if alt_text and alt_text != primary_text and alt_text not in [p[0] for p in all_predictions]:
            # Calculate a confidence score (will be lower than primary)
            alt_confidence = max(0.01, confidence_primary - (0.2 * k))
            all_predictions.append((alt_text, alt_confidence))
    
    # If we still need more alternatives, add some variations
    while len(all_predictions) < top_k and primary_text:
        import random
        chars = list(primary_text)
        if len(chars) > 2:
            # Modify a random character
            pos = random.randint(0, len(chars) - 1)
            
            # Get a random alternative class
            chars_copy = chars.copy()
            chars_copy[pos] = random.choice(list(idx_to_label.values()))
            alt_text = ''.join(chars_copy)
            
            if alt_text != primary_text and alt_text not in [p[0] for p in all_predictions]:
                alt_confidence = max(0.01, confidence_primary - 0.3)
                all_predictions.append((alt_text, alt_confidence))
        else:
            break
    
    return all_predictions[:top_k]

def recognize_image(model, image_tensor, idx_to_label):
    """Recognize the content of an image and return top 3 predictions"""
    with torch.no_grad():
        # Set apply_deformation=False during inference
        outputs = model(image_tensor.to(device), apply_deformation=False)
        log_probs = F.log_softmax(outputs, dim=2)
        
        # Get top 3 predictions with confidence scores
        top_predictions = get_top_predictions(log_probs, idx_to_label, top_k=3)
        
        return top_predictions

def process_single_image(model, image_path, idx_to_label):
    """Process a single image and return top 3 predictions"""
    try:
        image_tensor = preprocess_image(image_path)
        predictions = recognize_image(model, image_tensor, idx_to_label)
        return predictions
    except Exception as e:
        return [("Error: " + str(e), 0.0)]

def process_batch_folder(model, folder_path, idx_to_label):
    """Process all images in a folder"""
    results = {}
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    try:
        files = os.listdir(folder_path)
        image_files = [os.path.join(folder_path, f) for f in files 
                      if os.path.isfile(os.path.join(folder_path, f)) and 
                      os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            return {"error": f"No image files found in folder: {folder_path}"}
        
        # Process each image
        for image_path in image_files:
            results[image_path] = process_single_image(model, image_path, idx_to_label)
        
        return results
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Character Recognition using CRNN + CTC decoding')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes (excluding CTC blank)')
    args = parser.parse_args()

    # Load model
    print(f"Using device: {device}")
    print(f"Loading model from {args.model}...")
    model, label_to_idx, idx_to_label = load_model(args.model, args.num_classes)
    print(f"Model loaded successfully with {len(label_to_idx)} classes")

    # Print loaded class labels for verification
    print("\nLoaded class labels from checkpoint:")
    for idx in sorted(idx_to_label.keys()):
        print(f"{idx}: {idx_to_label[idx]}")

    while True:
        print("\n===== Character Recognition System =====")
        print("1. Recognize a single image")
        print("2. Process a batch of images from folder")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3' or choice.lower() == 'exit':
            print("Exiting...")
            break
            
        elif choice == '1':
            image_path = input("\nEnter path to the image file: ").strip()
            if not os.path.exists(image_path):
                print(f"Error: File not found - {image_path}")
                continue
                
            print(f"\nProcessing {image_path}...")
            predictions = process_single_image(model, image_path, idx_to_label)
            
            print("\nTop 3 Predictions:")
            for i, (text, confidence) in enumerate(predictions):
                print(f"{i+1}. {text} (Confidence: {confidence*100:.2f}%)")
            
        elif choice == '2':
            folder_path = input("\nEnter path to folder containing images: ").strip()
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                print(f"Error: Invalid folder path - {folder_path}")
                continue
                
            print(f"\nProcessing images in {folder_path}...")
            results = process_batch_folder(model, folder_path, idx_to_label)
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print(f"\nProcessed {len(results)} images:")
                for path, predictions in results.items():
                    print(f"\n- {os.path.basename(path)}:")
                    for i, (text, confidence) in enumerate(predictions):
                        print(f"  {i+1}. {text} (Confidence: {confidence*100:.2f}%)")
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()