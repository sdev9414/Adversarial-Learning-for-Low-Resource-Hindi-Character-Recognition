import os
import random
import shutil

def move_random_images(source_folder, destination_folder, num_images=100):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through each subfolder in the source folder
    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            print(f"\nProcessing subfolder: {subfolder}")
            
            # Get all image files (adjust extensions as needed)
            image_files = [f for f in os.listdir(subfolder_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Randomly select up to 'num_images'
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            # Create matching subfolder in destination
            destination_subfolder = os.path.join(destination_folder, subfolder)
            if not os.path.exists(destination_subfolder):
                os.makedirs(destination_subfolder)
                print(f"Created destination folder: {destination_subfolder}")

            # Move each selected image
            for image in selected_images:
                src_image_path = os.path.join(subfolder_path, image)
                dst_image_path = os.path.join(destination_subfolder, image)
                try:
                    shutil.move(src_image_path, dst_image_path)
                    print(f"Moved: {src_image_path} -> {dst_image_path}")
                except Exception as e:
                    print(f"Failed to move {src_image_path} -> {dst_image_path}: {e}")
            
            print(f"Moved {len(selected_images)} images from {subfolder}.")

# --------------------------
# Usage (update these paths)
# --------------------------
source_folder = r'E:\FinalSubmission\CharacterName'
destination_folder = r'E:\FinalSubmission\Test Chars'

move_random_images(source_folder, destination_folder)
