import cv2
import numpy as np
import os
import albumentations as A
from albumentations.core.composition import OneOf

# Create output directory if it doesn't exist
BASE_DIR = "Medication Photos"

# Define augmentation pipeline
augmentations = A.Compose([
    A.Rotate(limit=15, p=0.7),  # Rotate ±15 degrees
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.3),  # Add noise
    A.HorizontalFlip(p=0.5),
    A.Resize(height=224, width=224, p=1.0),  # Resize to fixed size
])

# Process each medication folder
for med_folder in os.listdir(BASE_DIR):
    med_path = os.path.join(BASE_DIR, med_folder)
    
    # Skip if not a directory
    if not os.path.isdir(med_path):
        continue
        
    print(f"\nProcessing {med_folder} folder...")
    
    # Process each image in the medication folder
    for filename in os.listdir(med_path):
        # Skip Zone.Identifier files and hidden files
        if filename.endswith('.Identifier') or filename.startswith('.'):
            continue
            
        img_path = os.path.join(med_path, filename)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Skipping {filename}, unable to read.")
            continue
        
        # Create augmented folder for this image
        base_name = os.path.splitext(filename)[0]
        aug_folder = os.path.join(med_path, f"{base_name}_augmented")
        os.makedirs(aug_folder, exist_ok=True)
        
        for i in range(10):  # Generate 10 augmented images per original
            augmented = augmentations(image=image)['image']
            output_filename = f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(aug_folder, output_filename), augmented)
        
        print(f"Augmented {filename} -> 10 new images saved in {aug_folder}")

print("\nAugmentation complete. Check the augmented folders inside each medication directory.")
