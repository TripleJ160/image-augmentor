# Image Augmentor

A Python script for augmenting medication images using the Albumentations library. This tool helps create multiple variations of medication images for training machine learning models.

## Features

- Processes images in medication-specific folders
- Creates 10 augmented versions of each image
- Applies various augmentations:
  - Random rotation (±15 degrees)
  - Random brightness and contrast adjustments
  - Gaussian noise
  - Horizontal flips
  - Resizing to 224x224 pixels
- Organizes augmented images in dedicated folders

## Requirements

- Python 3.x
- OpenCV (cv2)
- Albumentations
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/TripleJ160/image-augmentor.git
cd image-augmentor

# Install required packages
pip install opencv-python albumentations numpy
```

## Usage

1. Place your medication images in folders under the "Medication Photos" directory:
```
Medication Photos/
├── Aspirin/
│   ├── aspirin_1.jpg
│   └── aspirin_2.jpg
└── Atorvastatin/
    ├── atorvastatin_1.jpg
    └── atorvastatin_2.jpg
```

2. Run the script:
```bash
python main.py
```

3. Find augmented images in "_augmented" folders:
```
Medication Photos/
├── Aspirin/
│   ├── aspirin_1_augmented/
│   │   ├── aspirin_1_aug_0.jpg
│   │   └── ...
│   └── aspirin_2_augmented/
└── Atorvastatin/
    ├── atorvastatin_1_augmented/
    └── atorvastatin_2_augmented/
```

## License

MIT License