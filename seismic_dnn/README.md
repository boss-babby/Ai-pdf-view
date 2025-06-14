# Seismic DNN Model

This project implements a deep neural network model for processing and generating seismic images from limited data. The model can reconstruct high-resolution seismic images from sparse data points and track seismic patterns.

## Features

- Data preprocessing and augmentation for seismic signals
- Two model architectures:
  - Autoencoder for dimensionality reduction and feature extraction
  - U-Net for high-resolution image generation
- Real-time seismic data processing
- Spectrogram generation
- Training visualization

## Project Structure

```
seismic_dnn/
├── data/               # Data directory for seismic files
├── src/               
│   ├── models.py      # Neural network model architectures
│   ├── data_processing.py  # Data processing utilities
│   └── train.py       # Training script
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your seismic data files in the `data/` directory
2. Run the training script:
```bash
python src/train.py
```

## Model Architecture

### SeismicAutoencoder
- Reduces input dimensionality while preserving important features
- Encoder: Input → 512 → 256 → 128 (latent space)
- Decoder: 128 → 256 → 512 → Output
- Uses batch normalization and dropout for regularization

### SeismicUNet
- Generates high-resolution seismic images
- Encoder path with 4 downsampling steps
- Decoder path with 4 upsampling steps
- Skip connections for preserving spatial information
- Suitable for image-to-image translation tasks

## Data Processing

The project includes various data processing utilities:
- Windowing and stride operations
- Bandpass filtering
- Normalization
- Spectrogram generation
- Real-time data augmentation

## Contributing

Feel free to submit issues and enhancement requests!
