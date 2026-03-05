# CNN Emotion Detection Project

A fully offline Python project that detects human faces and classifies their emotions using a Convolutional Neural Network (CNN) built with PyTorch.

## Features
- **High-Quality Face Detection**: Uses MediaPipe Face Detection for superior accuracy compared to standard Haar Cascades.
- **Enhanced Training Pipeline**: Supports class-weighted training, OneCycleLR learning rate schedule, and advanced data augmentation.
- **Detailed Metrics**: Tracks Macro-F1, per-class precision/recall/F1, and validation loss.
- **Real-time Detection**: Live webcam feed with bounding boxes and emotion labels.
- **Offline Support**: Works entirely without an internet connection once dependencies and models are loaded.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Training (Optional)
This project is designed to run with `emotion_model.pth`. If you want to train from scratch or fine-tune:
1. Ensure your dataset is in the `archive/` folder with `train` and `test` subdirectories.
2. Run the training script:
   ```bash
   python train.py
   ```
The script will save the best model based on **Macro-F1** score to ensure robustness against class imbalance.

## Usage

### Live Webcam Mode
To start real-time emotion detection:
```bash
source .venv/bin/activate && python main.py
```
- Press **'q'** to exit the webcam window.

### Single Image Mode
To detect emotions in a specific image:
```bash
python main.py --image path/to/your/image.jpg
```

## Project Structure
- `main.py`: Entry point for webcam and image inference.
- `model.py`: Advanced CNN architecture with Residual Blocks.
- `model_loader.py`: Handles model initialization and weight loading.
- `utils.py`: Contains logic for face detection (MediaPipe), preprocessing (48x48 normalization), and drawing.
- `train.py`: Enhanced script to train/fine-tune the CNN.

## Recent Improvements
- **Face Detection**: Migrated from OpenCV Haar Cascades to **MediaPipe** for better multi-angle and lighting support.
- **Optimization**: Implemented **AdamW** and **OneCycleLR** for faster and more stable convergence.
- **Data Augmentation**: Added RandomAffine, ColorJitter, and RandomErasing to reduce overfitting.
