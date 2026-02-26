# CNN Emotion Detection Project

A fully offline Python project that detects human faces and classifies their emotions using a Convolutional Neural Network (CNN) built with PyTorch.

## Features
- **Real-time Detection**: Live webcam feed with bounding boxes and emotion labels.
- **Image Mode**: Run inference on a single image file.
- **Offline Support**: Works entirely without an internet connection once dependencies and models are loaded.
- **Trainable**: Includes a script to train or fine-tune the model using the FER-2013 dataset.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Training (Optional)
This project is designed to run with `emotion_model.pth`. If you don't have a pretrained weight file:
1. Download the `fer2013.csv` dataset (e.g., from Kaggle).
2. Place it in the project root directory.
3. Run the training script:
   ```bash
   python train.py
   ```
This will save the best model weights as `emotion_model.pth`.

## Usage

### Live Webcam Mode
To start real-time emotion detection:
```bash
python main.py
```
- Press **'q'** to exit the webcam window.

### Single Image Mode
To detect emotions in a specific image:
```bash
python main.py --image path/to/your/image.jpg
```

## Project Structure
- `main.py`: Entry point for webcam and image inference.
- `model.py`: PyTorch architecture of the CNN.
- `model_loader.py`: Handles model initialization and weight loading.
- `utils.py`: Contains logic for face detection (OpenCV Haar Cascade), preprocessing (48x48 grayscale), and drawing results.
- `train.py`: Script to train the CNN on the FER-2013 dataset.

## Extending the Project
- **New Emotions**: Update the labels in `model_loader.py` and the `num_classes` parameter in `model.py`.
- **Better Detection**: Replace the Haar Cascade in `utils.py` with a more advanced detector like MTCNN or MediaPipe for improved robustness.
