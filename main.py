import sys
from unittest.mock import MagicMock

# Robust mock for _lzma to satisfy the lzma standard library on macOS environments missing the C extension
mock_lzma = MagicMock()
# Constants expected by lzma.py
for name in ['FORMAT_XZ', 'FORMAT_ALONE', 'FORMAT_RAW', 'FORMAT_AUTO', 
            'CHECK_NONE', 'CHECK_CRC32', 'CHECK_CRC64', 'CHECK_SHA256',
            'FILTER_LZMA1', 'FILTER_LZMA2', 'FILTER_DELTA', 'FILTER_X86', 'FILTER_IA64', 
            'FILTER_ARM', 'FILTER_ARMTHUMB', 'FILTER_SPARC', 'FILTER_POWERPC',
            'MF_BT2', 'MF_BT3', 'MF_BT4', 'MF_HC3', 'MF_HC4', 'MODE_READ', 'MODE_WRITE']:
    setattr(mock_lzma, name, 1)
sys.modules['_lzma'] = mock_lzma

import cv2
import argparse
import sys
from model_loader import load_model, get_emotion_labels
from utils import detect_faces, preprocess_face, predict_emotion, draw_results

def run_webcam():
    """Runs live webcam emotion detection."""
    # Initialize model
    model, device = load_model('emotion_model.pth')
    labels = get_emotion_labels()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        faces = detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Preprocess
            face_tensor = preprocess_face(frame, (x, y, w, h))
            
            # Predict
            label, conf = predict_emotion(model, face_tensor, device, labels)
            
            # Draw
            frame = draw_results(frame, (x, y, w, h), label, conf)
            
        # Display results
        cv2.imshow('Emotion Detector (Offline CNN)', frame)
        
        # Check for user quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_image(image_path):
    """Runs emotion detection on a single image file."""
    # Initialize model
    model, device = load_model('emotion_model.pth')
    labels = get_emotion_labels()
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Detect faces
    faces = detect_faces(frame)
    if len(faces) == 0:
        print("No faces detected in the image.")
        
    for (x, y, w, h) in faces:
        # Preprocess
        face_tensor = preprocess_face(frame, (x, y, w, h))
        
        # Predict
        label, conf = predict_emotion(model, face_tensor, device, labels)
        
        # Draw
        frame = draw_results(frame, (x, y, w, h), label, conf)
        
    # Display results
    cv2.imshow('Emotion Detector - Image Result', frame)
    print("Inference complete. Press any key to close window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Emotion Detection Project")
    parser.add_argument("--image", type=str, help="Path to an image file for inference")
    args = parser.parse_args()
    
    if args.image:
        run_image(args.image)
    else:
        run_webcam()
