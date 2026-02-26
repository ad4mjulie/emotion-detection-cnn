import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Load the Haar Cascade for face detection (standard in OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """Detects faces in a frame and returns bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def preprocess_face(frame, face_coords):
    """
    Crops, grayscales, and normalizes a face ROI for CNN input.
    Expected input: 48x48 grayscale tensor.
    """
    x, y, w, h = face_coords
    roi = frame[y:y+h, x:x+w]
    
    # Preprocessing pipeline
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Convert to NumPy float32 and normalize
    roi_norm = roi_resized.astype('float32') / 255.0
    
    # Convert to PyTorch Tensor: (Channels, Height, Width)
    roi_tensor = torch.from_numpy(roi_norm).unsqueeze(0).unsqueeze(0)
    
    return roi_tensor

def predict_emotion(model, face_tensor, device, labels):
    """Runs inference on a face tensor and returns the label and confidence."""
    face_tensor = face_tensor.to(device)
    
    with torch.no_grad():
        output = model(face_tensor)
        probs = F.softmax(output, dim=1)
        
        # Get highest probability
        conf, pred = torch.max(probs, 1)
        
    return labels[pred.item()], conf.item()

def draw_results(frame, face_coords, label, confidence):
    """Draws bounding boxes and labels on the frame."""
    x, y, w, h = face_coords
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw label and confidence
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame
