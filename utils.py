import cv2
import numpy as np
import torch
import torch.nn.functional as F

import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_faces(frame):
    """Detects faces in a frame using MediaPipe and returns bounding boxes (x, y, w, h)."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    faces = []
    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            faces.append((x, y, width, height))
            
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
    # Consistent with training: (x / 255.0 - 0.5) / 0.5
    roi_norm = (roi_resized.astype('float32') / 255.0 - 0.5) / 0.5
    
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
