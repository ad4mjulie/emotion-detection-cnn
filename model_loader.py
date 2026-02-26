import torch
import os
from model import EmotionCNN

def get_emotion_labels():
    """Returns the ordered list of emotion labels matching the CNN output."""
    return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_model(model_path='emotion_model.pth'):
    """
    Loads the EmotionCNN model. 
    If a weight file exists, it loads the weights. 
    Otherwise, it returns a randomly initialized model (useful for pipeline testing).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN(num_classes=7).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        # Use map_location for CPU/GPU portability
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    else:
        print(f"Warning: {model_path} not found. Using randomly initialized model.")
        print("Note: The model will give random predictions until trained.")
        model.eval()
        
    return model, device
