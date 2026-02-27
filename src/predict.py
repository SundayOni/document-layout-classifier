"""
predict.py — Run inference on a new document image using the trained ResNet18 model.

Usage:
    python src/predict.py --image path/to/your/image.png
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Class labels
CLASSES = ['email', 'resume', 'scientific_publication']

# Image transform — same as validation transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    """Load the trained ResNet18 model from a saved checkpoint."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path, model):
    """Predict the document class for a single image."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    print(f'\nImage: {os.path.basename(image_path)}')
    print(f'Predicted class: {CLASSES[predicted_class]}')
    print(f'\nConfidence scores:')
    for cls, prob in zip(CLASSES, probabilities):
        bar = '█' * int(prob * 30)
        print(f'  {cls:<25} {prob:.4f}  {bar}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify a document image')
    parser.add_argument('--image', required=True, help='Path to document image')
    parser.add_argument('--model', default='models/resnet18_document_classifier.pth',
                        help='Path to trained model weights')
    args = parser.parse_args()

    model = load_model(args.model)
    predict(args.image, model)