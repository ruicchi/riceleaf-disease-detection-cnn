"""
Predict rice leaf disease from a single image.
"""

import sys
import torch
from torchvision import transforms
from PIL import Image
from model import RiceLeafCNN


# ──────────────────────────────────────
# Configuration
# ──────────────────────────────────────
IMAGE_SIZE = 128
MODEL_PATH = "rice_leaf_cnn.pth"
CLASS_NAMES = ["Brown Spot", "Healthy", "Hispa", "Leaf Blast"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained CNN model."""
    model = RiceLeafCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess a single image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(image_path):
    """Predict the disease class for a rice leaf image."""
    model = load_model()
    image_tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_pct = confidence.item() * 100

    print("\n🌾 Rice Leaf Disease Prediction")
    print("=" * 40)
    print(f"  Image:      {image_path}")
    print(f"  Prediction: {predicted_class}")
    print(f"  Confidence: {confidence_pct:.2f}%")
    print()
    print("  All probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[0][i].item() * 100
        bar = "█" * int(prob / 5)
        print(f"    {class_name:<30s} {prob:6.2f}%  {bar}")

    return predicted_class, confidence_pct


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_images/leaf01.jpg")
        sys.exit(1)

    predict(sys.argv[1])