import argparse

import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=20)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()


class_names = [
    "Abbotts Babbler",
    "Abbotts Booby",
    "Abyssinian Ground hornbill",
    "African Crowned Crane",
    "African Emerald Cuckoo",
    "African Firefinch",
    "African Oyster Catcher",
    "African Pied Hornbill",
    "African Pygmy Goose",
    "Albatross",
    "Alberts Towhee",
    "Alexandrine Parakeet",
    "Alpine Chough",
    "Altamira Yellowthroat",
    "American Avocet",
    "American Bittern",
    "American Coot",
    "American Flamingo",
    "American Goldfinch",
    "American Kestrel",
]


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_species(image_path: str, *, threshold: float = 0.0) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    conf = float(confidence.item())
    if conf < threshold:
        return "Bird not found", conf
    return class_names[predicted.item()], conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird species classification (PyTorch DeiT).")
    parser.add_argument("--image", required=True, help="Path to an input image (jpg/png).")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold (default: 0.6).")
    args = parser.parse_args()

    species, confidence = predict_species(args.image, threshold=args.threshold)
    print(f"Predicted Species: {species} (confidence={confidence:.3f})")

