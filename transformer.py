import torch
from torchvision import transforms
from PIL import Image
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=20)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()


class_names = ['Abbotts Babbler', 'Abbotts Booby', 'Abyssinian Ground hornbill', 'African Crowned Crane',
               'African Emerald Cuckoo', 'African Firefinch', 'African Oyster Catcher', 'African Pied Hornbill',
               'African Pygmy Goose', 'Albatross', 'Alberts Towhee', 'Alexandrine Parakeet', 'Alpine Chough',
               'Altamira Yellowthroat', 'American Avocet', 'American Bittern', 'American Coot',
               'American Flamingo', 'American Goldfinch', 'American Kestrel']


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_species(image_path, model, transform, class_names, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    species = class_names[predicted.item()]
    return species


image_path = r"d:\Major Project\Abbots Babbler.jpg"  
species = predict_species(image_path, model, transform, class_names, device)
print(f"Predicted Species: {species}")
