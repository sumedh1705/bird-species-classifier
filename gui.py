import torch
from torchvision import transforms
from PIL import Image
import timm
import tkinter as tk
from tkinter import filedialog, messagebox
import torch.nn.functional as F


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


def predict_species(image_path, model, transform, class_names, device, threshold=0.6):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        if confidence.item() < threshold:
            return "Bird not found"
        else:
            return class_names[predicted.item()]


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            species = predict_species(file_path, model, transform, class_names, device)
            messagebox.showinfo("Prediction", f"Predicted Species: {species}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {str(e)}")


root = tk.Tk()
root.title("Bird Species Classifier")
root.geometry("300x150")

btn = tk.Button(root, text="Select Bird Image", command=select_image, bg="green", fg="white", font=("Arial", 12, "bold"))
btn.pack(expand=True, padx=20, pady=40)

root.mainloop()
