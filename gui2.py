import torch
from torchvision import transforms
from PIL import Image, ImageTk
import timm
import tkinter as tk
from tkinter import filedialog, Label, Button
from tkinter import messagebox
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


def predict_species(image_path, threshold=0.6):
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


def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        try:
            species, confidence = predict_species(file_path, threshold=threshold_scale.get() / 100.0)
            result_label.config(text=f"Predicted: {species}  (confidence={confidence:.3f})")
            
            img = Image.open(file_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk

        except Exception as e:
            messagebox.showerror("Error", str(e))


window = tk.Tk()
window.title("Bird Species Recognition")
window.geometry("400x400")

btn = Button(window, text="Select Bird Image", command=select_image)
btn.pack(pady=10)

image_label = Label(window)
image_label.pack(pady=10)

threshold_scale = tk.Scale(window, from_=0, to=100, orient="horizontal", label="Confidence threshold (%)")
threshold_scale.set(60)
threshold_scale.pack(pady=5)

result_label = Label(window, text="Predicted Species: ", font=("Arial", 12))
result_label.pack(pady=10)

window.mainloop()
