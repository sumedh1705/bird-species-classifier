import argparse
import json
import os
import warnings

import absl.logging
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)


def load_labels(labels_path: str) -> dict:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_bird_species(*, model, labels: dict, image_path: str, image_size: int = 224) -> str:
    img = load_img(image_path, target_size=(image_size, image_size))
    img = img_to_array(img) / 255.0
    img = img.reshape(1, image_size, image_size, 3)
    prediction = model.predict(img, verbose=0)
    class_index = int(prediction.argmax(axis=-1)[0])
    return labels.get(str(class_index), f"UNKNOWN_CLASS_{class_index}")


def main():
    parser = argparse.ArgumentParser(description="Bird species classification (TensorFlow/Keras).")
    parser.add_argument("--image", required=True, help="Path to an input image (jpg/png).")
    parser.add_argument("--model", default="bird_species_model.h5", help="Path to .h5 model file.")
    parser.add_argument("--labels", default="labels.json", help="Path to labels.json file.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (default: 224).")
    args = parser.parse_args()

    model = load_model(args.model)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    labels = load_labels(args.labels)
    predicted_species = predict_bird_species(
        model=model, labels=labels, image_path=args.image, image_size=args.image_size
    )
    print(f"Predicted Bird Species: {predicted_species}")


if __name__ == "__main__":
    main()
