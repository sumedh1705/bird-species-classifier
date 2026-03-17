# Bird Species Classifier

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIHCSP63227.2024.10959970-blue)](https://doi.org/10.1109/IHCSP63227.2024.10959970)
![Python](https://img.shields.io/badge/Python-3.7%2B-informational)
![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-brightgreen)

This repository contains a bird species classification project with two runnable inference demos:

- A TensorFlow/Keras command-line classifier based on a VGG16 transfer learning pipeline.
- A PyTorch desktop GUI based on a DeiT model for a smaller 20-class demo.

The repository is intended for inference, demonstration, and academic reference. Training datasets are not included in the public repo.

## Quick Start

1. Clone the repository.
2. Install Git LFS and download model files.
3. Install dependencies for the demo you want to run.
4. Run the TensorFlow CLI or the PyTorch GUI.

## Prerequisites

- Python 3.7 or newer
- Git LFS installed locally
- A desktop environment for the Tkinter GUI

GPU is optional. Both demos can run on CPU.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Initialize Git LFS and pull the tracked model weights:

```bash
git lfs install
git lfs pull
```

Install only the dependencies you need:

```bash
pip install -r requirements-tf.txt
```

or

```bash
pip install -r requirements-torch.txt
```

If you want both stacks in one environment:

```bash
pip install -r requirements.txt
```

## Run the TensorFlow/Keras CLI

The TensorFlow path uses `main.py`, `bird_species_model.h5`, and `labels.json`.

Run:

```bash
python main.py --image "path\to\your\image.jpg"
```

Optional arguments:

- `--model` to point to a different `.h5` file
- `--labels` to point to a different `labels.json`
- `--image-size` to override the default input size of `224`

Example:

```bash
python main.py --image "sample.jpg"
```

Expected output format:

```text
Predicted Bird Species: American Goldfinch
```

## Run the PyTorch GUI

The PyTorch GUI uses `gui2.py` and `best_model.pth`.

Run:

```bash
python gui2.py
```

Then:

- Click `Select Bird Image`
- Choose a `.jpg`, `.jpeg`, or `.png` image
- Adjust the confidence threshold slider if needed

The GUI model is a lightweight 20-class demo and may return `Bird not found` when the confidence score is below the selected threshold.

## Project Overview

This project focuses on multiclass bird species recognition using deep learning and transfer learning techniques. The TensorFlow/Keras pipeline is based on VGG16, while the PyTorch GUI demo uses a DeiT model for a smaller curated class set.

The associated research paper was published in the proceedings of the 2024 IEEE 2nd International Conference on Innovations in High Speed Communication and Signal Processing (IHCSP).

Publication: [Multiclass Bird Species Classification Using VGG16 and TensorFlow: A Deep Learning Approach](https://doi.org/10.1109/IHCSP63227.2024.10959970)

## Repository Contents

| File | Purpose |
| --- | --- |
| `main.py` | TensorFlow/Keras command-line inference script |
| `bird_species_model.h5` | Trained TensorFlow/Keras model weights tracked with Git LFS |
| `labels.json` | Mapping from class indices to bird species names |
| `gui2.py` | PyTorch + Tkinter GUI for the 20-class demo |
| `best_model.pth` | Trained PyTorch DeiT weights tracked with Git LFS |
| `torch_infer.py` | Standalone PyTorch inference script |
| `requirements.txt` | Combined dependency list |
| `requirements-tf.txt` | TensorFlow/Keras dependency list |
| `requirements-torch.txt` | PyTorch dependency list |

## Troubleshooting

- If model loading fails, make sure Git LFS is installed and that you ran `git lfs pull`.
- If the GUI does not open, confirm that Tkinter is available in your Python installation and that you are running in a desktop session.
- If prediction fails on an image, verify that the file path is correct and the image is a supported format.

## Citation

If you use this repository in academic work, please cite:

```text
@INPROCEEDINGS{10959970,
  author={Ukey, Ekta and Shigaonkar, Sumedh and Shaikh, Sharif and Shaikh, Haji},
  booktitle={2024 IEEE 2nd International Conference on Innovations in High Speed Communication and Signal Processing (IHCSP)},
  title={Multiclass Bird Species Classification Using VGG16 and TensorFlow: A Deep Learning Approach},
  year={2024},
  pages={1-6},
  keywords={Deep learning;Training;Technological innovation;Accuracy;Transfer learning;Signal processing;Birds;Data augmentation;Data models;Tuning;Transfer Learning;VGG16;Bird Species Classification;Deep Learning;Data Augmentation},
  doi={10.1109/IHCSP63227.2024.10959970}
}
```
