#!/usr/bin/env python3


import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
from typing import List, Tuple


def make_model(num_classes=20, dropout_p=0.3):

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_feat, num_classes),
    )
    return model


def get_transform():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform


def get_class_names() -> List[str]:

    class_names = [
        "Class_0",
        "Class_1",
        "Class_2",
        "Class_3",
        "Class_4",
        "Class_5",
        "Class_6",
        "Class_7",
        "Class_8",
        "Class_9",
        "Class_10",
        "Class_11",
        "Class_12",
        "Class_13",
        "Class_14",
        "Class_15",
        "Class_16",
        "Class_17",
        "Class_18",
        "Class_19",
    ]
    return class_names


_model = None
_transform = None
_class_names = None
_device = None


def load_model_if_needed(ckpt_path: str):
    """Load the model from checkpoint if not already loaded."""
    global _model, _transform, _class_names, _device

    if _model is None:
        try:
            print(f"Loading model from: {ckpt_path}")

            # Check if file exists
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_path}")

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {_device}")

            _transform = get_transform()
            _class_names = get_class_names()
            print(f"Transform initialized: {_transform is not None}")
            print(f"Class names loaded: {len(_class_names)} classes")

            _model = make_model()
            print(f"Model created: {_model is not None}")

            checkpoint = torch.load(ckpt_path, map_location=_device)
            print(
                f"Checkpoint loaded, keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Direct state dict'}"
            )

            if "model" in checkpoint:
                _model.load_state_dict(checkpoint["model"])
            elif "state_dict" in checkpoint:
                _model.load_state_dict(checkpoint["state_dict"])
            else:
                _model.load_state_dict(checkpoint)

            _model.to(_device)
            _model.eval()

            print("Model loaded and set to eval mode successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()
            raise


def preprocess_input(input_img):

    if isinstance(input_img, np.ndarray):

        if len(input_img.shape) == 3:
            if input_img.shape[2] == 3:

                img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
            elif input_img.shape[2] == 1:

                image = Image.fromarray(input_img.squeeze(), mode="L").convert("RGB")
            else:

                image = Image.fromarray(input_img)
        elif len(input_img.shape) == 2:

            image = Image.fromarray(input_img, mode="L").convert("RGB")
        else:
            raise ValueError(f"Unsupported image shape: {input_img.shape}")
    else:

        image = input_img.convert("RGB")

    return image


def infer(input_img, ckpt_path: str, topk: int = 5) -> List[Tuple[str, float]]:

    try:

        load_model_if_needed(ckpt_path)


        if _model is None:
            raise RuntimeError("Model is None after loading")
        if _transform is None:
            raise RuntimeError("Transform is None after loading")
        if _class_names is None:
            raise RuntimeError("Class names are None after loading")
        if _device is None:
            raise RuntimeError("Device is None after loading")


        image = preprocess_input(input_img)

        input_tensor = _transform(image).unsqueeze(0).to(_device)

        with torch.no_grad():
            outputs = _model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get top-k predictions
            topk = min(topk, len(_class_names))
            top_probs, top_indices = torch.topk(probabilities, topk)

            # Convert to CPU and numpy
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]

            # Format results
            results = [
                (_class_names[idx], float(prob))
                for idx, prob in zip(top_indices, top_probs)
            ]

        return results

    except Exception as e:
        print(f"Error in inference: {e}")
        print(f"Input image type: {type(input_img)}")
        if isinstance(input_img, np.ndarray):
            print(f"Input image shape: {input_img.shape}")
            print(f"Input image dtype: {input_img.dtype}")
        print(f"Model: {_model is not None}")
        print(f"Transform: {_transform is not None}")
        print(f"Class names: {_class_names is not None}")
        print(f"Device: {_device}")
        import traceback

        traceback.print_exc()
        return []


def infer_single(input_img, ckpt_path: str) -> Tuple[str, float]:

    results = infer(input_img, ckpt_path, topk=1)
    return results[0] if results else ("Unknown", 0.0)
