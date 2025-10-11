#!/usr/bin/env python3




import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
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

    global _model, _transform, _class_names, _device

    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = make_model()


        checkpoint = torch.load(ckpt_path, map_location=_device)


        if "model" in checkpoint:
            _model.load_state_dict(checkpoint["model"])
        else:
            _model.load_state_dict(checkpoint)

        _model.to(_device)
        _model.eval()

        _transform = get_transform()
        _class_names = get_class_names()


def preprocess_input(input_img):



    if isinstance(input_img, np.ndarray):

        if len(input_img.shape) == 3:

            if input_img.shape[2] == 3:

                img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
            else:
                image = Image.fromarray(input_img)
        else:

            image = Image.fromarray(input_img, mode="L").convert("RGB")
    else:

        image = input_img.convert("RGB")

    return image


def infer(input_img, ckpt_path: str, topk: int = 5) -> List[Tuple[str, float]]:


    load_model_if_needed(ckpt_path)


    image = preprocess_input(input_img)


    input_tensor = _transform(image).unsqueeze(0).to(_device)


    with torch.no_grad():
        outputs = _model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get top-k predictions
        topk = min(topk, len(_class_names))
        top_probs, top_indices = torch.topk(probabilities, topk)


        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]


        results = [
            (_class_names[idx], float(prob))
            for idx, prob in zip(top_indices, top_probs)
        ]

    return results


def infer_single(input_img, ckpt_path: str) -> Tuple[str, float]:



    results = infer(input_img, ckpt_path, topk=1)
    return results[0] if results else ("Unknown", 0.0)
