from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from PIL import Image

import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet18_Weights

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "spine_quality_v9_exact_predictions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spine_quality_resnet18_v9_best.pth"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "visualizations"
    / "v9_gradcam_test.png"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# V9 MODEL
# ============================================================

class V9Model(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        self.backbone = models.resnet18(
            weights=weights
        )

        features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        for parameter in self.backbone.parameters():

            parameter.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Linear(
                features,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        features = self.backbone(
            x
        )

        return self.classifier(
            features
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = V9Model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# DICOM
# ============================================================

def load_dicom(path):

    ds = pydicom.dcmread(
        str(path),
        force=True
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Percentile normalization
    # --------------------------------------------------------

    p1 = np.percentile(
        image,
        1
    )

    p99 = np.percentile(
        image,
        99
    )

    if p99 > p1:

        image = (
            image - p1
        ) / (
            p99 - p1
        )

    else:

        image = np.zeros_like(
            image
        )

    image = np.clip(
        image,
        0,
        1
    )

    image_uint8 = (
        image * 255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        image_uint8,
        mode="L"
    )


# ============================================================
# RESIZE
# ============================================================

def resize_with_padding(
    image,
    size=224
):

    width, height = image.size

    scale = min(
        size / width,
        size / height
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    image = image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "L",
        (
            size,
            size
        ),
        0
    )

    left = (
        size - new_width
    ) // 2

    top = (
        size - new_height
    ) // 2

    canvas.paste(
        image,
        (
            left,
            top
        )
    )

    return canvas


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image):

    image = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    array = np.asarray(
        image
    ).astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).unsqueeze(
        0
    )

    tensor = tensor.repeat(
        3,
        1,
        1
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    tensor = (
        tensor - mean
    ) / std

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        DEVICE
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer
        self.activations = None

        self.forward_hook = (
            self.target_layer.register_forward_hook(
                self.save_activation
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

        # V9 имеет frozen backbone.
        # Поэтому явно сохраняем градиенты активаций.
        if output.requires_grad:
            output.retain_grad()

    def generate(
        self,
        image,
        target_class
    ):

        self.model.zero_grad()

        # Очень важно для Grad-CAM:
        # backbone V9 frozen, поэтому включаем
        # отслеживание градиента от входного изображения.
        image = image.clone().detach()
        image.requires_grad_(True)

        output = self.model(
            image
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM: активации target layer не получены."
            )

        if self.activations.grad is None:

            raise RuntimeError(
                "Grad-CAM: градиент активаций отсутствует. "
                "Проверь target_layer и requires_grad."
            )

        activations = (
            self.activations[0]
        )

        gradients = (
            self.activations.grad[0]
        )

        # Global Average Pooling градиентов
        weights = gradients.mean(
            dim=(1, 2)
        )

        # Взвешенная сумма feature maps
        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i in range(
            activations.shape[0]
        ):

            cam += (
                weights[i]
                * activations[i]
            )

        # Оставляем только положительное влияние
        cam = torch.relu(
            cam
        )

        # Нормализация 0...1
        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        return (
            cam,
            output.detach()
        )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    cam
):

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    ) / 255.0

    cam_image = Image.fromarray(
        np.uint8(
            cam * 255
        )
    )

    cam_image = cam_image.resize(
        image.size,
        Image.Resampling.BILINEAR
    )

    cam_array = np.asarray(
        cam_image
    ).astype(
        np.float32
    ) / 255.0

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )

    ax.imshow(
        image_array,
        cmap="gray"
    )

    ax.imshow(
        cam_array,
        cmap="jet",
        alpha=0.45
    )

    ax.axis(
        "off"
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("V9 GRAD-CAM")
    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        "TEST images:",
        len(df)
    )

    # --------------------------------------------------------
    # Add error column
    # --------------------------------------------------------

    df["error"] = (
        df["spine_quality"]
        != df["prediction"]
    )

    # Errors first
    df = df.sort_values(
        by="error",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # Last convolutional layer
    target_layer = (
        model.backbone.layer4[-1].conv2
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(14, 22)
    )

    axes = axes.flatten()

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        dicom_path = (
            PROJECT_ROOT
            / row["dicom_path"]
        )

        image = load_dicom(
            dicom_path
        )

        image = resize_with_padding(
            image,
            IMAGE_SIZE
        )

        tensor = preprocess(
            image
        )

        true_label = int(
            row["spine_quality"]
        )

        pred_label = int(
            row["prediction"]
        )

        probability = float(
            row["probability_class1"]
        )

        # Grad-CAM for the class predicted by model
        target_class = pred_label

        cam, logits = gradcam.generate(
            tensor,
            target_class
        )

        ax = axes[i]

        image_array = np.asarray(
            image
        ).astype(
            np.float32
        ) / 255.0

        cam_image = Image.fromarray(
            np.uint8(
                cam * 255
            )
        )

        cam_image = cam_image.resize(
            image.size,
            Image.Resampling.BILINEAR
        )

        cam_array = np.asarray(
            cam_image
        ).astype(
            np.float32
        ) / 255.0

        ax.imshow(
            image_array,
            cmap="gray"
        )

        ax.imshow(
            cam_array,
            cmap="jet",
            alpha=0.45
        )

        if true_label == pred_label:

            status = "CORRECT"

        else:

            status = "ERROR"

        ax.set_title(
            f"{i + 1}. {status}\n"
            f"TRUE={true_label}  "
            f"PRED={pred_label}  "
            f"P1={probability:.3f}",
            fontsize=9
        )

        ax.axis(
            "off"
        )

    plt.suptitle(
        "V9 — Grad-CAM на TEST",
        fontsize=16
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Готово:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()