from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spine_quality_resnet18_v13_multitask_balanced_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# МОДЕЛЬ V13
# ============================================================

class V13Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.backbone.fc = nn.Identity()

        # Финальное качество
        self.final_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 2)
        )

        # Позиционирование
        self.position_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Ось
        self.axis_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Артефакты
        self.artifacts_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        features = self.backbone(x)

        final_logits = self.final_head(features)

        position_logits = self.position_head(features)

        axis_logits = self.axis_head(features)

        artifacts_logits = self.artifacts_head(features)

        return (
            final_logits,
            position_logits,
            axis_logits,
            artifacts_logits
        )


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена:\n{MODEL_PATH}"
        )

    model = V13Model().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# НОРМАЛИЗАЦИЯ
# ============================================================

def percentile_normalize(image):

    image = image.astype(np.float32)

    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 <= p1:
        return np.zeros_like(image)

    image = np.clip(
        image,
        p1,
        p99
    )

    image = (
        image - p1
    ) / (
        p99 - p1
    )

    return image


# ============================================================
# PREPROCESSING DICOM
# ============================================================

def preprocess_dicom(dicom_path):

    dicom_path = Path(dicom_path)

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"DICOM не найден:\n{dicom_path}"
        )

    dataset = pydicom.dcmread(
        dicom_path
    )

    image = dataset.pixel_array

    image = percentile_normalize(
        image
    )

    image = (
        image * 255
    ).astype(np.uint8)

    image = Image.fromarray(
        image
    )

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Lambda(
            lambda x:
            x.repeat(3, 1, 1)
            if x.shape[0] == 1
            else x
        ),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor, dataset


# ============================================================
# PREDICTION
# ============================================================

def predict(dicom_path):

    model = load_model()

    image_tensor, dataset = (
        preprocess_dicom(
            dicom_path
        )
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        (
            final_logits,
            position_logits,
            axis_logits,
            artifacts_logits
        ) = model(
            image_tensor
        )

        final_probs = torch.softmax(
            final_logits,
            dim=1
        )[0]

        position_probs = torch.softmax(
            position_logits,
            dim=1
        )[0]

        axis_probs = torch.softmax(
            axis_logits,
            dim=1
        )[0]

        artifacts_probs = torch.softmax(
            artifacts_logits,
            dim=1
        )[0]

    probability = (
        final_probs[1].item()
    )

    prediction = (
        "Патология"
        if probability >= 0.50
        else "Норма"
    )

    return {
        "prediction": prediction,

        "probability": probability,

        "position_probability":
            position_probs[1].item(),

        "axis_probability":
            axis_probs[1].item(),

        "artifacts_probability":
            artifacts_probs[1].item(),

        "rows":
            int(dataset.Rows),

        "columns":
            int(dataset.Columns),

        "modality":
            getattr(
                dataset,
                "Modality",
                "UNKNOWN"
            )
    }


# ============================================================
# ЗАПУСК ИЗ КОМАНДНОЙ СТРОКИ
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "\nИспользование:\n"
            "python predict_v13.py путь_к_DICOM\n"
        )

        raise SystemExit(1)

    dicom_path = sys.argv[1]

    result = predict(
        dicom_path
    )

    print()
    print("=" * 60)
    print("DENSITOMETRY AI — V13")
    print("=" * 60)

    print(
        f"Результат: "
        f"{result['prediction']}"
    )

    print(
        f"Вероятность патологии: "
        f"{result['probability']:.3f}"
    )

    print()

    print(
        f"Позиционирование: "
        f"{result['position_probability']:.3f}"
    )

    print(
        f"Отклонение оси: "
        f"{result['axis_probability']:.3f}"
    )

    print(
        f"Артефакты: "
        f"{result['artifacts_probability']:.3f}"
    )

    print()

    print(
        f"Размер изображения: "
        f"{result['rows']} × "
        f"{result['columns']}"
    )

    print(
        f"Modality: "
        f"{result['modality']}"
    )

    print("=" * 60)