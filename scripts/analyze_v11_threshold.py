# scripts/analyze_v11_threshold.py

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_manifest.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "validation.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "spine_quality_resnet18_v11_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8


# ============================================================
# PATH
# ============================================================

def resolve_path(path_value):
    path = Path(str(path_value))

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


# ============================================================
# DICOM HASH
# ============================================================

def get_pixel_hash(image_path):
    image_path = resolve_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Не найден DICOM: {image_path}"
        )

    ds = pydicom.dcmread(
        image_path,
        stop_before_pixels=False
    )

    return hashlib.sha256(
        ds.PixelData
    ).hexdigest()


# ============================================================
# READ DICOM
# ============================================================

def read_dicom_image(image_path):
    image_path = resolve_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Не найден DICOM: {image_path}"
        )

    ds = pydicom.dcmread(
        image_path,
        stop_before_pixels=False
    )

    image = ds.pixel_array.astype(np.float32)
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Ожидалось 2D изображение, "
            f"получено {image.shape}: {image_path}"
        )

    return image


# ============================================================
# V11 PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.astype(np.float32)

    # Percentile 1-99
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 > p1:
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1)
    else:
        image = np.zeros_like(
            image,
            dtype=np.float32
        )

    # Центральный ROI: 70% ширины
    height, width = image.shape

    left = int(width * 0.15)
    right = int(width * 0.85)

    image = image[:, left:right]

    # 0-255
    image = (
        image * 255.0
    ).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(
        image,
        mode="L"
    )

    # Resize с сохранением пропорций
    width, height = image.size

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height
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
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    # Padding
    canvas = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        0
    )

    left_pad = (
        IMAGE_SIZE - new_width
    ) // 2

    top_pad = (
        IMAGE_SIZE - new_height
    ) // 2

    canvas.paste(
        image,
        (left_pad, top_pad)
    )

    # Tensor
    image = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # 1 канал -> 3 канала
    image = np.stack(
        [image, image, image],
        axis=0
    )

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor


# ============================================================
# DATASET
# ============================================================

class SpineDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = resolve_path(
            row["dicom_path"]
        )

        image = read_dicom_image(
            image_path
        )

        image = preprocess_image(
            image
        )

        label = int(
            float(row["spine_quality"])
        )

        study_id = str(
            row["study_id"]
        )

        return image, label, study_id


# ============================================================
# V11 MODEL
# ============================================================

class V11Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        features = self.backbone(x)

        logits = self.classifier(
            features
        )

        return logits


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Загрузка V11 model...")

    model = V11Model()

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

    # Убираем module. если checkpoint был сохранён DataParallel
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    state_dict = cleaned_state_dict

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing or unexpected:

        print()
        print("Missing keys:")

        for key in missing:
            print(" ", key)

        print()
        print("Unexpected keys:")

        for key in unexpected:
            print(" ", key)

        raise RuntimeError(
            "Checkpoint V11 не соответствует архитектуре модели."
        )

    model.to(DEVICE)
    model.eval()

    print("Модель загружена.")

    return model


# ============================================================
# EXACT VALIDATION
# ============================================================

def make_exact_validation(
    manifest,
    validation_ids
):

    df = manifest.copy()

    df["study_id"] = (
        df["study_id"].astype(str)
    )

    validation_ids = set(
        str(x)
        for x in validation_ids
    )

    # Validation studies
    df = df[
        df["study_id"].isin(
            validation_ids
        )
    ].copy()

    # Только spine
    df = df[
        df["anatomy"]
        .astype(str)
        .str.lower()
        == "spine"
    ].copy()

    # Только с label
    df = df[
        df["spine_quality"].notna()
    ].copy()

    print(
        "Удаление точных дубликатов..."
    )

    hashes = []

    for _, row in df.iterrows():

        pixel_hash = get_pixel_hash(
            row["dicom_path"]
        )

        hashes.append(
            pixel_hash
        )

    df["pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    ).copy()

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(
    model,
    dataset
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    probabilities = []
    labels = []
    studies = []

    print(
        "Получение predictions..."
    )

    with torch.no_grad():

        for (
            images,
            batch_labels,
            batch_studies
        ) in loader:

            images = images.to(
                DEVICE
            )

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1
            )

            p1 = probs[:, 1]

            probabilities.extend(
                p1.cpu()
                .numpy()
                .tolist()
            )

            labels.extend(
                batch_labels.numpy()
                .tolist()
            )

            studies.extend(
                list(batch_studies)
            )

    return (
        np.array(
            probabilities,
            dtype=np.float32
        ),
        np.array(
            labels,
            dtype=np.int64
        ),
        studies
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        y_pred
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    y_true,
    probabilities
):

    print()
    print("=" * 70)
    print(
        "V11 VALIDATION THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print()

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90
    ]

    results = []

    print(
        f"{'Threshold':>10} "
        f"{'Accuracy':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )

    print("-" * 55)

    for threshold in thresholds:

        (
            accuracy,
            precision,
            recall,
            f1,
            _
        ) = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"{threshold:10.2f} "
            f"{accuracy * 100:9.2f}% "
            f"{precision * 100:9.2f}% "
            f"{recall * 100:9.2f}% "
            f"{f1 * 100:9.2f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    # Лучший F1
    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print()
    print(
        "Лучший threshold по F1:"
    )

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{best_f1['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{best_f1['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{best_f1['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{best_f1['f1'] * 100:.2f}%"
    )

    # Лучшая Accuracy
    best_accuracy = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    print()
    print(
        "Лучший threshold по Accuracy:"
    )

    print(
        f"Threshold: "
        f"{best_accuracy['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{best_accuracy['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{best_accuracy['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{best_accuracy['f1'] * 100:.2f}%"
    )

    return results_df


# ============================================================
# PRINT PREDICTIONS
# ============================================================

def print_predictions(
    y_true,
    probabilities,
    studies
):

    print()
    print("=" * 70)
    print(
        "VALIDATION PREDICTIONS"
    )
    print("=" * 70)

    order = np.argsort(
        probabilities
    )

    for idx in order:

        print(
            f"{idx + 1:02d}. "
            f"study={studies[idx]} | "
            f"true={y_true[idx]} | "
            f"P(class1)="
            f"{probabilities[idx]:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "V11 THRESHOLD ANALYSIS"
    )

    print(
        f"Device: {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Исходных строк manifest: "
        f"{len(manifest)}"
    )

    spine = manifest[
        manifest["anatomy"]
        .astype(str)
        .str.lower()
        == "spine"
    ].copy()

    print(
        f"Spine строк: "
        f"{len(spine)}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    if "study_id" in validation.columns:

        validation_ids = (
            validation["study_id"]
            .astype(str)
            .tolist()
        )

    else:

        validation_ids = (
            validation.iloc[:, 0]
            .astype(str)
            .tolist()
        )

    validation_spine = spine[
        spine["study_id"]
        .astype(str)
        .isin(
            set(validation_ids)
        )
    ]

    print(
        f"Validation spine: "
        f"{len(validation_spine)}"
    )

    # --------------------------------------------------------
    # Exact validation
    # --------------------------------------------------------

    validation_df = make_exact_validation(
        manifest,
        validation_ids
    )

    print()
    print(
        "VALIDATION exact:"
    )

    print(
        f"Images: "
        f"{len(validation_df)}"
    )

    print(
        f"Studies: "
        f"{validation_df['study_id'].nunique()}"
    )

    print()

    print(
        validation_df[
            "spine_quality"
        ]
        .value_counts()
        .sort_index()
    )

    if len(validation_df) == 0:

        raise RuntimeError(
            "VALIDATION после фильтрации пуст."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SpineDataset(
        validation_df
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    (
        probabilities,
        y_true,
        studies
    ) = get_predictions(
        model,
        dataset
    )

    if len(y_true) == 0:

        raise RuntimeError(
            "Не удалось получить predictions."
        )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    results_df = analyze_thresholds(
        y_true,
        probabilities
    )

    # --------------------------------------------------------
    # Detailed predictions
    # --------------------------------------------------------

    print_predictions(
        y_true,
        probabilities,
        studies
    )

    # --------------------------------------------------------
    # Threshold 0.50
    # --------------------------------------------------------

    threshold = 0.50

    (
        accuracy,
        precision,
        recall,
        f1,
        y_pred
    ) = calculate_metrics(
        y_true,
        probabilities,
        threshold
    )

    print()
    print("=" * 70)
    print(
        "V11 VALIDATION @ THRESHOLD 0.50"
    )
    print("=" * 70)

    print(
        f"Accuracy:  "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1:        "
        f"{f1 * 100:.2f}%"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    thresholds_path = (
        RESULTS_DIR
        / "v11_validation_thresholds.csv"
    )

    results_df.to_csv(
        thresholds_path,
        index=False
    )

    predictions_path = (
        RESULTS_DIR
        / "v11_validation_predictions.csv"
    )

    predictions_df = pd.DataFrame({
        "study_id": studies,
        "true_label": y_true,
        "probability_class1": probabilities,
        "prediction_threshold_050": y_pred
    })

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print()
    print(
        f"Threshold results saved: "
        f"{thresholds_path}"
    )

    print(
        f"Predictions saved: "
        f"{predictions_path}"
    )

    print()
    print(
        "ГОТОВО."
    )


if __name__ == "__main__":
    main()