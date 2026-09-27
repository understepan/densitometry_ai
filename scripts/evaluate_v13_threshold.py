from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\hakaton\densitometry_ai")

MANIFEST = ROOT / "data" / "processed" / "labeled_manifest.csv"

TRAIN_SPLIT = ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT = ROOT / "data" / "splits" / "validation.csv"
TEST_SPLIT = ROOT / "data" / "splits" / "test.csv"

MODEL_PATH = (
    ROOT
    / "models"
    / "spine_quality_resnet18_v13_multitask_balanced_best.pth"
)

RESULTS_DIR = ROOT / "results"

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 8

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


# ============================================================
# PATH
# ============================================================

def resolve_path(path):

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден:\n{path}"
        )

    return path


# ============================================================
# NORMALIZATION
# ============================================================

def percentile_normalize(arr):

    arr = arr.astype(np.float32)

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:

        arr = arr - arr.min()

        max_value = arr.max()

        if max_value > 0:
            arr = arr / max_value

    else:

        arr = np.clip(
            arr,
            p1,
            p99
        )

        arr = (
            arr - p1
        ) / (
            p99 - p1
        )

    return np.clip(
        arr,
        0,
        1
    )


# ============================================================
# DICOM
# ============================================================

def load_dicom_image(path):

    path = resolve_path(path)

    ds = pydicom.dcmread(path)

    arr = ds.pixel_array.astype(
        np.float32
    )

    arr = percentile_normalize(arr)

    arr = (
        arr * 255
    ).astype(np.uint8)

    return Image.fromarray(arr)


# ============================================================
# EXACT PIXEL DEDUP
# ============================================================

def exact_dedup(df):

    hashes = []

    for path in df["dicom_path"]:

        path = resolve_path(path)

        ds = pydicom.dcmread(path)

        pixel_bytes = (
            ds.pixel_array.tobytes()
        )

        digest = hashlib.md5(
            pixel_bytes
        ).hexdigest()

        hashes.append(digest)

    result = df.copy()

    result["pixel_hash"] = hashes

    result = (
        result
        .drop_duplicates(
            subset=["pixel_hash"],
            keep="first"
        )
        .drop(
            columns=["pixel_hash"]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# DATASET
# ============================================================

class SpineDataset(Dataset):

    def __init__(self, df):

        self.df = (
            df.reset_index(drop=True)
        )

        self.transform = transforms.Compose([

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.ToTensor(),

            transforms.Lambda(
                lambda x:
                x.repeat(3, 1, 1)
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

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = load_dicom_image(
            row["dicom_path"]
        )

        image = self.transform(
            image
        )

        label = int(
            row["spine_quality"]
        )

        return image, label


# ============================================================
# V13
# ============================================================

class V13Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        num_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.final_head = nn.Sequential(

            nn.Linear(
                num_features,
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

        self.position_head = nn.Sequential(

            nn.Linear(
                num_features,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )

        self.axis_head = nn.Sequential(

            nn.Linear(
                num_features,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )

        self.artifacts_head = nn.Sequential(

            nn.Linear(
                num_features,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        final_logits = (
            self.final_head(features)
        )

        position_logits = (
            self.position_head(features)
        )

        axis_logits = (
            self.axis_head(features)
        )

        artifacts_logits = (
            self.artifacts_head(features)
        )

        return (
            final_logits,
            position_logits,
            axis_logits,
            artifacts_logits
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = V13Model().to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print()
    print("Checkpoint information:")

    print(
        "Epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Validation accuracy:",
        checkpoint.get("val_accuracy")
    )

    print(
        "Validation F1:",
        checkpoint.get("val_f1")
    )

    return model


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

def predict_probabilities(
    model,
    loader
):

    probabilities = []
    labels = []

    with torch.no_grad():

        for images, y in loader:

            images = images.to(
                DEVICE
            )

            final_logits, _, _, _ = (
                model(images)
            )

            probs = torch.softmax(
                final_logits,
                dim=1
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy()
            )

            labels.extend(
                y.numpy()
            )

    return (
        np.array(labels),
        np.array(probabilities)
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

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("V13 — VALIDATION THRESHOLD → TEST")
print("=" * 80)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

manifest["study_id"] = (
    manifest["study_id"].astype(str)
)


train_split = pd.read_csv(
    TRAIN_SPLIT
)

val_split = pd.read_csv(
    VAL_SPLIT
)

test_split = pd.read_csv(
    TEST_SPLIT
)


train_ids = set(
    train_split["study_id"]
    .astype(str)
)

val_ids = set(
    val_split["study_id"]
    .astype(str)
)

test_ids = set(
    test_split["study_id"]
    .astype(str)
)


# ============================================================
# BUILD SPLITS
# ============================================================

base = manifest[
    (manifest["anatomy"] == "spine")
    &
    (manifest["spine_quality"].notna())
].copy()


train_df = base[
    base["study_id"].isin(
        train_ids
    )
].copy()


val_df = base[
    base["study_id"].isin(
        val_ids
    )
].copy()


test_df = base[
    base["study_id"].isin(
        test_ids
    )
].copy()


print()
print("Before dedup:")

print(
    f"TRAIN:      {len(train_df)}"
)

print(
    f"VALIDATION: {len(val_df)}"
)

print(
    f"TEST:       {len(test_df)}"
)


# ============================================================
# EXACT DEDUP
# ============================================================

train_df = exact_dedup(
    train_df
)

val_df = exact_dedup(
    val_df
)

test_df = exact_dedup(
    test_df
)


print()
print("After exact PixelData dedup:")

print(
    f"TRAIN:      {len(train_df)}"
)

print(
    f"VALIDATION: {len(val_df)}"
)

print(
    f"TEST:       {len(test_df)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

val_dataset = SpineDataset(
    val_df
)

test_dataset = SpineDataset(
    test_df
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = load_model()


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("Получаем VALIDATION probabilities...")

val_true, val_prob = (
    predict_probabilities(
        model,
        val_loader
    )
)


print(
    "Получаем TEST probabilities..."
)

test_true, test_prob = (
    predict_probabilities(
        model,
        test_loader
    )
)


# ============================================================
# VALIDATION THRESHOLDS
# ============================================================

print()
print("=" * 80)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 80)

validation_results = []


for threshold in THRESHOLDS:

    metrics = calculate_metrics(
        val_true,
        val_prob,
        threshold
    )

    validation_results.append(
        metrics
    )

    print(
        f"threshold={threshold:.2f} | "
        f"accuracy={metrics['accuracy']*100:.2f}% | "
        f"precision={metrics['precision']*100:.2f}% | "
        f"recall={metrics['recall']*100:.2f}% | "
        f"F1={metrics['f1']*100:.2f}%"
    )


validation_results_df = pd.DataFrame(
    validation_results
)


# ============================================================
# SELECT THRESHOLD
# ============================================================

best_row = (
    validation_results_df
    .sort_values(
        [
            "f1",
            "accuracy"
        ],
        ascending=False
    )
    .iloc[0]
)


best_threshold = float(
    best_row["threshold"]
)


print()
print("=" * 80)
print("SELECTED THRESHOLD")
print("=" * 80)

print(
    f"Threshold = {best_threshold:.2f}"
)

print(
    f"Validation accuracy = "
    f"{best_row['accuracy']*100:.2f}%"
)

print(
    f"Validation F1 = "
    f"{best_row['f1']*100:.2f}%"
)


# ============================================================
# TEST WITH SELECTED THRESHOLD
# ============================================================

test_metrics = calculate_metrics(
    test_true,
    test_prob,
    best_threshold
)


test_pred = (
    test_prob >= best_threshold
).astype(int)


cm = confusion_matrix(
    test_true,
    test_pred,
    labels=[0, 1]
)


print()
print("=" * 80)
print("TEST WITH VALIDATION-SELECTED THRESHOLD")
print("=" * 80)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : "
    f"{test_metrics['accuracy']*100:.2f}%"
)

print(
    f"Precision : "
    f"{test_metrics['precision']*100:.2f}%"
)

print(
    f"Recall    : "
    f"{test_metrics['recall']*100:.2f}%"
)

print(
    f"F1        : "
    f"{test_metrics['f1']*100:.2f}%"
)

print()
print("Confusion matrix:")

print(
    "              Pred 0   Pred 1"
)

print(
    f"True 0       {cm[0,0]:7d}  "
    f"{cm[0,1]:7d}"
)

print(
    f"True 1       {cm[1,0]:7d}  "
    f"{cm[1,1]:7d}"
)


# ============================================================
# TEST PROBABILITIES
# ============================================================

test_details = test_df[
    [
        "study_id",
        "dicom_path",
        "spine_quality",
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "comment"
    ]
].copy()


test_details[
    "probability_class1"
] = test_prob


test_details[
    "prediction"
] = test_pred


test_details[
    "correct"
] = (
    test_pred
    ==
    test_true
)


# ============================================================
# SAVE
# ============================================================

RESULTS_DIR.mkdir(
    exist_ok=True
)


validation_path = (
    RESULTS_DIR /
    "v13_validation_thresholds.csv"
)


validation_results_df.to_csv(
    validation_path,
    index=False,
    encoding="utf-8-sig"
)


details_path = (
    RESULTS_DIR /
    "v13_test_validation_selected_threshold.csv"
)


test_details.to_csv(
    details_path,
    index=False,
    encoding="utf-8-sig"
)


summary_path = (
    RESULTS_DIR /
    "v13_threshold_summary.csv"
)


pd.DataFrame(
    [
        {
            "selected_threshold":
                best_threshold,

            "validation_accuracy":
                best_row["accuracy"],

            "validation_precision":
                best_row["precision"],

            "validation_recall":
                best_row["recall"],

            "validation_f1":
                best_row["f1"],

            "test_accuracy":
                test_metrics["accuracy"],

            "test_precision":
                test_metrics["precision"],

            "test_recall":
                test_metrics["recall"],

            "test_f1":
                test_metrics["f1"],
        }
    ]
).to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(validation_path)
print(details_path)
print(summary_path)

print()
print("=" * 80)
print("ГОТОВО")
print("=" * 80)