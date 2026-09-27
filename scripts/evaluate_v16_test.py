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

ROOT = Path(
    r"C:\hakaton\densitometry_ai"
)

MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

TEST_SPLIT = (
    ROOT
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "spine_quality_resnet18_v16_ablation_best.pth"
)

RESULT_PATH = (
    ROOT
    / "results"
    / "v16_test_exact_predictions.csv"
)

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 8


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
# IMAGE
# ============================================================

def normalize_percentile(arr):

    arr = arr.astype(
        np.float32
    )

    p1 = np.percentile(
        arr,
        1
    )

    p99 = np.percentile(
        arr,
        99
    )

    if p99 <= p1:

        minimum = arr.min()
        maximum = arr.max()

        if maximum > minimum:

            arr = (
                arr - minimum
            ) / (
                maximum - minimum
            )

        else:

            arr = np.zeros_like(
                arr
            )

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


def load_image(path):

    path = resolve_path(path)

    ds = pydicom.dcmread(path)

    arr = ds.pixel_array.astype(
        np.float32
    )

    arr = normalize_percentile(
        arr
    )

    arr = (
        arr * 255
    ).astype(np.uint8)

    return Image.fromarray(
        arr
    )


# ============================================================
# EXACT DEDUP
# ============================================================

def exact_dedup(df):

    df = df.copy()

    hashes = []

    for path in df["dicom_path"]:

        path = resolve_path(path)

        ds = pydicom.dcmread(
            path
        )

        digest = hashlib.md5(
            ds.pixel_array.tobytes()
        ).hexdigest()

        hashes.append(
            digest
        )

    df["pixel_hash"] = hashes

    df = (
        df
        .drop_duplicates(
            subset=["pixel_hash"],
            keep="first"
        )
        .drop(
            columns=["pixel_hash"]
        )
        .reset_index(drop=True)
    )

    return df


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

        return len(
            self.df
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = load_image(
            row["dicom_path"]
        )

        image = self.transform(
            image
        )

        return image


# ============================================================
# V16 MODEL
# ============================================================

class V16Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

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

        features = self.backbone(x)

        return self.classifier(
            features
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("V16 — EXACT TEST EVALUATION")
print("=" * 80)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# LOAD MANIFEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

manifest["study_id"] = (
    manifest["study_id"]
    .astype(str)
)

test_split = pd.read_csv(
    TEST_SPLIT
)

test_ids = set(
    test_split["study_id"]
    .astype(str)
)


# ============================================================
# FILTER
# ============================================================

test_df = manifest[
    (manifest["anatomy"] == "spine")
    &
    (manifest["spine_quality"].notna())
    &
    (manifest["study_id"].isin(test_ids))
].copy()


print()
print(
    f"TEST before dedup: "
    f"{len(test_df)}"
)


test_df = exact_dedup(
    test_df
)


print(
    f"TEST after dedup: "
    f"{len(test_df)}"
)

print(
    f"Unique studies: "
    f"{test_df['study_id'].nunique()}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()

print(
    "TEST classes:"
)

print(
    test_df[
        "spine_quality"
    ]
    .astype(int)
    .value_counts()
    .sort_index()
)


# ============================================================
# DATASET
# ============================================================

dataset = SpineDataset(
    test_df
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = V16Model().to(
    DEVICE
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print()
print(
    "Checkpoint:"
)

print(
    "Epoch:",
    checkpoint.get("epoch")
)

print(
    "Validation accuracy:",
    checkpoint.get(
        "val_accuracy"
    )
)

print(
    "Validation F1:",
    checkpoint.get(
        "val_f1"
    )
)


# ============================================================
# PREDICT
# ============================================================

probabilities = []

with torch.no_grad():

    for images in loader:

        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        probabilities.extend(
            probs.cpu().numpy()
        )


probabilities = np.asarray(
    probabilities
)


# ============================================================
# THRESHOLD 0.50
# ============================================================

threshold = 0.50

y_true = (
    test_df[
        "spine_quality"
    ]
    .astype(int)
    .to_numpy()
)

y_pred = (
    probabilities
    >= threshold
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

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("V16 TEST RESULTS")
print("=" * 80)

print(
    f"Threshold : {threshold:.2f}"
)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1        : {f1 * 100:.2f}%"
)

print()
print(
    "Confusion matrix:"
)

print(
    "              Pred 0   Pred 1"
)

print(
    f"True 0       {cm[0, 0]:7d}  "
    f"{cm[0, 1]:7d}"
)

print(
    f"True 1       {cm[1, 0]:7d}  "
    f"{cm[1, 1]:7d}"
)


# ============================================================
# DETAILED PREDICTIONS
# ============================================================

result = test_df[
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


result[
    "probability_class1"
] = probabilities


result[
    "prediction"
] = y_pred


result[
    "correct"
] = (
    result["prediction"]
    ==
    result["spine_quality"]
)


# ============================================================
# PRINT PREDICTIONS
# ============================================================

print()
print("=" * 80)
print("DETAILED PREDICTIONS")
print("=" * 80)

for _, row in result.iterrows():

    print(
        f"{row['study_id']} | "
        f"true={int(row['spine_quality'])} | "
        f"pred={int(row['prediction'])} | "
        f"p1={row['probability_class1']:.4f} | "
        f"{'OK' if row['correct'] else 'ERROR'}"
    )


# ============================================================
# SAVE
# ============================================================

RESULT_PATH.parent.mkdir(
    exist_ok=True
)

result.to_csv(
    RESULT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    RESULT_PATH
)

print()
print("=" * 80)
print("ГОТОВО")
print("=" * 80)