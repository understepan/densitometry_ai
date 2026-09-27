from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

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
TEST_SPLIT = ROOT / "data" / "splits" / "test.csv"

V13_MODEL = (
    ROOT
    / "models"
    / "spine_quality_resnet18_v13_multitask_balanced_best.pth"
)

V15_MODEL = (
    ROOT
    / "models"
    / "spine_quality_resnet18_v15_best.pth"
)

RESULTS_DIR = ROOT / "results"

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
            f"DICOM не найден:\n{path}"
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
# LOAD DICOM
# ============================================================

def load_dicom_image(path):

    import pydicom

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

    import pydicom

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

    result = result.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    )

    return result.drop(
        columns=["pixel_hash"]
    ).reset_index(drop=True)


# ============================================================
# DATASET
# ============================================================

class SpineTestDataset(Dataset):

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
# V13 MODEL
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

        final_logits = self.final_head(
            features
        )

        position_logits = self.position_head(
            features
        )

        axis_logits = self.axis_head(
            features
        )

        artifacts_logits = self.artifacts_head(
            features
        )

        return (
            final_logits,
            position_logits,
            axis_logits,
            artifacts_logits
        )


# ============================================================
# V15 MODEL
# ============================================================

class V15Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        num_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(

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

    def forward(self, x):

        features = self.backbone(x)

        return self.classifier(
            features
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_v13():

    model = V13Model().to(
        DEVICE
    )

    checkpoint = torch.load(
        V13_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_v15():

    model = V15Model().to(
        DEVICE
    )

    checkpoint = torch.load(
        V15_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, loader, model_name):

    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            if model_name == "V13":

                final_logits, _, _, _ = (
                    model(images)
                )

                logits = final_logits

            else:

                logits = model(
                    images
                )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_true.extend(
                labels.numpy()
            )

            all_pred.extend(
                predictions.cpu().numpy()
            )

            all_prob.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob)
    )


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    name,
    y_true,
    y_pred
):

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

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1       : {f1 * 100:.2f}%"
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

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("V13 + V15 — EXACT TEST EVALUATION")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)


# ------------------------------------------------------------
# MANIFEST
# ------------------------------------------------------------

manifest = pd.read_csv(
    MANIFEST
)

test_split = pd.read_csv(
    TEST_SPLIT
)

manifest["study_id"] = (
    manifest["study_id"]
    .astype(str)
)

test_ids = (
    test_split["study_id"]
    .astype(str)
)


# ------------------------------------------------------------
# TEST SPINE
# ------------------------------------------------------------

test_df = manifest[
    (manifest["anatomy"] == "spine")
    &
    (manifest["spine_quality"].notna())
    &
    (manifest["study_id"].isin(test_ids))
].copy()


print()
print(
    f"TEST spine before dedup: "
    f"{len(test_df)}"
)


# ------------------------------------------------------------
# DEDUP
# ------------------------------------------------------------

test_df = exact_dedup(
    test_df
)


print(
    f"TEST spine after exact dedup: "
    f"{len(test_df)}"
)


print(
    f"Unique studies: "
    f"{test_df['study_id'].nunique()}"
)


print()
print("TEST classes:")

print(
    test_df[
        "spine_quality"
    ].value_counts().sort_index()
)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

test_dataset = SpineTestDataset(
    test_df
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

print()
print("Loading V13...")

v13 = load_v13()

print("V13 loaded.")

print()
print("Loading V15...")

v15 = load_v15()

print("V15 loaded.")


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print()
print("Running V13...")

v13_true, v13_pred, v13_prob = predict(
    v13,
    test_loader,
    "V13"
)


print("Running V15...")

v15_true, v15_pred, v15_prob = predict(
    v15,
    test_loader,
    "V15"
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

v13_metrics = print_metrics(
    "V13 TEST",
    v13_true,
    v13_pred
)

v15_metrics = print_metrics(
    "V15 TEST",
    v15_true,
    v15_pred
)


# ============================================================
# DETAILED PREDICTIONS
# ============================================================

result_df = test_df[
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


result_df[
    "V13_prediction"
] = v13_pred


result_df[
    "V13_probability_class1"
] = v13_prob


result_df[
    "V15_prediction"
] = v15_pred


result_df[
    "V15_probability_class1"
] = v15_prob


result_df[
    "V13_correct"
] = (
    v13_true == v13_pred
)


result_df[
    "V15_correct"
] = (
    v15_true == v15_pred
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

RESULTS_DIR.mkdir(
    exist_ok=True
)


output_path = (
    RESULTS_DIR
    / "v13_v15_test_comparison.csv"
)


result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


summary_df = pd.DataFrame(
    [
        v13_metrics,
        v15_metrics
    ]
)


summary_path = (
    RESULTS_DIR
    / "v13_v15_test_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print()
print("Detailed predictions saved:")
print(output_path)

print()
print("Summary saved:")
print(summary_path)

print()
print("=" * 70)
print("ГОТОВО")
print("=" * 70)