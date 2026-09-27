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
    / "spine_quality_resnet18_v13_multitask_balanced_best.pth"
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
            f"Файл не найден:\n{path}"
        )

    return path


# ============================================================
# IMAGE
# ============================================================

def percentile_normalize(arr):

    arr = arr.astype(np.float32)

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:

        arr = arr - arr.min()

        maximum = arr.max()

        if maximum > 0:
            arr = arr / maximum

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
# EXACT DEDUP
# ============================================================

def exact_dedup(df):

    hashes = []

    for path in df["dicom_path"]:

        path = resolve_path(path)

        ds = pydicom.dcmread(path)

        digest = hashlib.md5(
            ds.pixel_array.tobytes()
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
        .drop(columns=["pixel_hash"])
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

        return image


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

        # Final quality
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

        # Position
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

        # Axis
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

        # Artifacts
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
# LOAD
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
    print("Checkpoint:")
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
# PREDICTION
# ============================================================

def predict(model, loader):

    outputs = {
        "final": [],
        "position": [],
        "axis": [],
        "artifacts": []
    }

    with torch.no_grad():

        for images in loader:

            images = images.to(
                DEVICE
            )

            (
                final_logits,
                position_logits,
                axis_logits,
                artifacts_logits
            ) = model(images)

            outputs["final"].extend(
                torch.softmax(
                    final_logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

            outputs["position"].extend(
                torch.softmax(
                    position_logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

            outputs["axis"].extend(
                torch.softmax(
                    axis_logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

            outputs["artifacts"].extend(
                torch.softmax(
                    artifacts_logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

    return outputs


# ============================================================
# METRICS
# ============================================================

def evaluate_head(
    name,
    true_values,
    probabilities
):

    valid = (
        pd.notna(true_values)
    )

    true_values = (
        np.asarray(true_values)[valid]
        .astype(int)
    )

    probabilities = (
        np.asarray(probabilities)[valid]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        true_values,
        predictions
    )

    precision = precision_score(
        true_values,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_values,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_values,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        true_values,
        predictions,
        labels=[0, 1]
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Samples  : {len(true_values)}"
    )

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
    print("Confusion:")
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

    return {
        "head": name,
        "samples": len(true_values),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("V13 MULTITASK HEADS — EXACT TEST")
print("=" * 80)

print(
    f"Device: {DEVICE}"
)


# ------------------------------------------------------------
# MANIFEST
# ------------------------------------------------------------

manifest = pd.read_csv(
    MANIFEST
)

manifest["study_id"] = (
    manifest["study_id"].astype(str)
)

test_split = pd.read_csv(
    TEST_SPLIT
)

test_ids = set(
    test_split["study_id"]
    .astype(str)
)


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


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

dataset = SpineDataset(
    test_df
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model = load_model()


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print()
print("Получаем predictions...")

predictions = predict(
    model,
    loader
)


# ============================================================
# EVALUATE FOUR HEADS
# ============================================================

summary = []


summary.append(
    evaluate_head(
        "FINAL QUALITY",
        test_df["spine_quality"],
        predictions["final"]
    )
)


summary.append(
    evaluate_head(
        "POSITION",
        test_df["spine_position"],
        predictions["position"]
    )
)


summary.append(
    evaluate_head(
        "AXIS",
        test_df["spine_axis"],
        predictions["axis"]
    )
)


summary.append(
    evaluate_head(
        "ARTIFACTS",
        test_df["spine_artifacts"],
        predictions["artifacts"]
    )
)


# ============================================================
# DETAILED TABLE
# ============================================================

details = test_df[
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


details[
    "final_probability"
] = predictions["final"]


details[
    "final_prediction"
] = (
    details["final_probability"]
    >= 0.50
).astype(int)


details[
    "position_probability"
] = predictions["position"]


details[
    "position_prediction"
] = (
    details["position_probability"]
    >= 0.50
).astype(int)


details[
    "axis_probability"
] = predictions["axis"]


details[
    "axis_prediction"
] = (
    details["axis_probability"]
    >= 0.50
).astype(int)


details[
    "artifacts_probability"
] = predictions["artifacts"]


details[
    "artifacts_prediction"
] = (
    details["artifacts_probability"]
    >= 0.50
).astype(int)


# ============================================================
# FINAL ERROR + AUX ERRORS
# ============================================================

details[
    "final_correct"
] = (
    details["final_prediction"]
    ==
    details["spine_quality"]
)


details[
    "position_correct"
] = (
    details["position_prediction"]
    ==
    details["spine_position"]
)


details[
    "axis_correct"
] = (
    details["axis_prediction"]
    ==
    details["spine_axis"]
)


details[
    "artifacts_correct"
] = (
    details["artifacts_prediction"]
    ==
    details["spine_artifacts"]
)


# ============================================================
# FINAL ERROR VS AUXILIARY ERRORS
# ============================================================

def aux_error_count(row):

    count = 0

    if not pd.isna(
        row["spine_position"]
    ):

        if not row["position_correct"]:
            count += 1

    if not pd.isna(
        row["spine_axis"]
    ):

        if not row["axis_correct"]:
            count += 1

    if not pd.isna(
        row["spine_artifacts"]
    ):

        if not row["artifacts_correct"]:
            count += 1

    return count


details[
    "aux_error_count"
] = details.apply(
    aux_error_count,
    axis=1
)


# ============================================================
# PRINT EACH TEST STUDY
# ============================================================

print()
print("=" * 80)
print("DETAILED TEST ANALYSIS")
print("=" * 80)


for _, row in details.iterrows():

    print()
    print(
        f"Study: {row['study_id']}"
    )

    print(
        f"TRUE FINAL: "
        f"{int(row['spine_quality'])}"
    )

    print(
        f"FINAL      : "
        f"{int(row['final_prediction'])} "
        f"(p={row['final_probability']:.3f}) "
        f"{'OK' if row['final_correct'] else 'ERROR'}"
    )

    print(
        f"POSITION   : "
        f"true={int(row['spine_position'])} "
        f"pred={int(row['position_prediction'])} "
        f"p={row['position_probability']:.3f} "
        f"{'OK' if row['position_correct'] else 'ERROR'}"
    )

    print(
        f"AXIS       : "
        f"true={int(row['spine_axis'])} "
        f"pred={int(row['axis_prediction'])} "
        f"p={row['axis_probability']:.3f} "
        f"{'OK' if row['axis_correct'] else 'ERROR'}"
    )

    print(
        f"ARTIFACTS  : "
        f"true={int(row['spine_artifacts'])} "
        f"pred={int(row['artifacts_prediction'])} "
        f"p={row['artifacts_probability']:.3f} "
        f"{'OK' if row['artifacts_correct'] else 'ERROR'}"
    )

    print(
        f"Auxiliary errors: "
        f"{int(row['aux_error_count'])}"
    )

    if pd.notna(row["comment"]):

        comment = str(
            row["comment"]
        ).strip()

        if comment:

            print(
                f"Comment: {comment}"
            )


# ============================================================
# FINAL ERROR RELATION
# ============================================================

print()
print("=" * 80)
print("FINAL ERROR vs AUXILIARY ERRORS")
print("=" * 80)


correct_final = details[
    details["final_correct"]
]

wrong_final = details[
    ~details["final_correct"]
]


print()
print(
    "FINAL correct:"
)

print(
    f"Samples: {len(correct_final)}"
)

print(
    "Auxiliary errors distribution:"
)

print(
    correct_final[
        "aux_error_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print(
    "FINAL incorrect:"
)

print(
    f"Samples: {len(wrong_final)}"
)

print(
    "Auxiliary errors distribution:"
)

print(
    wrong_final[
        "aux_error_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

RESULTS_DIR.mkdir(
    exist_ok=True
)


summary_df = pd.DataFrame(
    summary
)


summary_path = (
    RESULTS_DIR
    / "v13_multitask_heads_test_summary.csv"
)


details_path = (
    RESULTS_DIR
    / "v13_multitask_heads_test_details.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


details.to_csv(
    details_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(summary_path)
print(details_path)

print()
print("=" * 80)
print("ГОТОВО")
print("=" * 80)