from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

TRAIN_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VAL_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

TEST_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spine_quality_resnet18_v9_best.pth"
)


IMAGE_SIZE = 224
BATCH_SIZE = 8


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

    image = (
        image * 255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        image,
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
# DATASET
# ============================================================

class SpineDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        path = (
            PROJECT_ROOT
            / row["dicom_path"]
        )

        image = load_dicom(
            path
        )

        image = resize_with_padding(
            image,
            IMAGE_SIZE
        )

        image = transforms.ToTensor()(
            image
        )

        image = image.repeat(
            3,
            1,
            1
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(
            3,
            1,
            1
        )

        image = (
            image - mean
        ) / std

        label = int(
            row["spine_quality"]
        )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# MODEL
# ============================================================

class V9Model(nn.Module):

    def __init__(self):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
        )

        self.backbone = models.resnet18(
            weights=weights
        )

        features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        for parameter in (
            self.backbone.parameters()
        ):

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
# PREPARE SPLIT
# ============================================================

def prepare_split(
    split_path
):

    df = pd.read_csv(
        MANIFEST_PATH
    )

    split = pd.read_csv(
        split_path
    )

    split_ids = set(
        split["study_id"].astype(str)
    )

    df["study_id"] = df[
        "study_id"
    ].astype(str)

    df["spine_quality"] = pd.to_numeric(
        df["spine_quality"],
        errors="coerce"
    )

    df = df[
        (df["anatomy"] == "spine")
        & df["spine_quality"].isin([0, 1])
        & df["study_id"].isin(split_ids)
    ].copy()

    # --------------------------------------------------------
    # Exact PixelData deduplication
    # --------------------------------------------------------

    hashes = []

    for _, row in df.iterrows():

        path = (
            PROJECT_ROOT
            / row["dicom_path"]
        )

        ds = pydicom.dcmread(
            str(path),
            force=True
        )

        hashes.append(
            hash(ds.PixelData)
        )

    df["pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    ).copy()

    df = df.drop(
        columns=["pixel_hash"]
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(
    model,
    dataframe,
    device
):

    dataset = SpineDataset(
        dataframe
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    true_values = []

    probabilities = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            logits = model(
                images
            )

            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            true_values.extend(
                labels.numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    return (
        np.array(true_values),
        np.array(probabilities)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "V9 THRESHOLD ANALYSIS"
    )

    print(
        "=" * 70
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = V9Model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Модель загружена."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_df = prepare_split(
        VAL_SPLIT_PATH
    )

    print()
    print(
        "Validation:",
        len(val_df)
    )

    val_true, val_prob = get_predictions(
        model,
        val_df,
        device
    )

    # --------------------------------------------------------
    # Search threshold
    # --------------------------------------------------------

    thresholds = np.arange(
        0.20,
        0.81,
        0.05
    )

    results = []

    for threshold in thresholds:

        predictions = (
            val_prob >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            val_true,
            predictions
        )

        precision = precision_score(
            val_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            val_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            val_true,
            predictions,
            zero_division=0
        )

        results.append(
            (
                threshold,
                accuracy,
                precision,
                recall,
                f1
            )
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    print()
    print(
        "VALIDATION — ПОДБОР ПОРОГА"
    )

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "accuracy": "{:.2f}".format,
                "precision": "{:.2f}".format,
                "recall": "{:.2f}".format,
                "f1": "{:.2f}".format
            }
        )
    )

    # --------------------------------------------------------
    # Select threshold ONLY using validation F1
    # --------------------------------------------------------

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print()
    print(
        "Выбранный порог по VALIDATION:"
    )

    print(
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_row['f1'] * 100:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{best_row['accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_df = prepare_split(
        TEST_SPLIT_PATH
    )

    print()
    print(
        "TEST:",
        len(test_df)
    )

    test_true, test_prob = get_predictions(
        model,
        test_df,
        device
    )

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    test_accuracy = accuracy_score(
        test_true,
        test_pred
    )

    test_precision = precision_score(
        test_true,
        test_pred,
        zero_division=0
    )

    test_recall = recall_score(
        test_true,
        test_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        test_true,
        test_pred,
        zero_division=0
    )

    print()
    print(
        "RESULTS TEST С ПОРОГОМ,"
        " ВЫБРАННЫМ ПО VALIDATION"
    )

    print(
        f"Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Class 1 precision: "
        f"{test_precision * 100:.2f}%"
    )

    print(
        f"Class 1 recall: "
        f"{test_recall * 100:.2f}%"
    )

    print(
        f"Class 1 F1: "
        f"{test_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion
    # --------------------------------------------------------

    tn = int(
        ((test_true == 0) & (test_pred == 0)).sum()
    )

    fp = int(
        ((test_true == 0) & (test_pred == 1)).sum()
    )

    fn = int(
        ((test_true == 1) & (test_pred == 0)).sum()
    )

    tp = int(
        ((test_true == 1) & (test_pred == 1)).sum()
    )

    print()
    print(
        "Confusion:"
    )

    print(
        f"True0 pred0={tn} pred1={fp}"
    )

    print(
        f"True1 pred0={fn} pred1={tp}"
    )

    print()
    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()