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
    f1_score,
    confusion_matrix
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

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "spine_quality_v9_exact_predictions.csv"
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
# RESIZE WITH PADDING
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

class SpineTestDataset(Dataset):

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
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "EVALUATION V9 — EXACT UNIQUE TEST"
    )

    print(
        "=" * 70
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    test_split = pd.read_csv(
        TEST_SPLIT_PATH
    )

    test_ids = set(
        test_split[
            "study_id"
        ].astype(str)
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
        & df["study_id"].isin(test_ids)
    ].copy()

    print(
        "TEST после фильтрации spine:",
        len(df)
    )

    print(
        "Class 0:",
        (
            df["spine_quality"] == 0
        ).sum()
    )

    print(
        "Class 1:",
        (
            df["spine_quality"] == 1
        ).sum()
    )

    # --------------------------------------------------------
    # Exact PixelData deduplication
    # --------------------------------------------------------

    print()
    print(
        "Проверка точных дубликатов PixelData..."
    )

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

    before = len(df)

    df = df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    ).copy()

    after = len(df)

    print(
        "До удаления:",
        before
    )

    print(
        "После удаления:",
        after
    )

    print(
        "Удалено дубликатов:",
        before - after
    )

    print()
    print(
        "Финальный TEST:"
    )

    print(
        "Уникальных изображений:",
        len(df)
    )

    print(
        "Уникальных исследований:",
        df["study_id"].nunique()
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SpineTestDataset(
        df
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
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
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print()
    print(
        "Модель успешно загружена."
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    precision = precision_score(
        all_true,
        all_pred,
        zero_division=0
    )

    recall = recall_score(
        all_true,
        all_pred,
        zero_division=0
    )

    f1 = f1_score(
        all_true,
        all_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=[0, 1]
    )

    print()
    print(
        "RESULTS V9"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Class 1 precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Class 1 recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Class 1 F1: "
        f"{f1 * 100:.2f}%"
    )

    print()
    print(
        "Confusion:"
    )

    print(
        f"True0 pred0={matrix[0, 0]} "
        f"pred1={matrix[0, 1]}"
    )

    print(
        f"True1 pred0={matrix[1, 0]} "
        f"pred1={matrix[1, 1]}"
    )

    # --------------------------------------------------------
    # Detailed predictions
    # --------------------------------------------------------

    result_df = df[
        [
            "study_id",
            "dicom_path",
            "spine_quality"
        ]
    ].copy()

    result_df["prediction"] = (
        all_pred
    )

    result_df["probability_class1"] = (
        all_prob
    )

    result_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print()
    print(
        "Подробные предсказания сохранены:"
    )

    print(
        RESULT_PATH
    )

    print()
    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()