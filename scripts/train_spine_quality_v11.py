from pathlib import Path
import random

import numpy as np
import pandas as pd
import pydicom

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.models import ResNet18_Weights

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spine_quality_resnet18_v11_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 25

PATIENCE = 7

NUM_WORKERS = 0


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


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
    # Percentile normalization — как в V9
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
# CENTRAL ROI
# ============================================================

def crop_central_roi(
    image
):
    """
    Убираем часть бокового контекста.

    Важно:
    - верх изображения сохраняем;
    - низ изображения сохраняем;
    - обрезаем только левую и правую части.

    Это НЕ медицинская сегментация позвоночника.
    Это контролируемый эксперимент с уменьшением
    нерелевантного бокового контекста.
    """

    width, height = image.size

    # Сохраняем центральные 70% ширины.
    left = int(
        width * 0.15
    )

    right = int(
        width * 0.85
    )

    cropped = image.crop(
        (
            left,
            0,
            right,
            height
        )
    )

    return cropped


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

class SpineDataset(Dataset):

    def __init__(
        self,
        dataframe,
        train=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.train = train

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

        # ----------------------------------------------------
        # V11 CHANGE:
        # central ROI
        # ----------------------------------------------------

        image = crop_central_roi(
            image
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = resize_with_padding(
            image,
            IMAGE_SIZE
        )

        # ----------------------------------------------------
        # Gentle translation — как в V9
        # ----------------------------------------------------

        if self.train:

            shift_x = random.randint(
                -4,
                4
            )

            shift_y = random.randint(
                -4,
                4
            )

            shifted = Image.new(
                "L",
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                0
            )

            shifted.paste(
                image,
                (
                    shift_x,
                    shift_y
                )
            )

            image = shifted

        # ----------------------------------------------------
        # To tensor
        # ----------------------------------------------------

        array = np.asarray(
            image
        ).astype(
            np.float32
        ) / 255.0

        image = torch.from_numpy(
            array
        ).unsqueeze(
            0
        )

        # 1 → 3 channels
        image = image.repeat(
            3,
            1,
            1
        )

        # ImageNet normalization
        mean = torch.tensor(
            [
                0.485,
                0.456,
                0.406
            ]
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ]
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
# PREPARE DATA
# ============================================================

def prepare_data():

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        "Исходных строк:",
        len(df)
    )

    # Только spine
    df = df[
        df["anatomy"] == "spine"
    ].copy()

    # Только валидные final labels
    df["spine_quality"] = pd.to_numeric(
        df["spine_quality"],
        errors="coerce"
    )

    df = df[
        df["spine_quality"].isin(
            [0, 1]
        )
    ].copy()

    df["study_id"] = df[
        "study_id"
    ].astype(str)

    # --------------------------------------------------------
    # Existing study-level split
    # --------------------------------------------------------

    train_split = pd.read_csv(
        TRAIN_SPLIT_PATH
    )

    val_split = pd.read_csv(
        VAL_SPLIT_PATH
    )

    train_ids = set(
        train_split[
            "study_id"
        ].astype(str)
    )

    val_ids = set(
        val_split[
            "study_id"
        ].astype(str)
    )

    train_df = df[
        df["study_id"].isin(
            train_ids
        )
    ].copy()

    val_df = df[
        df["study_id"].isin(
            val_ids
        )
    ].copy()

    # --------------------------------------------------------
    # Exact PixelData deduplication
    # --------------------------------------------------------

    def deduplicate(
        dataframe
    ):

        hashes = []

        for _, row in dataframe.iterrows():

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

        dataframe = dataframe.copy()

        dataframe["pixel_hash"] = hashes

        dataframe = dataframe.drop_duplicates(
            subset=["pixel_hash"],
            keep="first"
        )

        dataframe = dataframe.drop(
            columns=["pixel_hash"]
        )

        return dataframe.reset_index(
            drop=True
        )

    train_df = deduplicate(
        train_df
    )

    val_df = deduplicate(
        val_df
    )

    print()
    print(
        "TRAIN:",
        len(train_df)
    )

    print(
        "VALIDATION:",
        len(val_df)
    )

    print()

    print(
        "TRAIN classes:"
    )

    print(
        train_df[
            "spine_quality"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "VALIDATION classes:"
    )

    print(
        val_df[
            "spine_quality"
        ]
        .value_counts()
        .sort_index()
    )

    return (
        train_df,
        val_df
    )


# ============================================================
# MODEL
# ============================================================

class V11Model(nn.Module):

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

        # Backbone frozen — как V9
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

    def forward(
        self,
        x
    ):

        features = self.backbone(
            x
        )

        return self.classifier(
            features
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_values,
    predictions
):

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

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    true_values = []

    predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            logits = model(
                images
            )

            pred = torch.argmax(
                logits,
                dim=1
            )

            true_values.extend(
                labels.numpy()
            )

            predictions.extend(
                pred.cpu().numpy()
            )

    return calculate_metrics(
        true_values,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "BASELINE V11 — CENTRAL ROI"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_df, val_df = (
        prepare_data()
    )

    train_dataset = SpineDataset(
        train_df,
        train=True
    )

    val_dataset = SpineDataset(
        val_df,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = V11Model()

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    counts = (
        train_df[
            "spine_quality"
        ]
        .value_counts()
        .sort_index()
    )

    total = counts.sum()

    weight0 = (
        total
        / (
            2 * counts[0]
        )
    )

    weight1 = (
        total
        / (
            2 * counts[1]
        )
    )

    class_weights = torch.tensor(
        [
            weight0,
            weight1
        ],
        dtype=torch.float32,
        device=device
    )

    print()
    print(
        "Class weights:",
        class_weights
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer — как V9
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-3,
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    best_f1 = -1.0

    best_accuracy = -1.0

    best_epoch = 0

    no_improvement = 0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_true = []

        train_pred = []

        train_loss_sum = 0.0

        train_count = 0

        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                images.size(0)
            )

            train_loss_sum += (
                loss.item()
                * batch_size
            )

            train_count += batch_size

            pred = torch.argmax(
                logits,
                dim=1
            )

            train_true.extend(
                labels.cpu().numpy()
            )

            train_pred.extend(
                pred.cpu().numpy()
            )

        train_loss = (
            train_loss_sum
            / train_count
        )

        (
            train_accuracy,
            train_precision,
            train_recall,
            train_f1
        ) = calculate_metrics(
            train_true,
            train_pred
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_true = []

        val_pred = []

        val_loss_sum = 0.0

        val_count = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device
                )

                labels = labels.to(
                    device
                )

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels
                )

                batch_size = (
                    images.size(0)
                )

                val_loss_sum += (
                    loss.item()
                    * batch_size
                )

                val_count += batch_size

                pred = torch.argmax(
                    logits,
                    dim=1
                )

                val_true.extend(
                    labels.cpu().numpy()
                )

                val_pred.extend(
                    pred.cpu().numpy()
                )

        val_loss = (
            val_loss_sum
            / val_count
        )

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = calculate_metrics(
            val_true,
            val_pred
        )

        scheduler.step(
            val_f1
        )

        print()
        print(
            f"Epoch {epoch:02d}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Train class1 F1: "
            f"{train_f1 * 100:.2f}%"
        )

        print(
            f"Val loss: {val_loss:.4f}"
        )

        print(
            f"Val accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val precision: "
            f"{val_precision * 100:.2f}%"
        )

        print(
            f"Val recall: "
            f"{val_recall * 100:.2f}%"
        )

        print(
            f"Val class1 F1: "
            f"{val_f1 * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        improved = False

        if val_f1 > best_f1:

            improved = True

        elif (
            val_f1 == best_f1
            and val_accuracy > best_accuracy
        ):

            improved = True

        if improved:

            best_f1 = val_f1

            best_accuracy = val_accuracy

            best_epoch = epoch

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_accuracy":
                        val_accuracy,

                    "val_precision":
                        val_precision,

                    "val_recall":
                        val_recall,

                    "val_f1":
                        val_f1
                },
                MODEL_PATH
            )

            print(
                "BEST MODEL SAVED"
            )

        else:

            no_improvement += 1

        if no_improvement >= PATIENCE:

            print()
            print(
                "Early stopping"
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "V11 TRAINING FINISHED"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation accuracy:",
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        "Best validation class1 F1:",
        f"{best_f1 * 100:.2f}%"
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()