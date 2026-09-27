from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

import pydicom

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    / "spine_quality_resnet18_v8_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 20

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
# PIXEL DATA HASH
# ============================================================

def get_pixel_hash(path):

    ds = pydicom.dcmread(
        str(path),
        force=True
    )

    return hash(ds.PixelData)


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe():

    df = pd.read_csv(MANIFEST_PATH)

    print("Исходных строк manifest:", len(df))

    # --------------------------------------------------------
    # Только позвоночник
    # --------------------------------------------------------

    df = df[
        df["anatomy"].astype(str).str.lower() == "spine"
    ].copy()

    print(
        "После anatomy=spine:",
        len(df)
    )

    # --------------------------------------------------------
    # Основные метки
    # --------------------------------------------------------

    label_columns = [
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "spine_quality",
    ]

    for col in label_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Итоговая метка
    df = df[
        df["spine_quality"].isin([0, 1])
    ].copy()

    print(
        "После проверки spine_quality:",
        len(df)
    )

    # --------------------------------------------------------
    # Пути
    # --------------------------------------------------------

    df["dicom_path"] = df[
        "dicom_path"
    ].astype(str)

    # --------------------------------------------------------
    # Загружаем существующие split
    # --------------------------------------------------------

    train_split = pd.read_csv(
        TRAIN_SPLIT_PATH
    )

    val_split = pd.read_csv(
        VAL_SPLIT_PATH
    )

    train_studies = set(
        train_split["study_id"].astype(str)
    )

    val_studies = set(
        val_split["study_id"].astype(str)
    )

    df["study_id"] = df[
        "study_id"
    ].astype(str)

    train_df = df[
        df["study_id"].isin(train_studies)
    ].copy()

    val_df = df[
        df["study_id"].isin(val_studies)
    ].copy()

    print()
    print("До удаления PixelData-дубликатов:")

    print(
        "TRAIN:",
        len(train_df)
    )

    print(
        "VALIDATION:",
        len(val_df)
    )

    # ========================================================
    # REMOVE EXACT PIXEL DUPLICATES
    # ========================================================

    def deduplicate(dataframe):

        dataframe = dataframe.copy()

        hashes = []

        for path in dataframe["dicom_path"]:

            full_path = PROJECT_ROOT / path

            try:

                pixel_hash = get_pixel_hash(
                    full_path
                )

            except Exception as e:

                print(
                    "Ошибка чтения:",
                    full_path
                )

                raise e

            hashes.append(pixel_hash)

        dataframe["pixel_hash"] = hashes

        dataframe = dataframe.drop_duplicates(
            subset=["pixel_hash"],
            keep="first"
        ).copy()

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
    print("После удаления PixelData-дубликатов:")

    print(
        "TRAIN:",
        len(train_df)
    )

    print(
        "VALIDATION:",
        len(val_df)
    )

    # --------------------------------------------------------
    # Проверка пересечения исследований
    # --------------------------------------------------------

    train_ids = set(
        train_df["study_id"]
    )

    val_ids = set(
        val_df["study_id"]
    )

    intersection = train_ids & val_ids

    print(
        "TRAIN ∩ VALIDATION:",
        len(intersection)
    )

    if intersection:

        raise RuntimeError(
            "Обнаружено пересечение TRAIN и VALIDATION"
        )

    # --------------------------------------------------------
    # Вывод классов
    # --------------------------------------------------------

    print()
    print("TRAIN classes:")

    print(
        train_df[
            "spine_quality"
        ].value_counts().sort_index()
    )

    print()
    print("VALIDATION classes:")

    print(
        val_df[
            "spine_quality"
        ].value_counts().sort_index()
    )

    return train_df, val_df


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_dicom(path):

    ds = pydicom.dcmread(
        str(path),
        force=True
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    # Percentile normalization
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
# DATASET
# ============================================================

class SpineMultiTaskDataset(Dataset):

    def __init__(
        self,
        dataframe,
        train=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.train = train

        if train:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),

                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03)
                ),

                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            contrast=0.05
                        )
                    ],
                    p=0.5
                ),

                transforms.ToTensor()
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),

                transforms.ToTensor()
            ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = PROJECT_ROOT / row[
            "dicom_path"
        ]

        image = preprocess_dicom(
            path
        )

        image = self.transform(
            image
        )

        # 1 канал -> 3 канала
        image = image.repeat(
            3,
            1,
            1
        )

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(
            3, 1, 1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(
            3, 1, 1
        )

        image = (
            image - mean
        ) / std

        # ----------------------------------------------------
        # Auxiliary labels
        # ----------------------------------------------------

        position = row[
            "spine_position"
        ]

        axis = row[
            "spine_axis"
        ]

        artifacts = row[
            "spine_artifacts"
        ]

        quality = row[
            "spine_quality"
        ]

        # Если вспомогательная метка отсутствует,
        # используем -1.
        position = (
            int(position)
            if position in [0, 1]
            else -1
        )

        axis = (
            int(axis)
            if axis in [0, 1]
            else -1
        )

        artifacts = (
            int(artifacts)
            if artifacts in [0, 1]
            else -1
        )

        quality = int(
            quality
        )

        return (
            image,
            torch.tensor(
                position,
                dtype=torch.long
            ),
            torch.tensor(
                axis,
                dtype=torch.long
            ),
            torch.tensor(
                artifacts,
                dtype=torch.long
            ),
            torch.tensor(
                quality,
                dtype=torch.long
            )
        )


# ============================================================
# MODEL
# ============================================================

class MultiTaskResNet18(nn.Module):

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

        self.dropout = nn.Dropout(
            0.30
        )

        self.position_head = nn.Linear(
            features,
            2
        )

        self.axis_head = nn.Linear(
            features,
            2
        )

        self.artifacts_head = nn.Linear(
            features,
            2
        )

        self.quality_head = nn.Linear(
            features,
            2
        )

    def forward(self, x):

        features = self.backbone(
            x
        )

        features = self.dropout(
            features
        )

        position = self.position_head(
            features
        )

        axis = self.axis_head(
            features
        )

        artifacts = self.artifacts_head(
            features
        )

        quality = self.quality_head(
            features
        )

        return (
            position,
            axis,
            artifacts,
            quality
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
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

    all_true = []
    all_pred = []

    with torch.no_grad():

        for (
            images,
            position,
            axis,
            artifacts,
            quality
        ) in loader:

            images = images.to(
                device
            )

            (
                position_logits,
                axis_logits,
                artifacts_logits,
                quality_logits
            ) = model(images)

            predictions = (
                torch.argmax(
                    quality_logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

            true_values = (
                quality
                .numpy()
            )

            all_true.extend(
                true_values
            )

            all_pred.extend(
                predictions
            )

    return calculate_metrics(
        all_true,
        all_pred
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
        "BASELINE №8 — MULTI-TASK RESNET18"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_df, val_df = (
        prepare_dataframe()
    )

    train_dataset = (
        SpineMultiTaskDataset(
            train_df,
            train=True
        )
    )

    val_dataset = (
        SpineMultiTaskDataset(
            val_df,
            train=False
        )
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

    model = MultiTaskResNet18()

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    final_counts = (
        train_df[
            "spine_quality"
        ]
        .value_counts()
        .sort_index()
    )

    position_counts = (
        train_df[
            "spine_position"
        ]
        .dropna()
        .value_counts()
        .sort_index()
    )

    axis_counts = (
        train_df[
            "spine_axis"
        ]
        .dropna()
        .value_counts()
        .sort_index()
    )

    artifact_counts = (
        train_df[
            "spine_artifacts"
        ]
        .dropna()
        .value_counts()
        .sort_index()
    )

    def make_weights(counts):

        total = counts.sum()

        weights = []

        for cls in [0, 1]:

            if cls in counts:

                weights.append(
                    total
                    / (
                        2 * counts[cls]
                    )
                )

            else:

                weights.append(
                    1.0
                )

        return torch.tensor(
            weights,
            dtype=torch.float32,
            device=device
        )

    final_weights = make_weights(
        final_counts
    )

    position_weights = make_weights(
        position_counts
    )

    axis_weights = make_weights(
        axis_counts
    )

    artifact_weights = make_weights(
        artifact_counts
    )

    print()
    print(
        "Final class weights:",
        final_weights
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion_final = nn.CrossEntropyLoss(
        weight=final_weights
    )

    criterion_position = nn.CrossEntropyLoss(
        weight=position_weights,
        ignore_index=-1
    )

    criterion_axis = nn.CrossEntropyLoss(
        weight=axis_weights,
        ignore_index=-1
    )

    criterion_artifacts = nn.CrossEntropyLoss(
        weight=artifact_weights,
        ignore_index=-1
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=2e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_f1 = -1.0

    best_accuracy = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        train_true = []

        train_pred = []

        for (
            images,
            position,
            axis,
            artifacts,
            quality
        ) in train_loader:

            images = images.to(
                device
            )

            position = position.to(
                device
            )

            axis = axis.to(
                device
            )

            artifacts = artifacts.to(
                device
            )

            quality = quality.to(
                device
            )

            optimizer.zero_grad()

            (
                position_logits,
                axis_logits,
                artifacts_logits,
                quality_logits
            ) = model(images)

            loss_position = (
                criterion_position(
                    position_logits,
                    position
                )
            )

            loss_axis = (
                criterion_axis(
                    axis_logits,
                    axis
                )
            )

            loss_artifacts = (
                criterion_artifacts(
                    artifacts_logits,
                    artifacts
                )
            )

            loss_final = (
                criterion_final(
                    quality_logits,
                    quality
                )
            )

            # ------------------------------------------------
            # Multi-task loss
            # ------------------------------------------------

            loss = (
                0.15 * loss_position
                + 0.20 * loss_axis
                + 0.20 * loss_artifacts
                + 1.00 * loss_final
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                torch.argmax(
                    quality_logits,
                    dim=1
                )
            )

            train_true.extend(
                quality.detach()
                .cpu()
                .numpy()
            )

            train_pred.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

        train_loss = (
            total_loss
            / len(train_dataset)
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

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step(
            val_f1
        )

        print()
        print(
            f"Epoch {epoch}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Train acc: {train_accuracy * 100:.2f}%"
        )

        print(
            f"Train class1 precision: "
            f"{train_precision * 100:.2f}%"
        )

        print(
            f"Train class1 recall: "
            f"{train_recall * 100:.2f}%"
        )

        print(
            f"Train class1 F1: "
            f"{train_f1 * 100:.2f}%"
        )

        print(
            f"Val acc: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val class1 precision: "
            f"{val_precision * 100:.2f}%"
        )

        print(
            f"Val class1 recall: "
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

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_accuracy":
                        val_accuracy,

                    "val_f1":
                        val_f1,
                },
                MODEL_PATH
            )

            print(
                "BEST"
            )

        else:

            epochs_without_improvement += 1

            print(
                "no improve"
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping"
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "======================================"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation class1 F1: "
        f"{best_f1 * 100:.2f}%"
    )

    print(
        "Model saved:",
        MODEL_PATH
    )

    print(
        "======================================"
    )


if __name__ == "__main__":

    main()