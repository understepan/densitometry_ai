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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "spine_quality_resnet18_v10_best.pth"
)


IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20


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

class SpineMultiTaskDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        training=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.training = training

        if training:

            self.augmentation = transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(
                            0.03,
                            0.03
                        ),
                        scale=(
                            0.97,
                            1.03
                        )
                    ),
                    transforms.ColorJitter(
                        contrast=0.05
                    )
                ]
            )

        else:

            self.augmentation = None

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

        if self.augmentation is not None:

            image = self.augmentation(
                image
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

        position = int(
            row["spine_position"]
        )

        axis = int(
            row["spine_axis"]
        )

        artifacts = int(
            row["spine_artifacts"]
        )

        final = int(
            row["spine_quality"]
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
                final,
                dtype=torch.long
            )
        )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_split(
    split_path
):

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    split = pd.read_csv(
        split_path
    )

    split_ids = set(
        split["study_id"].astype(str)
    )

    manifest["study_id"] = (
        manifest["study_id"]
        .astype(str)
    )

    numeric_columns = [
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "spine_quality"
    ]

    for column in numeric_columns:

        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce"
        )

    df = manifest[
        (manifest["anatomy"] == "spine")
        & manifest["study_id"].isin(
            split_ids
        )
        & manifest[
            numeric_columns
        ].notna().all(axis=1)
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
# MODEL
# ============================================================

class V10Model(nn.Module):

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

        # Сначала backbone полностью заморожен
        for parameter in (
            self.backbone.parameters()
        ):

            parameter.requires_grad = False

        # ----------------------------------------------------
        # Четыре головы
        # ----------------------------------------------------

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

        self.final_head = nn.Sequential(

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

        position = self.position_head(
            features
        )

        axis = self.axis_head(
            features
        )

        artifacts = self.artifacts_head(
            features
        )

        final = self.final_head(
            features
        )

        return (
            position,
            axis,
            artifacts,
            final
        )


# ============================================================
# METRICS
# ============================================================

def binary_metrics(
    true,
    pred
):

    return {
        "accuracy": accuracy_score(
            true,
            pred
        ),

        "precision": precision_score(
            true,
            pred,
            zero_division=0
        ),

        "recall": recall_score(
            true,
            pred,
            zero_division=0
        ),

        "f1": f1_score(
            true,
            pred,
            zero_division=0
        )
    }


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    criterion,
    training
):

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0

    true_final = []
    pred_final = []

    for (
        images,
        position,
        axis,
        artifacts,
        final
    ) in loader:

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

        final = final.to(
            device
        )

        if training:

            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):

            (
                position_logits,
                axis_logits,
                artifacts_logits,
                final_logits
            ) = model(
                images
            )

            loss_position = criterion(
                position_logits,
                position
            )

            loss_axis = criterion(
                axis_logits,
                axis
            )

            loss_artifacts = criterion(
                artifacts_logits,
                artifacts
            )

            loss_final = criterion(
                final_logits,
                final
            )

            # Главная задача имеет
            # значительно больший вес
            loss = (
                1.00 * loss_final
                + 0.10 * loss_position
                + 0.10 * loss_axis
                + 0.10 * loss_artifacts
            )

            if training:

                loss.backward()

                optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            final_logits,
            dim=1
        )

        true_final.extend(
            final.cpu().numpy()
        )

        pred_final.extend(
            predictions.cpu().numpy()
        )

    total_loss /= len(
        loader.dataset
    )

    metrics = binary_metrics(
        true_final,
        pred_final
    )

    return (
        total_loss,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "BASELINE №10 — MULTI-TASK RESNET18 V2"
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
    # DATA
    # --------------------------------------------------------

    train_df = prepare_split(
        TRAIN_SPLIT_PATH
    )

    val_df = prepare_split(
        VAL_SPLIT_PATH
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
        "TRAIN final classes:"
    )

    print(
        train_df[
            "spine_quality"
        ].value_counts()
        .sort_index()
    )

    print()
    print(
        "VALIDATION final classes:"
    )

    print(
        val_df[
            "spine_quality"
        ].value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = SpineMultiTaskDataset(
        train_df,
        training=True
    )

    val_dataset = SpineMultiTaskDataset(
        val_df,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = V10Model()

    model = model.to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Initially only heads are trainable
    # --------------------------------------------------------

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_f1 = -1
    best_accuracy = -1
    best_epoch = 0

    patience = 6
    no_improve = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # После 5 эпох осторожно размораживаем layer4
        # ----------------------------------------------------

        if epoch == 6:

            print()
            print(
                "Размораживаем layer4..."
            )

            for parameter in (
                model.backbone.layer4.parameters()
            ):

                parameter.requires_grad = True

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params":
                        model.backbone.layer4.parameters(),
                        "lr": 1e-5
                    },
                    {
                        "params":
                        model.position_head.parameters(),
                        "lr": 1e-4
                    },
                    {
                        "params":
                        model.axis_head.parameters(),
                        "lr": 1e-4
                    },
                    {
                        "params":
                        model.artifacts_head.parameters(),
                        "lr": 1e-4
                    },
                    {
                        "params":
                        model.final_head.parameters(),
                        "lr": 1e-4
                    }
                ],
                weight_decay=1e-4
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2
            )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            training=True
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            criterion,
            training=False
        )

        scheduler.step(
            val_metrics["f1"]
        )

        print()
        print(
            f"Epoch {epoch}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train acc: "
            f"{train_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Train class1 precision: "
            f"{train_metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Train class1 recall: "
            f"{train_metrics['recall'] * 100:.2f}%"
        )

        print(
            f"Train class1 F1: "
            f"{train_metrics['f1'] * 100:.2f}%"
        )

        print(
            f"Val loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val acc: "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Val class1 precision: "
            f"{val_metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Val class1 recall: "
            f"{val_metrics['recall'] * 100:.2f}%"
        )

        print(
            f"Val class1 F1: "
            f"{val_metrics['f1'] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        improved = (

            val_metrics["f1"]
            > best_f1

            or

            (
                val_metrics["f1"]
                == best_f1

                and

                val_metrics["accuracy"]
                > best_accuracy
            )
        )

        if improved:

            best_f1 = val_metrics["f1"]

            best_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            no_improve = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_f1":
                        best_f1,

                    "val_accuracy":
                        best_accuracy
                },
                MODEL_PATH
            )

            print(
                "BEST"
            )

        else:

            no_improve += 1

            print(
                "no improve"
            )

        if no_improve >= patience:

            print()
            print(
                "Early stopping"
            )

            break

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        f"Best epoch: {best_epoch}"
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
        f"Model saved: {MODEL_PATH}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()