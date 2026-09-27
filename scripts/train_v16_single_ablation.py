from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\hakaton\densitometry_ai")

MANIFEST = ROOT / "data" / "processed" / "labeled_manifest.csv"
TRAIN_SPLIT = ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT = ROOT / "data" / "splits" / "validation.csv"

MODEL_PATH = ROOT / "models" / "spine_quality_resnet18_v16_ablation_best.pth"

SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 7
IMAGE_SIZE = 224

DEVICE = torch.device("cpu")


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# IMAGE
# ============================================================

def resolve_path(path):

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}"
        )

    return path


def normalize_percentile(arr):

    arr = arr.astype(np.float32)

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:

        minimum = arr.min()
        maximum = arr.max()

        if maximum > minimum:
            arr = (arr - minimum) / (maximum - minimum)
        else:
            arr = np.zeros_like(arr)

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

    arr = normalize_percentile(arr)

    arr = (
        arr * 255
    ).astype(np.uint8)

    return Image.fromarray(arr)


# ============================================================
# EXACT PIXEL DEDUP
# ============================================================

def pixel_hash(path):

    path = resolve_path(path)

    ds = pydicom.dcmread(path)

    return hashlib.md5(
        ds.pixel_array.tobytes()
    ).hexdigest()


def exact_dedup(df):

    df = df.copy()

    hashes = []

    for path in df["dicom_path"]:
        hashes.append(
            pixel_hash(path)
        )

    df["pixel_hash"] = hashes

    df = (
        df
        .drop_duplicates(
            subset=["pixel_hash"],
            keep="first"
        )
        .drop(columns=["pixel_hash"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# DATASET
# ============================================================

class SpineDataset(Dataset):

    def __init__(
        self,
        df,
        train=False
    ):

        self.df = (
            df.reset_index(drop=True)
        )

        self.train = train

        if train:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                ),

                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03)
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

        else:

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

        image = load_image(
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
# MODEL
# ============================================================

class V16Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze entire pretrained backbone.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

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
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold=0.50
):

    y_pred = (
        np.asarray(y_prob)
        >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    losses = []
    y_true = []
    y_prob = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            losses.append(
                loss.item()
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_prob.extend(
                probabilities.cpu().numpy()
            )

    metrics = calculate_metrics(
        y_true,
        y_prob,
        threshold=0.50
    )

    metrics["loss"] = float(
        np.mean(losses)
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("V16 — SINGLE-TASK ABLATION")
print("=" * 80)

print(
    "Цель: проверить V13 без auxiliary losses."
)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# READ DATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

manifest["study_id"] = (
    manifest["study_id"]
    .astype(str)
)


train_split = pd.read_csv(
    TRAIN_SPLIT
)

val_split = pd.read_csv(
    VAL_SPLIT
)

train_ids = set(
    train_split["study_id"]
    .astype(str)
)

val_ids = set(
    val_split["study_id"]
    .astype(str)
)


# ============================================================
# FILTER SPINE
# ============================================================

train_df = manifest[
    (manifest["anatomy"] == "spine")
    &
    (manifest["spine_quality"].notna())
    &
    (manifest["study_id"].isin(train_ids))
].copy()


val_df = manifest[
    (manifest["anatomy"] == "spine")
    &
    (manifest["spine_quality"].notna())
    &
    (manifest["study_id"].isin(val_ids))
].copy()


print()
print(
    f"TRAIN before dedup: "
    f"{len(train_df)}"
)

print(
    f"VALIDATION before dedup: "
    f"{len(val_df)}"
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


print()
print(
    f"TRAIN after dedup: "
    f"{len(train_df)}"
)

print(
    f"VALIDATION after dedup: "
    f"{len(val_df)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_counts = (
    train_df["spine_quality"]
    .astype(int)
    .value_counts()
    .sort_index()
)

val_counts = (
    val_df["spine_quality"]
    .astype(int)
    .value_counts()
    .sort_index()
)


print()
print(
    "TRAIN classes:"
)

print(
    f"0 = {train_counts.get(0, 0)}"
)

print(
    f"1 = {train_counts.get(1, 0)}"
)


print()
print(
    "VALIDATION classes:"
)

print(
    f"0 = {val_counts.get(0, 0)}"
)

print(
    f"1 = {val_counts.get(1, 0)}"
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = SpineDataset(
    train_df,
    train=True
)

val_dataset = SpineDataset(
    val_df,
    train=False
)


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

train_labels = (
    train_df["spine_quality"]
    .astype(int)
    .to_numpy()
)

class_counts = np.bincount(
    train_labels,
    minlength=2
)

sample_weights = (
    1.0 /
    class_counts[train_labels]
)

sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
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


# ============================================================
# CLASS WEIGHTS
# ============================================================

total = len(train_labels)

class_weights = torch.tensor(
    [
        total / (2 * class_counts[0]),
        total / (2 * class_counts[1])
    ],
    dtype=torch.float32
).to(DEVICE)


print()
print(
    "Class weights:",
    class_weights
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


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


# ============================================================
# TRAIN
# ============================================================

best_f1 = -1
best_accuracy = -1
best_epoch = 0
patience_counter = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    train_losses = []
    train_true = []
    train_prob = []

    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
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

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        train_losses.append(
            loss.item()
        )

        train_true.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        train_prob.extend(
            probabilities.detach()
            .cpu()
            .numpy()
        )


    train_metrics = calculate_metrics(
        train_true,
        train_prob,
        threshold=0.50
    )


    val_metrics = evaluate(
        model,
        val_loader,
        criterion
    )


    scheduler.step(
        val_metrics["f1"]
    )


    print(
        f"Epoch {epoch:02d} | "
        f"Train loss {np.mean(train_losses):.4f} | "
        f"Train acc "
        f"{train_metrics['accuracy'] * 100:.2f}% | "
        f"Train F1 "
        f"{train_metrics['f1'] * 100:.2f}% | "
        f"Val loss "
        f"{val_metrics['loss']:.4f} | "
        f"Val acc "
        f"{val_metrics['accuracy'] * 100:.2f}% | "
        f"Val precision "
        f"{val_metrics['precision'] * 100:.2f}% | "
        f"Val recall "
        f"{val_metrics['recall'] * 100:.2f}% | "
        f"Val F1 "
        f"{val_metrics['f1'] * 100:.2f}%"
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    is_better = (

        val_metrics["f1"] > best_f1

        or (

            val_metrics["f1"] == best_f1
            and
            val_metrics["accuracy"]
            > best_accuracy

        )
    )


    if is_better:

        best_f1 = (
            val_metrics["f1"]
        )

        best_accuracy = (
            val_metrics["accuracy"]
        )

        best_epoch = epoch

        patience_counter = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],

                "val_precision":
                    val_metrics[
                        "precision"
                    ],

                "val_recall":
                    val_metrics[
                        "recall"
                    ],

                "val_f1":
                    val_metrics[
                        "f1"
                    ]
            },
            MODEL_PATH
        )

        print(
            "  -> BEST CHECKPOINT SAVED"
        )

    else:

        patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "Early stopping"
            )

            break


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("TRAINING FINISHED")
print("=" * 80)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Best validation F1: "
    f"{best_f1 * 100:.2f}%"
)

print()
print(
    f"Model saved:"
)

print(
    MODEL_PATH
)

print()
print("=" * 80)
print("ГОТОВО")
print("=" * 80)