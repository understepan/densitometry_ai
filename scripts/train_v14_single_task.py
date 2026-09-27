from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\hakaton\densitometry_ai")

MANIFEST = ROOT / "data" / "processed" / "labeled_manifest.csv"
TRAIN_SPLIT = ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT = ROOT / "data" / "splits" / "validation.csv"

MODEL_PATH = ROOT / "models" / "spine_quality_resnet18_v14_best.pth"

DEVICE = torch.device("cpu")

SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 30
PATIENCE = 7

IMAGE_SIZE = 224


# ============================================================
# SEED
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# IMAGE PREPROCESSING
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
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1)

    arr = np.clip(arr, 0, 1)

    return arr


def load_dicom_image(path):
    import pydicom

    path = Path(path)

    # Пути в manifest относительные.
    # Превращаем их в абсолютные относительно проекта.
    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"DICOM не найден:\n{path}"
        )

    ds = pydicom.dcmread(path)

    arr = ds.pixel_array.astype(np.float32)

    arr = percentile_normalize(arr)

    arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr)


class SpineDataset(Dataset):

    def __init__(self, df, train=False):

        self.df = df.reset_index(drop=True)
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03)
                ),
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1)
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = load_dicom_image(row["dicom_path"])

        image = self.transform(image)

        label = int(row["spine_quality"])

        return image, label


# ============================================================
# EXACT PIXEL DEDUPLICATION
# ============================================================

def exact_dedup(df):

    hashes = []

    import pydicom
    import hashlib

    for path in df["dicom_path"]:

        path = Path(path)

        # В manifest путь хранится относительно проекта.
        # Если путь относительный — добавляем ROOT.
        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"DICOM не найден:\n{path}"
            )

        ds = pydicom.dcmread(path)

        pixel_bytes = ds.pixel_array.tobytes()

        digest = hashlib.md5(
            pixel_bytes
        ).hexdigest()

        hashes.append(digest)

    df = df.copy()

    df["pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    )

    return df.drop(
        columns=["pixel_hash"]
    ).reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V14 SINGLE-TASK TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")

manifest = pd.read_csv(MANIFEST)

train_ids = pd.read_csv(TRAIN_SPLIT)["study_id"].astype(str)
val_ids = pd.read_csv(VAL_SPLIT)["study_id"].astype(str)

manifest["study_id"] = manifest["study_id"].astype(str)

spine_df = manifest[
    (manifest["anatomy"] == "spine") &
    (manifest["spine_quality"].notna())
].copy()

train_df = spine_df[
    spine_df["study_id"].isin(train_ids)
].copy()

val_df = spine_df[
    spine_df["study_id"].isin(val_ids)
].copy()

print()
print(f"Исходных строк manifest: {len(manifest)}")
print(f"Spine строк: {len(spine_df)}")

print()
print(f"TRAIN before dedup: {len(train_df)}")
print(f"VALIDATION before dedup: {len(val_df)}")

print()
print("Удаление точных PixelData-дубликатов...")

train_df = exact_dedup(train_df)
val_df = exact_dedup(val_df)

print()
print("После exact dedup:")
print(f"TRAIN: {len(train_df)}")
print(f"VALIDATION: {len(val_df)}")


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("FINAL LABELS")
print("=" * 70)

print("TRAIN:")
print(train_df["spine_quality"].value_counts().sort_index())

print()
print("VALIDATION:")
print(val_df["spine_quality"].value_counts().sort_index())


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

train_labels = train_df["spine_quality"].astype(int).values

class_counts = np.bincount(
    train_labels,
    minlength=2
)

sample_weights = np.array([
    1.0 / class_counts[label]
    for label in train_labels
])

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

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

weights = ResNet18_Weights.DEFAULT

backbone = models.resnet18(
    weights=weights
)

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

num_features = backbone.fc.in_features

backbone.fc = nn.Identity()


class V14Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = backbone

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

        logits = self.classifier(
            features
        )

        return logits


model = V14Model().to(DEVICE)


# ============================================================
# LOSS
# ============================================================

class_weights = torch.tensor(
    [
        len(train_labels) / (2.0 * class_counts[0]),
        len(train_labels) / (2.0 * class_counts[1])
    ],
    dtype=torch.float32,
    device=DEVICE
)

print()
print("Class weights:")
print(class_weights)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

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

epochs_without_improvement = 0


for epoch in range(1, NUM_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []

    train_true = []
    train_pred = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_true.extend(
            labels.cpu().numpy()
        )

        train_pred.extend(
            predictions.cpu().numpy()
        )

    train_accuracy = accuracy_score(
        train_true,
        train_pred
    )

    train_precision = precision_score(
        train_true,
        train_pred,
        zero_division=0
    )

    train_recall = recall_score(
        train_true,
        train_pred,
        zero_division=0
    )

    train_f1 = f1_score(
        train_true,
        train_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_losses = []

    val_true = []
    val_pred = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            val_losses.append(
                loss.item()
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_true.extend(
                labels.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )

    val_accuracy = accuracy_score(
        val_true,
        val_pred
    )

    val_precision = precision_score(
        val_true,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        val_true,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        val_true,
        val_pred,
        zero_division=0
    )

    scheduler.step(
        val_f1
    )

    print()
    print(f"Epoch {epoch:02d}")

    print(
        f"Train loss={np.mean(train_losses):.4f} | "
        f"Acc={train_accuracy * 100:.2f}% | "
        f"Precision={train_precision * 100:.2f}% | "
        f"Recall={train_recall * 100:.2f}% | "
        f"F1={train_f1 * 100:.2f}%"
    )

    print(
        f"Val   loss={np.mean(val_losses):.4f} | "
        f"Acc={val_accuracy * 100:.2f}% | "
        f"Precision={val_precision * 100:.2f}% | "
        f"Recall={val_recall * 100:.2f}% | "
        f"F1={val_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    improved = (
        val_f1 > best_f1
        or (
            val_f1 == best_f1
            and val_accuracy > best_accuracy
        )
    )

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

                "val_precision":
                    val_precision,

                "val_recall":
                    val_recall,

                "val_f1":
                    val_f1
            },
            MODEL_PATH
        )

        print("  BEST MODEL SAVED")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print()
        print("Early stopping.")

        break


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

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

print()
print("Model saved:")
print(MODEL_PATH)

print()
print("=" * 70)
print("ГОТОВО")
print("=" * 70)