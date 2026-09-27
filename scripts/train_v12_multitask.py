from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

VALIDATION_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR
    / "spine_quality_resnet18_v12_multitask_best.pth"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 8

NUM_WORKERS = 0

EPOCHS = 25

PATIENCE = 7

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 2e-4

DROPOUT = 0.30

IMAGE_SIZE = 224


# ============================================================
# RANDOM SEED
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("V12 MULTITASK TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(path_value):

    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# PIXEL HASH
# ============================================================

def get_pixel_hash(path_value):

    path = resolve_path(path_value)

    ds = pydicom.dcmread(path)

    arr = ds.pixel_array

    return hash(arr.tobytes())


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def percentile_normalize(arr):

    arr = arr.astype(np.float32)

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:
        return np.zeros_like(
            arr,
            dtype=np.uint8
        )

    arr = np.clip(
        arr,
        p1,
        p99
    )

    arr = (
        (arr - p1)
        / (p99 - p1)
        * 255.0
    )

    return arr.astype(np.uint8)


def resize_with_padding(image):

    image = Image.fromarray(image)

    width, height = image.size

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height
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
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        0
    )

    left = (
        IMAGE_SIZE - new_width
    ) // 2

    top = (
        IMAGE_SIZE - new_height
    ) // 2

    canvas.paste(
        image,
        (left, top)
    )

    return canvas


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=3,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class SpineMultitaskDataset(Dataset):

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

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = resolve_path(
            row["dicom_path"]
        )

        ds = pydicom.dcmread(path)

        arr = ds.pixel_array

        arr = percentile_normalize(
            arr
        )

        image = resize_with_padding(
            arr
        )

        # grayscale -> RGB
        image = np.stack(
            [np.array(image)] * 3,
            axis=-1
        )

        image = Image.fromarray(
            image
        )

        if self.train:
            image = train_transform(
                image
            )
        else:
            image = validation_transform(
                image
            )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        final_label = float(
            row["spine_quality"]
        )

        position = row["spine_position"]

        axis = row["spine_axis"]

        artifacts = row["spine_artifacts"]

        # NaN -> -1
        # -1 означает "этой auxiliary-разметки нет"
        if pd.isna(position):
            position = -1
        else:
            position = int(position)

        if pd.isna(axis):
            axis = -1
        else:
            axis = int(axis)

        if pd.isna(artifacts):
            artifacts = -1
        else:
            artifacts = int(artifacts)

        return {
            "image": image,

            "final": torch.tensor(
                int(final_label),
                dtype=torch.long
            ),

            "position": torch.tensor(
                position,
                dtype=torch.long
            ),

            "axis": torch.tensor(
                axis,
                dtype=torch.long
            ),

            "artifacts": torch.tensor(
                artifacts,
                dtype=torch.long
            ),

            "study_id": str(
                row["study_id"]
            )
        }


# ============================================================
# EXACT DEDUP
# ============================================================

def exact_dedup(df):

    print("Удаление точных PixelData-дубликатов...")

    hashes = []

    for _, row in df.iterrows():

        hashes.append(
            get_pixel_hash(
                row["dicom_path"]
            )
        )

    df = df.copy()

    df["_pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=["_pixel_hash"]
    ).copy()

    df = df.drop(
        columns=["_pixel_hash"]
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

train_split = pd.read_csv(
    TRAIN_SPLIT_PATH
)

validation_split = pd.read_csv(
    VALIDATION_SPLIT_PATH
)


print()
print(
    f"Исходных строк manifest: "
    f"{len(manifest)}"
)


# ============================================================
# SPINE ONLY
# ============================================================

spine = manifest[
    manifest["anatomy"].astype(str).str.lower()
    == "spine"
].copy()


print(
    f"Spine строк: {len(spine)}"
)


# ============================================================
# STUDY SPLITS
# ============================================================

train_studies = set(
    train_split["study_id"]
    .astype(str)
)

validation_studies = set(
    validation_split["study_id"]
    .astype(str)
)


train_df = spine[
    spine["study_id"]
    .astype(str)
    .isin(train_studies)
].copy()


validation_df = spine[
    spine["study_id"]
    .astype(str)
    .isin(validation_studies)
].copy()


# ============================================================
# FINAL LABEL
# ============================================================

train_df = train_df[
    train_df["spine_quality"].notna()
].copy()

validation_df = validation_df[
    validation_df["spine_quality"].notna()
].copy()


print()
print(
    f"TRAIN before dedup: "
    f"{len(train_df)}"
)

print(
    f"VALIDATION before dedup: "
    f"{len(validation_df)}"
)


# ============================================================
# EXACT DEDUP
# ============================================================

train_df = exact_dedup(
    train_df
)

validation_df = exact_dedup(
    validation_df
)


print()
print("После exact dedup:")

print(
    f"TRAIN: {len(train_df)}"
)

print(
    f"VALIDATION: {len(validation_df)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("FINAL LABELS:")

print(
    "TRAIN:"
)

print(
    train_df["spine_quality"]
    .value_counts()
    .sort_index()
)

print(
    "VALIDATION:"
)

print(
    validation_df["spine_quality"]
    .value_counts()
    .sort_index()
)


# ============================================================
# AUXILIARY LABEL DISTRIBUTIONS
# ============================================================

print()
print("=" * 70)
print("AUXILIARY LABELS")
print("=" * 70)


for name in [
    "spine_position",
    "spine_axis",
    "spine_artifacts"
]:

    print()
    print(name)

    print(
        train_df[name]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


# ============================================================
# DATASETS
# ============================================================

train_dataset = SpineMultitaskDataset(
    train_df,
    train=True
)

validation_dataset = SpineMultitaskDataset(
    validation_df,
    train=False
)


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

train_labels = (
    train_df["spine_quality"]
    .astype(int)
    .values
)

class_counts = np.bincount(
    train_labels,
    minlength=2
)

class_weights = (
    len(train_labels)
    / (
        2.0
        * class_counts
    )
)

sample_weights = np.array([
    class_weights[label]
    for label in train_labels
])


sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(
        sample_weights
    ),
    num_samples=len(
        sample_weights
    ),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

class V12MultitaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.backbone.fc = nn.Identity()

        self.shared_dropout = nn.Dropout(
            DROPOUT
        )

        # Main task
        self.final_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 2)
        )

        # Auxiliary tasks
        self.position_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.axis_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.artifacts_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        features = self.backbone(x)

        features = self.shared_dropout(
            features
        )

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

        return {
            "final": final_logits,
            "position": position_logits,
            "axis": axis_logits,
            "artifacts": artifacts_logits
        }


# ============================================================
# MODEL
# ============================================================

model = V12MultitaskModel()

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

final_class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

final_criterion = nn.CrossEntropyLoss(
    weight=final_class_weights
)

aux_criterion = nn.CrossEntropyLoss(
    ignore_index=-1
)


# ============================================================
# LOSS WEIGHTS
# ============================================================

FINAL_WEIGHT = 1.0

POSITION_WEIGHT = 0.25

AXIS_WEIGHT = 0.25

ARTIFACTS_WEIGHT = 0.25


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    y_true = []
    y_pred = []

    for batch in train_loader:

        images = batch["image"].to(
            DEVICE
        )

        final_targets = batch[
            "final"
        ].to(DEVICE)

        position_targets = batch[
            "position"
        ].to(DEVICE)

        axis_targets = batch[
            "axis"
        ].to(DEVICE)

        artifacts_targets = batch[
            "artifacts"
        ].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        final_loss = final_criterion(
            outputs["final"],
            final_targets
        )

        position_loss = aux_criterion(
            outputs["position"],
            position_targets
        )

        axis_loss = aux_criterion(
            outputs["axis"],
            axis_targets
        )

        artifacts_loss = aux_criterion(
            outputs["artifacts"],
            artifacts_targets
        )

        loss = (
            FINAL_WEIGHT * final_loss
            + POSITION_WEIGHT * position_loss
            + AXIS_WEIGHT * axis_loss
            + ARTIFACTS_WEIGHT * artifacts_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs["final"],
            dim=1
        )

        y_true.extend(
            final_targets.cpu().numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    avg_loss = (
        total_loss
        / len(train_dataset)
    )

    return avg_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in validation_loader:

            images = batch["image"].to(
                DEVICE
            )

            final_targets = batch[
                "final"
            ].to(DEVICE)

            position_targets = batch[
                "position"
            ].to(DEVICE)

            axis_targets = batch[
                "axis"
            ].to(DEVICE)

            artifacts_targets = batch[
                "artifacts"
            ].to(DEVICE)

            outputs = model(images)

            final_loss = final_criterion(
                outputs["final"],
                final_targets
            )

            position_loss = aux_criterion(
                outputs["position"],
                position_targets
            )

            axis_loss = aux_criterion(
                outputs["axis"],
                axis_targets
            )

            artifacts_loss = aux_criterion(
                outputs["artifacts"],
                artifacts_targets
            )

            loss = (
                FINAL_WEIGHT * final_loss
                + POSITION_WEIGHT * position_loss
                + AXIS_WEIGHT * axis_loss
                + ARTIFACTS_WEIGHT * artifacts_loss
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs["final"],
                dim=1
            )

            y_true.extend(
                final_targets.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    avg_loss = (
        total_loss
        / len(validation_dataset)
    )

    return avg_loss, metrics


# ============================================================
# TRAINING LOOP
# ============================================================

best_f1 = -1.0

best_accuracy = -1.0

best_epoch = 0

epochs_without_improvement = 0


print()
print("=" * 70)
print("START TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_metrics = (
        train_one_epoch()
    )

    val_loss, val_metrics = (
        validate()
    )

    (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1
    ) = train_metrics

    (
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = val_metrics

    scheduler.step(
        val_f1
    )

    print()
    print(
        f"Epoch {epoch:02d}"
    )

    print(
        f"Train loss={train_loss:.4f} | "
        f"Acc={train_accuracy * 100:.2f}% | "
        f"Precision={train_precision * 100:.2f}% | "
        f"Recall={train_recall * 100:.2f}% | "
        f"F1={train_f1 * 100:.2f}%"
    )

    print(
        f"Val   loss={val_loss:.4f} | "
        f"Acc={val_accuracy * 100:.2f}% | "
        f"Precision={val_precision * 100:.2f}% | "
        f"Recall={val_recall * 100:.2f}% | "
        f"F1={val_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # BEST MODEL
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

        checkpoint = {
            "model_state_dict": model.state_dict(),

            "epoch": epoch,

            "val_accuracy": val_accuracy,

            "val_precision": val_precision,

            "val_recall": val_recall,

            "val_f1": val_f1
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print(
            "  BEST MODEL SAVED"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping."
        )

        break


# ============================================================
# FINAL
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
print(
    f"Model saved:"
)

print(
    MODEL_PATH
)

print()
print("=" * 70)
print("ГОТОВО")
print("=" * 70)