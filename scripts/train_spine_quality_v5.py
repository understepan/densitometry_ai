from pathlib import Path
import csv
import hashlib
import random

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "validation.csv"

MODEL_OUTPUT = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_v5_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 30
PATIENCE = 7
SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DATASET
# ============================================================

class SpineQualityDataset(Dataset):

    def __init__(
        self,
        csv_file,
        project_dir,
        image_size=224,
        train=False,
    ):
        self.csv_file = Path(csv_file)
        self.project_dir = Path(project_dir)
        self.image_size = image_size
        self.train = train

        with open(
            self.csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)
            all_rows = list(reader)

        # Только позвоночник с известной разметкой
        rows = []

        for row in all_rows:

            if row["anatomy"] != "spine":
                continue

            if row["spine_quality"] not in {"0", "1"}:
                continue

            rows.append(row)

        # ----------------------------------------------------
        # Удаляем exact PixelData duplicates
        # ----------------------------------------------------

        unique_rows = []
        hashes = set()

        for row in rows:

            full_path = (
                self.project_dir
                / row["dicom_path"]
            )

            ds = pydicom.dcmread(full_path)

            pixel_hash = hashlib.sha256(
                ds.pixel_array.tobytes()
            ).hexdigest()

            if pixel_hash in hashes:
                continue

            hashes.add(pixel_hash)
            unique_rows.append(row)

        self.rows = unique_rows

        print(
            f"{self.csv_file.name}: "
            f"исходных строк = {len(rows)}, "
            f"уникальных = {len(self.rows)}"
        )

        if not self.rows:
            raise ValueError(
                "После фильтрации не осталось изображений."
            )

    def __len__(self):
        return len(self.rows)

    # --------------------------------------------------------
    # V3 preprocessing
    # --------------------------------------------------------

    def load_image(self, dicom_path):

        full_path = (
            self.project_dir
            / dicom_path
        )

        ds = pydicom.dcmread(full_path)

        image = ds.pixel_array.astype(
            np.float32
        )

        # Обычная V3 min-max нормализация
        min_value = image.min()
        max_value = image.max()

        if max_value > min_value:

            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32
            )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # Именно V3:
        # простое изменение размера до 224x224
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False,
        )

        image = image.squeeze(0)

        return image

    # --------------------------------------------------------
    # V3 augmentation
    # --------------------------------------------------------

    def augment(self, image):

        affine = torch.nn.functional.affine_grid(
            torch.eye(
                2,
                3,
                dtype=torch.float32
            ).unsqueeze(0),
            size=(
                1,
                1,
                self.image_size,
                self.image_size
            ),
            align_corners=False
        )

        # Здесь используем torchvision transform ниже,
        # поэтому этот метод не применяется.
        return image

    def __getitem__(self, index):

        row = self.rows[index]

        image = self.load_image(
            row["dicom_path"]
        )

        label = torch.tensor(
            int(row["spine_quality"]),
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label,
            "study_id": row["study_id"],
            "dicom_path": row["dicom_path"],
        }


# ============================================================
# V3 AUGMENTATION
# ============================================================

from torchvision import transforms


train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=4,
        translate=(0.02, 0.02),
        scale=(0.97, 1.03),
        fill=0,
    ),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
    ),
])


# ============================================================
# MODEL
# ============================================================

class SpineQualityResNet18V5(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(
            weights=weights
        )

        num_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(
                num_features,
                2
            )
        )

    def forward(self, x):

        if x.shape[1] == 1:

            x = x.repeat(
                1,
                3,
                1,
                1
            )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device
        ).view(1, 3, 1, 1)

        x = (
            x - mean
        ) / std

        return self.model(x)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    labels
):

    predictions = np.array(
        predictions
    )

    labels = np.array(
        labels
    )

    accuracy = np.mean(
        predictions == labels
    )

    tp = np.sum(
        (predictions == 1)
        & (labels == 1)
    )

    fp = np.sum(
        (predictions == 1)
        & (labels == 0)
    )

    fn = np.sum(
        (predictions == 0)
        & (labels == 1)
    )

    if tp + fp > 0:
        precision = (
            tp / (tp + fp)
        )
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = (
            tp / (tp + fn)
        )
    else:
        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

def train():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print(
        "BASELINE №5 — V3 PREPROCESSING + UNIQUE IMAGES"
    )
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = SpineQualityDataset(
        TRAIN_CSV,
        PROJECT_DIR,
        IMAGE_SIZE,
        train=True,
    )

    val_dataset = SpineQualityDataset(
        VAL_CSV,
        PROJECT_DIR,
        IMAGE_SIZE,
        train=False,
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    train_labels = [
        int(row["spine_quality"])
        for row in train_dataset.rows
    ]

    class0 = train_labels.count(0)
    class1 = train_labels.count(1)

    print()
    print(
        f"TRAIN class 0: {class0}"
    )

    print(
        f"TRAIN class 1: {class1}"
    )

    # --------------------------------------------------------
    # WEIGHTED SAMPLER
    # --------------------------------------------------------

    class_weights = {
        0: 1.0 / class0,
        1: 1.0 / class1,
    }

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SpineQualityResNet18V5()

    model = model.to(device)

    # --------------------------------------------------------
    # OPTIMIZER — exactly as V3
    # --------------------------------------------------------

    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():

        if "model.fc" in name:

            head_parameters.append(
                parameter
            )

        else:

            backbone_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_parameters,
                "lr": 3e-5,
            },
            {
                "params":
                    head_parameters,
                "lr": 3e-4,
            },
        ],
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_val_f1 = -1.0

    best_epoch = 0
    patience_counter = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_losses = []
        train_predictions = []
        train_labels_epoch = []

        for batch in train_loader:

            images = batch["image"].to(
                device
            )

            labels = batch["label"].to(
                device
            )

            # V3 augmentation применяется
            # после загрузки изображения

            augmented_images = []

            for image in images:

                image = train_transform(
                    image
                )

                augmented_images.append(
                    image
                )

            images = torch.stack(
                augmented_images
            ).to(device)

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
                .detach()
                .cpu()
                .numpy()
            )

            train_predictions.extend(
                predictions.tolist()
            )

            train_labels_epoch.extend(
                labels.detach()
                .cpu()
                .numpy()
                .tolist()
            )

        (
            train_accuracy,
            _,
            train_recall,
            train_f1
        ) = calculate_metrics(
            train_predictions,
            train_labels_epoch
        )

        train_loss = np.mean(
            train_losses
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_losses = []
        val_predictions = []
        val_labels_epoch = []

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(
                    device
                )

                labels = batch["label"].to(
                    device
                )

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

                val_losses.append(
                    loss.item()
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                val_predictions.extend(
                    predictions.tolist()
                )

                val_labels_epoch.extend(
                    labels.cpu()
                    .numpy()
                    .tolist()
                )

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = calculate_metrics(
            val_predictions,
            val_labels_epoch
        )

        val_loss = np.mean(
            val_losses
        )

        scheduler.step(
            val_accuracy
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}%"
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
            f"Val loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val accuracy: "
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
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_f1 > best_val_f1
            )
        ):

            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            best_epoch = epoch

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
                MODEL_OUTPUT,
            )

            patience_counter = 0

            print(
                ">>> BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                "EARLY STOPPING"
            )

            break

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 60)
    print("V5 TRAINING FINISHED")
    print("=" * 60)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation class1 F1: "
        f"{best_val_f1 * 100:.2f}%"
    )

    print()
    print(
        "Model saved:"
    )

    print(
        MODEL_OUTPUT
    )


if __name__ == "__main__":
    train()