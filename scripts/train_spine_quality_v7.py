from pathlib import Path
import csv
import hashlib
import random

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "validation.csv"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_v7_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8

MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7

SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class SpineDataset(Dataset):

    def __init__(
        self,
        csv_file,
        project_dir,
        image_size=224,
        train=False
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

        # ----------------------------------------------------
        # Только позвоночник с известной разметкой
        # ----------------------------------------------------

        rows = [
            row
            for row in all_rows
            if row["anatomy"] == "spine"
            and row["spine_quality"] in {"0", "1"}
        ]

        # ----------------------------------------------------
        # Удаляем точные дубликаты PixelData
        # ----------------------------------------------------

        unique_rows = []
        hashes = set()

        for row in rows:

            dicom_path = (
                self.project_dir
                / row["dicom_path"]
            )

            dataset = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=False
            )

            image_hash = hashlib.md5(
                dataset.PixelData
            ).hexdigest()

            if image_hash in hashes:
                continue

            hashes.add(image_hash)

            unique_rows.append(row)

        self.rows = unique_rows

        if not self.rows:

            raise ValueError(
                "После фильтрации не осталось изображений."
            )

    def __len__(self):

        return len(self.rows)

    def load_image(self, dicom_path):

        full_path = (
            self.project_dir
            / dicom_path
        )

        dataset = pydicom.dcmread(full_path)

        image = (
            dataset.pixel_array
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Percentile normalization
        # ----------------------------------------------------

        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)

        if p99 > p1:

            image = np.clip(
                image,
                p1,
                p99
            )

            image = (
                image - p1
            ) / (p99 - p1)

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32
            )

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Сохраняем aspect ratio
        # ----------------------------------------------------

        height = image.shape[-2]
        width = image.shape[-1]

        scale = min(
            self.image_size / height,
            self.image_size / width
        )

        new_height = max(
            1,
            int(round(height * scale))
        )

        new_width = max(
            1,
            int(round(width * scale))
        )

        image = F.interpolate(
            image.unsqueeze(0),
            size=(
                new_height,
                new_width
            ),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        canvas = torch.zeros(
            1,
            self.image_size,
            self.image_size,
            dtype=torch.float32
        )

        top = (
            self.image_size
            - new_height
        ) // 2

        left = (
            self.image_size
            - new_width
        ) // 2

        canvas[
            :,
            top:top + new_height,
            left:left + new_width
        ] = image

        image = canvas

        # ----------------------------------------------------
        # V7 augmentation
        #
        # Очень небольшие изменения.
        # Не используем сильные повороты.
        # ----------------------------------------------------

        if self.train:

            # Небольшой horizontal shift
            if random.random() < 0.25:

                shift = random.randint(
                    -4,
                    4
                )

                image = torch.roll(
                    image,
                    shifts=shift,
                    dims=2
                )

            # Небольшой vertical shift
            if random.random() < 0.25:

                shift = random.randint(
                    -3,
                    3
                )

                image = torch.roll(
                    image,
                    shifts=shift,
                    dims=1
                )

            # Небольшое изменение контраста
            if random.random() < 0.30:

                factor = random.uniform(
                    0.95,
                    1.05
                )

                image = (
                    image * factor
                ).clamp(
                    0.0,
                    1.0
                )

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
            "label": label
        }


# ============================================================
# MODEL
# ============================================================

class SpineQualityResNet18V7(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
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

        # grayscale -> RGB
        if x.shape[1] == 1:

            x = x.repeat(
                1,
                3,
                1,
                1
            )

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device
        ).view(
            1,
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device
        ).view(
            1,
            3,
            1,
            1
        )

        x = (
            x - mean
        ) / std

        return self.model(x)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predictions
):

    true_labels = np.array(
        true_labels
    )

    predictions = np.array(
        predictions
    )

    tp = np.sum(
        (true_labels == 1)
        & (predictions == 1)
    )

    tn = np.sum(
        (true_labels == 0)
        & (predictions == 0)
    )

    fp = np.sum(
        (true_labels == 0)
        & (predictions == 1)
    )

    fn = np.sum(
        (true_labels == 1)
        & (predictions == 0)
    )

    total = len(true_labels)

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training
):

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0

    true_labels = []
    predictions = []

    for batch in loader:

        images = batch["image"].to(
            device
        )

        labels = batch["label"].to(
            device
        )

        if training:

            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            if training:

                loss.backward()

                optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predicted = torch.argmax(
            logits,
            dim=1
        )

        true_labels.extend(
            labels.cpu().numpy().tolist()
        )

        predictions.extend(
            predicted.cpu().numpy().tolist()
        )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        true_labels,
        predictions
    )

    return average_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BASELINE №7 — "
        "PRETRAINED RESNET18 + GENTLE FINE-TUNING"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = SpineDataset(
        TRAIN_CSV,
        PROJECT_DIR,
        IMAGE_SIZE,
        train=True
    )

    val_dataset = SpineDataset(
        VAL_CSV,
        PROJECT_DIR,
        IMAGE_SIZE,
        train=False
    )

    print()
    print(
        "TRAIN исходных строк:",
        len(
            [
                row
                for row in train_dataset.rows
            ]
        )
    )

    print(
        "TRAIN уникальных:",
        len(train_dataset)
    )

    print(
        "VALIDATION уникальных:",
        len(val_dataset)
    )

    train_class0 = sum(
        int(row["spine_quality"] == "0")
        for row in train_dataset.rows
    )

    train_class1 = sum(
        int(row["spine_quality"] == "1")
        for row in train_dataset.rows
    )

    val_class0 = sum(
        int(row["spine_quality"] == "0")
        for row in val_dataset.rows
    )

    val_class1 = sum(
        int(row["spine_quality"] == "1")
        for row in val_dataset.rows
    )

    print()
    print(
        "TRAIN class 0:",
        train_class0
    )

    print(
        "TRAIN class 1:",
        train_class1
    )

    print(
        "VALIDATION class 0:",
        val_class0
    )

    print(
        "VALIDATION class 1:",
        val_class1
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

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

    model = SpineQualityResNet18V7()

    model.to(device)

    # --------------------------------------------------------
    # V7: сначала замораживаем backbone
    # --------------------------------------------------------

    for parameter in model.model.parameters():

        parameter.requires_grad = False

    # Разрешаем обучать последний блок
    for parameter in model.model.layer4.parameters():

        parameter.requires_grad = True

    # И классификатор
    for parameter in model.model.fc.parameters():

        parameter.requires_grad = True

    # --------------------------------------------------------
    # LOSS
    #
    # Обычный CrossEntropyLoss.
    # Никакого weighted loss.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    #
    # layer4 немного быстрее,
    # classifier ещё быстрее.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.model.layer4.parameters(),
                "lr": 1e-5
            },
            {
                "params":
                    model.model.fc.parameters(),
                "lr": 1e-4
            }
        ],
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best_f1 = -1.0
    best_accuracy = -1.0
    best_epoch = 0

    no_improvement = 0

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print()
        print(
            f"Epoch {epoch}"
        )

        # ----------------------------------------------------
        # После 5 эпох постепенно размораживаем layer3
        # ----------------------------------------------------

        if epoch == 6:

            print(
                ">>> Размораживаем layer3"
            )

            for parameter in (
                model.model.layer3.parameters()
            ):

                parameter.requires_grad = True

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params":
                            model.model.layer3.parameters(),
                        "lr": 5e-6
                    },
                    {
                        "params":
                            model.model.layer4.parameters(),
                        "lr": 1e-5
                    },
                    {
                        "params":
                            model.model.fc.parameters(),
                        "lr": 5e-5
                    }
                ],
                weight_decay=1e-4
            )

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=2
                )
            )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_metrics['accuracy'] * 100:.2f}%"
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
            f"Val accuracy: "
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
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_metrics["f1"]
        )

        # ----------------------------------------------------
        # BEST MODEL
        #
        # Главный критерий:
        # F1 класса 1.
        #
        # При одинаковом F1:
        # accuracy.
        # ----------------------------------------------------

        improved = False

        if (
            val_metrics["f1"]
            > best_f1
        ):

            improved = True

        elif (
            val_metrics["f1"]
            == best_f1
            and val_metrics["accuracy"]
            > best_accuracy
        ):

            improved = True

        if improved:

            best_f1 = val_metrics["f1"]

            best_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "val_accuracy":
                        best_accuracy,
                    "val_class1_f1":
                        best_f1
                },
                MODEL_PATH
            )

            print(
                ">>> BEST MODEL SAVED"
            )

        else:

            no_improvement += 1

            print(
                f"No improvement: "
                f"{no_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            no_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            break

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V7 TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best epoch: "
        f"{best_epoch}"
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
        "Model saved:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()