from pathlib import Path
import csv
import hashlib
import random

import numpy as np
import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "validation.csv"

MODEL_FILE = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_v6_best.pth"
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

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PIXEL HASH
# ============================================================

def get_pixel_hash(path):

    dataset = pydicom.dcmread(path)

    pixel_array = dataset.pixel_array

    return hashlib.sha256(
        pixel_array.tobytes()
    ).hexdigest()


# ============================================================
# ЧТЕНИЕ CSV
# ============================================================

def read_rows(csv_file):

    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = []

        for row in reader:

            if row["anatomy"] != "spine":
                continue

            if row["spine_quality"] not in {"0", "1"}:
                continue

            rows.append(row)

    return rows


# ============================================================
# УДАЛЕНИЕ PIXELDATA-ДУБЛИКАТОВ
# ============================================================

def make_unique_rows(rows):

    unique_rows = []

    seen = set()

    for row in rows:

        path = PROJECT_DIR / row["dicom_path"]

        h = get_pixel_hash(path)

        if h in seen:
            continue

        seen.add(h)

        new_row = dict(row)

        new_row["_pixel_hash"] = h

        unique_rows.append(new_row)

    return unique_rows


# ============================================================
# DATASET
# ============================================================

class SpineDataset(Dataset):

    def __init__(
        self,
        rows,
        project_dir,
        train=False
    ):
        self.project_dir = Path(project_dir)
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def load_image(self, path):

        dataset = pydicom.dcmread(path)

        image = dataset.pixel_array.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Percentile normalization
        # ----------------------------------------------------

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
                image,
                dtype=np.float32
            )

        image = np.clip(
            image,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Сохраняем пропорции
        # ----------------------------------------------------

        h, w = image.shape[1:]

        scale = min(
            IMAGE_SIZE / h,
            IMAGE_SIZE / w
        )

        new_h = max(
            1,
            int(round(h * scale))
        )

        new_w = max(
            1,
            int(round(w * scale))
        )

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        canvas = torch.zeros(
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            dtype=torch.float32
        )

        top = (
            IMAGE_SIZE - new_h
        ) // 2

        left = (
            IMAGE_SIZE - new_w
        ) // 2

        canvas[
            :,
            top:top + new_h,
            left:left + new_w
        ] = image

        image = canvas

        # ----------------------------------------------------
        # TRAIN augmentation
        # ----------------------------------------------------

        if self.train:

            # Небольшой сдвиг
            if random.random() < 0.5:

                shift_x = random.randint(
                    -4,
                    4
                )

                shift_y = random.randint(
                    -4,
                    4
                )

                shifted = torch.zeros_like(
                    image
                )

                src_x1 = max(
                    0,
                    -shift_x
                )

                src_x2 = min(
                    IMAGE_SIZE,
                    IMAGE_SIZE - shift_x
                )

                src_y1 = max(
                    0,
                    -shift_y
                )

                src_y2 = min(
                    IMAGE_SIZE,
                    IMAGE_SIZE - shift_y
                )

                dst_x1 = max(
                    0,
                    shift_x
                )

                dst_x2 = dst_x1 + (
                    src_x2 - src_x1
                )

                dst_y1 = max(
                    0,
                    shift_y
                )

                dst_y2 = dst_y1 + (
                    src_y2 - src_y1
                )

                shifted[
                    :,
                    dst_y1:dst_y2,
                    dst_x1:dst_x2
                ] = image[
                    :,
                    src_y1:src_y2,
                    src_x1:src_x2
                ]

                image = shifted

            # Небольшое изменение контраста
            if random.random() < 0.5:

                contrast = random.uniform(
                    0.90,
                    1.10
                )

                image = (
                    image - 0.5
                ) * contrast + 0.5

                image = torch.clamp(
                    image,
                    0.0,
                    1.0
                )

        return image

    def __getitem__(self, index):

        row = self.rows[index]

        path = (
            self.project_dir
            / row["dicom_path"]
        )

        image = self.load_image(path)

        label = torch.tensor(
            int(row["spine_quality"]),
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label
        }


# ============================================================
# МОДЕЛЬ V6
# ============================================================

class SpineQualityResNet18V6(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        num_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Sequential(

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                num_features,
                2
            )
        )

    def forward(self, x):

        # 1 канал → 3 канала
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
# МЕТРИКИ
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

    accuracy = (
        predictions == labels
    ).mean() * 100

    tp = np.sum(
        (labels == 1)
        & (predictions == 1)
    )

    fp = np.sum(
        (labels == 0)
        & (predictions == 1)
    )

    fn = np.sum(
        (labels == 1)
        & (predictions == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return (
        accuracy,
        precision * 100,
        recall * 100,
        f1 * 100
    )


# ============================================================
# ОДНА ЭПОХА
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    predictions = []

    labels = []

    for batch in loader:

        images = batch["image"].to(
            DEVICE
        )

        target = batch["label"].to(
            DEVICE
        )

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(
            output,
            target
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        labels.extend(
            target.cpu().numpy()
        )

    loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            predictions,
            labels
        )
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    predictions = []

    labels = []

    for batch in loader:

        images = batch["image"].to(
            DEVICE
        )

        target = batch["label"].to(
            DEVICE
        )

        output = model(images)

        loss = criterion(
            output,
            target
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        pred = torch.argmax(
            output,
            dim=1
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        labels.extend(
            target.cpu().numpy()
        )

    loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy, precision, recall, f1 = (
        calculate_metrics(
            predictions,
            labels
        )
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE №6 — UNIQUE IMAGES + WEIGHTED LOSS")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # READ DATA
    # --------------------------------------------------------

    train_rows = read_rows(
        TRAIN_CSV
    )

    val_rows = read_rows(
        VAL_CSV
    )

    print()
    print(
        "TRAIN исходных строк:",
        len(train_rows)
    )

    print(
        "VALIDATION исходных строк:",
        len(val_rows)
    )

    # --------------------------------------------------------
    # UNIQUE
    # --------------------------------------------------------

    train_rows = make_unique_rows(
        train_rows
    )

    val_rows = make_unique_rows(
        val_rows
    )

    print()
    print(
        "TRAIN уникальных:",
        len(train_rows)
    )

    print(
        "VALIDATION уникальных:",
        len(val_rows)
    )

    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    train_class0 = sum(
        row["spine_quality"] == "0"
        for row in train_rows
    )

    train_class1 = sum(
        row["spine_quality"] == "1"
        for row in train_rows
    )

    val_class0 = sum(
        row["spine_quality"] == "0"
        for row in val_rows
    )

    val_class1 = sum(
        row["spine_quality"] == "1"
        for row in val_rows
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

    print()
    print(
        "VALIDATION class 0:",
        val_class0
    )

    print(
        "VALIDATION class 1:",
        val_class1
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = SpineDataset(
        train_rows,
        PROJECT_DIR,
        train=True
    )

    val_dataset = SpineDataset(
        val_rows,
        PROJECT_DIR,
        train=False
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
    print()
    print("Проверка DataLoader...")

    test_batch = next(iter(train_loader))

    print(
        "image type:",
        type(test_batch["image"])
    )

    print(
        "image shape:",
        test_batch["image"].shape
    )

    print(
        "label type:",
        type(test_batch["label"])
    )

    print(
        "label shape:",
        test_batch["label"].shape
    )

    print(
        "label values:",
        test_batch["label"].tolist()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SpineQualityResNet18V6()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # WEIGHTED LOSS
    # --------------------------------------------------------

    total = (
        train_class0
        + train_class1
    )

    weight0 = (
        total
        / (2 * train_class0)
    )

    weight1 = (
        total
        / (2 * train_class1)
    )

    class_weights = torch.tensor(
        [weight0, weight1],
        dtype=torch.float32,
        device=DEVICE
    )

    print()
    print(
        "Class weights:",
        class_weights.cpu().numpy()
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.model.parameters(),
                "lr": 2e-5
            }
        ],
        weight_decay=2e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_f1 = -1.0
    best_accuracy = -1.0

    best_epoch = 0

    patience_counter = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print()
        print(
            f"Epoch {epoch}"
        )

        (
            train_loss,
            train_acc,
            train_precision,
            train_recall,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1
        ) = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Train accuracy: {train_acc:.2f}%"
        )

        print(
            f"Train class1 recall: "
            f"{train_recall:.2f}%"
        )

        print(
            f"Train class1 F1: "
            f"{train_f1:.2f}%"
        )

        print(
            f"Val loss: {val_loss:.4f}"
        )

        print(
            f"Val accuracy: {val_acc:.2f}%"
        )

        print(
            f"Val class1 precision: "
            f"{val_precision:.2f}%"
        )

        print(
            f"Val class1 recall: "
            f"{val_recall:.2f}%"
        )

        print(
            f"Val class1 F1: "
            f"{val_f1:.2f}%"
        )

        # ----------------------------------------------------
        # LR scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_f1
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        improved = (
            val_f1 > best_f1
            or (
                val_f1 == best_f1
                and val_acc > best_accuracy
            )
        )

        if improved:

            best_f1 = val_f1

            best_accuracy = val_acc

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_class1_f1": val_f1,
                    "train_unique": len(train_rows),
                    "validation_unique": len(val_rows)
                },
                MODEL_FILE
            )

            print(
                ">>> BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

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
    print("V6 TRAINING FINISHED")
    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Best validation class1 F1: "
        f"{best_f1:.2f}%"
    )

    print()
    print(
        "Model saved:"
    )

    print(
        MODEL_FILE
    )


if __name__ == "__main__":
    main()