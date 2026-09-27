from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing.dataset import SpineQualityDataset
from src.models.resnet18_model import SpineQualityResNet18


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 8

NUM_EPOCHS = 15

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Фиксируем случайность
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Пути
# ============================================================

TRAIN_CSV = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "train.csv"
)

VAL_CSV = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "validation.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "spine_quality_resnet18_best.pth"
)


# ============================================================
# Информация
# ============================================================

print("=" * 60)
print("ОБУЧЕНИЕ SPINE QUALITY")
print("=" * 60)

print()
print(f"Устройство: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print("Используется CPU.")

print()


# ============================================================
# Dataset
# ============================================================

train_dataset = SpineQualityDataset(
    csv_file=TRAIN_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
)

val_dataset = SpineQualityDataset(
    csv_file=VAL_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
)


print(f"Train изображений: {len(train_dataset)}")
print(f"Validation изображений: {len(val_dataset)}")


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Считаем количество объектов каждого класса
# ============================================================

train_labels = []

for row in train_dataset.rows:
    train_labels.append(
        int(row["spine_quality"])
    )

class_0_count = train_labels.count(0)
class_1_count = train_labels.count(1)

print()
print("Распределение TRAIN:")

print(f"Класс 0: {class_0_count}")
print(f"Класс 1: {class_1_count}")


# ============================================================
# Веса классов
# ============================================================

total = class_0_count + class_1_count

weight_0 = total / (2 * class_0_count)
weight_1 = total / (2 * class_1_count)

class_weights = torch.tensor(
    [weight_0, weight_1],
    dtype=torch.float32,
    device=DEVICE,
)

print()
print("Веса классов:")
print(f"class 0: {weight_0:.4f}")
print(f"class 1: {weight_1:.4f}")


# ============================================================
# Модель
# ============================================================

print()
print("Создание ResNet18...")

model = SpineQualityResNet18(
    pretrained=True
)

model = model.to(DEVICE)

print("Модель создана.")


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Переменные для сохранения лучшей модели
# ============================================================

best_val_accuracy = 0.0


# ============================================================
# TRAIN
# ============================================================

for epoch in range(NUM_EPOCHS):

    print()
    print("-" * 60)
    print(
        f"ЭПОХА {epoch + 1}/{NUM_EPOCHS}"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = (
        train_loss_sum / train_total
    )

    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION MODE
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_sum += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = (
        val_loss_sum / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # Вывод
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Сохраняем лучшую модель
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
            },
            BEST_MODEL_PATH
        )

        print()
        print("Новая лучшая модель!")

        print(
            f"Сохранена: {BEST_MODEL_PATH}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )


# ============================================================
# Конец
# ============================================================

print()
print("=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("=" * 60)

print()
print(
    f"Лучшая Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print()
print(
    f"Лучшая модель сохранена:"
)

print(BEST_MODEL_PATH)