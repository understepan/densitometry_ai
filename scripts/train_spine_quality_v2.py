from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from src.preprocessing.dataset import SpineQualityDataset
from src.models.resnet18_model import SpineQualityResNet18


# ============================================================
# 1. ПУТИ И НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "validation.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "spine_quality_resnet18_v2_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_EPOCHS = 30
PATIENCE = 7

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. ФИКСИРУЕМ СЛУЧАЙНОСТЬ
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. АУГМЕНТАЦИЯ
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=4,
        translate=(0.02, 0.02),
        scale=(0.97, 1.03),
        fill=0,
    ),
])


# ============================================================
# 4. DATASET И DATALOADER
# ============================================================

train_dataset = SpineQualityDataset(
    csv_file=TRAIN_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
    transform=train_transform,
)

val_dataset = SpineQualityDataset(
    csv_file=VAL_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
    transform=None,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 5. ПРОВЕРЯЕМ РАЗМЕРЫ ВЫБОРОК И БАЛАНС КЛАССОВ
# ============================================================

def count_classes(dataset):
    counts = {0: 0, 1: 0}

    for row in dataset.rows:
        label = int(row["spine_quality"])
        counts[label] += 1

    return counts


train_counts = count_classes(train_dataset)
val_counts = count_classes(val_dataset)

print("=" * 65)
print("BASELINE №2 — PRETRAINED RESNET18")
print("=" * 65)

print(f"Устройство: {DEVICE}")
print(f"Train: {len(train_dataset)} изображений")
print(f"Validation: {len(val_dataset)} изображений")

print(f"Train классы: {train_counts}")
print(f"Validation классы: {val_counts}")

print(f"Batch size: {BATCH_SIZE}")
print(f"Максимум эпох: {MAX_EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")


# ============================================================
# 6. СОЗДАЁМ МОДЕЛЬ С PRETRAINED ВЕСАМИ
# ============================================================

model = SpineQualityResNet18(pretrained=True)
model = model.to(DEVICE)


# ============================================================
# 7. ВЕСА КЛАССОВ
# ============================================================

total_train = train_counts[0] + train_counts[1]

class_weights = torch.tensor(
    [
        total_train / (2 * train_counts[0]),
        total_train / (2 * train_counts[1]),
    ],
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print(f"Веса классов: {class_weights.cpu().tolist()}")


# ============================================================
# 8. OPTIMIZER
# ============================================================

# Маленький learning rate для pretrained backbone.
# Более высокий learning rate для нового классификатора.

optimizer = torch.optim.AdamW(
    [
        {
            "params": [
                p for name, p in model.named_parameters()
                if not name.startswith("model.fc.")
            ],
            "lr": 1e-5,
        },
        {
            "params": model.model.fc.parameters(),
            "lr": 1e-3,
        },
    ],
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# 9. ОДНА ЭПОХА
# ============================================================

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_true = []
    all_pred = []

    for batch in loader:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "true": all_true,
        "pred": all_pred,
    }


# ============================================================
# 10. ОБУЧЕНИЕ С EARLY STOPPING
# ============================================================

best_val_accuracy = -1.0
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):

    train_metrics = run_epoch(
        model,
        train_loader,
        optimizer=optimizer,
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        optimizer=None,
    )

    scheduler.step(val_metrics["loss"])

    current_lr_backbone = optimizer.param_groups[0]["lr"]
    current_lr_head = optimizer.param_groups[1]["lr"]

    print("-" * 65)
    print(f"Эпоха {epoch}/{MAX_EPOCHS}")

    print(
        f"TRAIN: "
        f"loss={train_metrics['loss']:.4f}, "
        f"accuracy={train_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"VALIDATION: "
        f"loss={val_metrics['loss']:.4f}, "
        f"accuracy={val_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"LR backbone={current_lr_backbone:.7f}, "
        f"LR head={current_lr_head:.7f}"
    )

    # Сначала сравниваем accuracy.
    # При равной accuracy предпочитаем меньший val loss.
    improved = (
        val_metrics["accuracy"] > best_val_accuracy
        or (
            val_metrics["accuracy"] == best_val_accuracy
            and val_metrics["loss"] < best_val_loss
        )
    )

    if improved:
        best_val_accuracy = val_metrics["accuracy"]
        best_val_loss = val_metrics["loss"]

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": best_val_accuracy,
                "val_loss": best_val_loss,
                "class_weights": class_weights.cpu(),
                "seed": SEED,
            },
            MODEL_PATH,
        )

        print("Сохранена новая лучшая модель.")

    else:
        epochs_without_improvement += 1

        print(
            "Нет улучшения: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: обучение остановлено.")
        break


# ============================================================
# 11. ИТОГ
# ============================================================

print("=" * 65)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("=" * 65)

print(f"Лучшая Validation Accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Лучшая Validation Loss: {best_val_loss:.4f}")
print(f"Модель сохранена: {MODEL_PATH}")