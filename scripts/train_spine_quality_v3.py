from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.preprocessing.dataset import SpineQualityDataset
from src.models.resnet18_model import SpineQualityResNet18


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "validation.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "spine_quality_resnet18_v3_best.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 8

MAX_EPOCHS = 30

EARLY_STOPPING_PATIENCE = 7

LEARNING_RATE_BACKBONE = 3e-5
LEARNING_RATE_HEAD = 3e-4

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30


# ============================================================
# ФИКСИРУЕМ RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Для воспроизводимости
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# МОДЕЛЬ V3
# ============================================================

class SpineQualityResNet18V3(nn.Module):

    def __init__(self, pretrained=True, dropout=0.30):
        super().__init__()

        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None

        # Используем существующую модель V1/V2
        self.base = SpineQualityResNet18(pretrained=pretrained)

        # Получаем количество признаков перед старым классификатором
        num_features = self.base.model.fc.in_features

        # Новый классификатор:
        # признаки → Dropout → 2 класса
        self.base.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.base(x)


# ============================================================
# СОЗДАНИЕ TRAIN DATASET
# ============================================================

def create_train_dataset():

    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=4,
            translate=(0.02, 0.02),
            scale=(0.97, 1.03),
            fill=0
        ),

        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08
        )
    ])

    dataset = SpineQualityDataset(
        csv_file=TRAIN_CSV,
        project_dir=PROJECT_DIR,
        image_size=IMAGE_SIZE,
        transform=train_transform
    )

    return dataset


# ============================================================
# VALIDATION DATASET
# ============================================================

def create_val_dataset():

    dataset = SpineQualityDataset(
        csv_file=VAL_CSV,
        project_dir=PROJECT_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    return dataset


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

def create_weighted_sampler(dataset):

    labels = []

    for row in dataset.rows:
        labels.append(int(row["spine_quality"]))

    labels = np.array(labels)

    class_counts = np.bincount(labels, minlength=2)

    print()
    print("Распределение классов TRAIN:")
    print(f"Класс 0: {class_counts[0]}")
    print(f"Класс 1: {class_counts[1]}")

    # Вес класса = 1 / количество объектов класса
    class_weights = np.zeros(2, dtype=np.float32)

    for class_id in range(2):
        if class_counts[class_id] > 0:
            class_weights[class_id] = 1.0 / class_counts[class_id]

    sample_weights = np.array(
        [class_weights[label] for label in labels],
        dtype=np.float64
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    print()
    print("Веса sampler:")
    print(f"Класс 0: {class_weights[0]:.6f}")
    print(f"Класс 1: {class_weights[1]:.6f}")

    return sampler


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(all_labels, all_predictions):

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = float(
        np.mean(all_labels == all_predictions)
    )

    tp = int(np.sum(
        (all_labels == 1) &
        (all_predictions == 1)
    ))

    tn = int(np.sum(
        (all_labels == 0) &
        (all_predictions == 0)
    ))

    fp = int(np.sum(
        (all_labels == 0) &
        (all_predictions == 1)
    ))

    fn = int(np.sum(
        (all_labels == 1) &
        (all_predictions == 0)
    ))

    precision_0 = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    recall_0 = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1_0 = (
        2 * precision_0 * recall_0 /
        (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0
        else 0.0
    )

    precision_1 = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall_1 = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1_1 = (
        2 * precision_1 * recall_1 /
        (precision_1 + recall_1)
        if (precision_1 + recall_1) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,

        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    for batch in loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )

    epoch_loss = running_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    return epoch_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("BASELINE №3 — RESNET18 + SAMPLER + DROPOUT")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = create_train_dataset()
    val_dataset = create_val_dataset()

    print()
    print(f"TRAIN images: {len(train_dataset)}")
    print(f"VALIDATION images: {len(val_dataset)}")

    # --------------------------------------------------------
    # SAMPLER
    # --------------------------------------------------------

    sampler = create_weighted_sampler(
        train_dataset
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SpineQualityResNet18V3(
        pretrained=True,
        dropout=DROPOUT
    )

    model = model.to(device)

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    # Здесь НЕ используем сильное class weighting одновременно
    # с WeightedRandomSampler.
    #
    # Sampler уже балансирует классы.
    #
    # Поэтому обычный CrossEntropyLoss.

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if "fc" in name:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": LEARNING_RATE_BACKBONE
            },
            {
                "params": head_parameters,
                "lr": LEARNING_RATE_HEAD
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # TRAINING VARIABLES
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_val_f1_1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_metrics["accuracy"]
        )

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch:02d}"
        )

        print(
            f"TRAIN loss: {train_loss:.4f} | "
            f"accuracy: {train_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"TRAIN class1 recall: "
            f"{train_metrics['recall_1'] * 100:.2f}% | "
            f"F1: "
            f"{train_metrics['f1_1'] * 100:.2f}%"
        )

        print(
            f"VAL loss: {val_loss:.4f} | "
            f"accuracy: {val_metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"VAL class1 precision: "
            f"{val_metrics['precision_1'] * 100:.2f}% | "
            f"recall: "
            f"{val_metrics['recall_1'] * 100:.2f}% | "
            f"F1: "
            f"{val_metrics['f1_1'] * 100:.2f}%"
        )

        print(
            f"VAL confusion: "
            f"TN={val_metrics['tn']} "
            f"FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']} "
            f"TP={val_metrics['tp']}"
        )

        print(
            f"LR backbone: {current_lr_backbone:.2e} | "
            f"LR head: {current_lr_head:.2e}"
        )

        # ----------------------------------------------------
        # СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
        # ----------------------------------------------------

        # Основной критерий:
        # validation accuracy.
        #
        # При одинаковой accuracy используем F1 класса 1.

        is_better = False

        if val_metrics["accuracy"] > best_val_accuracy:

            is_better = True

        elif (
            val_metrics["accuracy"] == best_val_accuracy
            and val_metrics["f1_1"] > best_val_f1_1
        ):

            is_better = True

        if is_better:

            best_val_accuracy = val_metrics["accuracy"]
            best_val_f1_1 = val_metrics["f1_1"]

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "val_accuracy":
                        val_metrics["accuracy"],

                    "val_f1_1":
                        val_metrics["f1_1"],

                    "val_recall_1":
                        val_metrics["recall_1"],

                    "val_precision_1":
                        val_metrics["precision_1"],

                    "val_loss":
                        val_loss,

                    "seed": SEED
                },
                MODEL_PATH
            )

            print()
            print(">>> ЛУЧШАЯ МОДЕЛЬ СОХРАНЕНА")
            print(
                f">>> Epoch: {epoch}"
            )
            print(
                f">>> Validation accuracy: "
                f"{val_metrics['accuracy'] * 100:.2f}%"
            )
            print(
                f">>> Validation class1 F1: "
                f"{val_metrics['f1_1'] * 100:.2f}%"
            )

        else:

            epochs_without_improvement += 1

            print(
                f">>> Улучшения нет: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                ">>> EARLY STOPPING"
            )

            break

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ОБУЧЕНИЕ V3 ЗАВЕРШЕНО")
    print("=" * 70)

    print()
    print(
        f"Лучшая эпоха: {best_epoch}"
    )

    print(
        f"Лучшая validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Лучший validation F1 класса 1: "
        f"{best_val_f1_1 * 100:.2f}%"
    )

    print()
    print(
        f"Модель сохранена:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()