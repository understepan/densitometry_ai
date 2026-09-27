from pathlib import Path
import sys
from collections import Counter

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

IMAGE_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_CSV = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_best.pth"
)


# ============================================================
# Проверяем наличие модели
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Модель не найдена:\n{MODEL_PATH}"
    )


# ============================================================
# Загружаем Dataset
# ============================================================

test_dataset = SpineQualityDataset(
    csv_file=TEST_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("=" * 60)
print("ТЕСТИРОВАНИЕ SPINE QUALITY")
print("=" * 60)

print()
print(f"Устройство: {DEVICE}")
print(f"TEST изображений: {len(test_dataset)}")


# ============================================================
# Создаём модель
# ============================================================

model = SpineQualityResNet18(
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print()
print(
    f"Загружена модель из эпохи: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation Accuracy модели: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# Тестирование
# ============================================================

all_predictions = []
all_labels = []
all_studies = []

correct = 0
total = 0


with torch.no_grad():

    for batch in test_loader:

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        all_studies.extend(
            batch["study_id"]
        )


# ============================================================
# Общая accuracy
# ============================================================

accuracy = correct / total

print()
print("-" * 60)

print(
    f"Правильных предсказаний: "
    f"{correct} из {total}"
)

print(
    f"TEST Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# Confusion Matrix
# ============================================================

confusion = {
    0: {0: 0, 1: 0},
    1: {0: 0, 1: 0},
}

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):

    confusion[true_label][predicted_label] += 1


print()
print("CONFUSION MATRIX")
print()
print("                 Предсказание")
print("                 0       1")
print(
    f"Истина 0       "
    f"{confusion[0][0]:<7}"
    f"{confusion[0][1]}"
)

print(
    f"Истина 1       "
    f"{confusion[1][0]:<7}"
    f"{confusion[1][1]}"
)


# ============================================================
# Метрики по классам
# ============================================================

print()
print("МЕТРИКИ ПО КЛАССАМ")

for class_id in [0, 1]:

    tp = confusion[class_id][class_id]

    fp = sum(
        confusion[other][class_id]
        for other in [0, 1]
        if other != class_id
    )

    fn = sum(
        confusion[class_id][other]
        for other in [0, 1]
        if other != class_id
    )

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
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

    print()
    print(f"Класс {class_id}:")
    print(
        f"  Precision: {precision * 100:.2f}%"
    )
    print(
        f"  Recall:    {recall * 100:.2f}%"
    )
    print(
        f"  F1:        {f1 * 100:.2f}%"
    )


# ============================================================
# Распределение истинных классов
# ============================================================

true_counter = Counter(
    all_labels
)

pred_counter = Counter(
    all_predictions
)

print()
print("ИСТИННЫЕ КЛАССЫ:")

for class_id in [0, 1]:
    print(
        f"  {class_id}: "
        f"{true_counter[class_id]}"
    )

print()
print("ПРЕДСКАЗАННЫЕ КЛАССЫ:")

for class_id in [0, 1]:
    print(
        f"  {class_id}: "
        f"{pred_counter[class_id]}"
    )


# ============================================================
# Уникальные исследования
# ============================================================

unique_studies = set(
    all_studies
)

print()
print(
    f"Уникальных исследований в TEST: "
    f"{len(unique_studies)}"
)

print()
print("=" * 60)
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print("=" * 60)