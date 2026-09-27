from pathlib import Path
import sys
import csv
from collections import defaultdict

import torch
from torch.utils.data import DataLoader



# ============================================================
# Пути проекта
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


from src.preprocessing.dataset import SpineQualityDataset
from src.models.resnet18_model import SpineQualityResNet18


# ============================================================
# Настройки
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 1

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

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Проверки
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Модель не найдена:\n{MODEL_PATH}"
    )


# ============================================================
# Dataset
# ============================================================

dataset = SpineQualityDataset(
    csv_file=TEST_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("=" * 70)
print("ОЦЕНКА SPINE QUALITY НА УРОВНЕ ИССЛЕДОВАНИЯ")
print("=" * 70)

print()
print(f"Устройство: {DEVICE}")
print(f"TEST изображений: {len(dataset)}")


# ============================================================
# Загружаем модель
# ============================================================

print()
print("Создание ResNet18...")

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
print("Модель загружена:")
print(MODEL_PATH)


print()
print(
    f"Эпоха: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation Accuracy: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# Предсказания по изображениям
# ============================================================

image_results = []


with torch.no_grad():

    for index, batch in enumerate(loader):

        image = batch["image"].to(DEVICE)

        true_label = int(
            batch["label"].item()
        )

        study_id = batch["study_id"][0]

        dicom_path = batch["dicom_path"][0]


        # ----------------------------------------------------
        # Предсказание
        # ----------------------------------------------------

        output = model(image)


        # ----------------------------------------------------
        # Вероятности
        # ----------------------------------------------------

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


        probability_0 = float(
            probabilities[0].item()
        )

        probability_1 = float(
            probabilities[1].item()
        )


        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )


        correct = (
            prediction == true_label
        )


        image_results.append(
            {
                "index": index,
                "study_id": study_id,
                "dicom_path": dicom_path,
                "true_label": true_label,
                "predicted_label": prediction,
                "probability_0": probability_0,
                "probability_1": probability_1,
                "correct": correct,
            }
        )


# ============================================================
# Предсказания по изображениям
# ============================================================

print()
print("-" * 70)
print("ПРЕДСКАЗАНИЯ ПО ИЗОБРАЖЕНИЯМ")
print("-" * 70)

print()


for number, result in enumerate(
    image_results,
    start=1
):

    status = (
        "CORRECT"
        if result["correct"]
        else "ERROR"
    )


    print(
        f"{number:02d}. "
        f"TRUE={result['true_label']} "
        f"PRED={result['predicted_label']} "
        f"P0={result['probability_0']:.3f} "
        f"P1={result['probability_1']:.3f} "
        f"{status}"
    )


# ============================================================
# Группируем изображения по исследованиям
# ============================================================

studies = defaultdict(list)


for result in image_results:

    studies[
        result["study_id"]
    ].append(result)


# ============================================================
# Предсказания на уровне исследования
# ============================================================

study_results = []


for study_id, items in studies.items():

    true_labels = [
        item["true_label"]
        for item in items
    ]


    # Проверяем, что внутри одного исследования
    # label не меняется
    if len(set(true_labels)) != 1:

        raise ValueError(
            "В одном исследовании обнаружены "
            f"разные TRUE labels:\n{study_id}"
        )


    true_label = true_labels[0]


    # --------------------------------------------------------
    # Средние вероятности
    # --------------------------------------------------------

    mean_probability_0 = sum(
        item["probability_0"]
        for item in items
    ) / len(items)


    mean_probability_1 = sum(
        item["probability_1"]
        for item in items
    ) / len(items)


    # --------------------------------------------------------
    # Решение по исследованию
    # --------------------------------------------------------

    if (
        mean_probability_1
        >
        mean_probability_0
    ):

        prediction = 1

    else:

        prediction = 0


    correct = (
        prediction == true_label
    )


    study_results.append(
        {
            "study_id": study_id,
            "num_images": len(items),
            "true_label": true_label,
            "predicted_label": prediction,
            "mean_probability_0": mean_probability_0,
            "mean_probability_1": mean_probability_1,
            "correct": correct,
        }
    )


# ============================================================
# Сортируем исследования
# ============================================================

study_results.sort(
    key=lambda x: x["study_id"]
)


# ============================================================
# Вывод результатов по исследованиям
# ============================================================

print()
print("-" * 70)
print("ОЦЕНКА ПО ИССЛЕДОВАНИЯМ")
print("-" * 70)


for result in study_results:

    status = (
        "CORRECT"
        if result["correct"]
        else "ERROR"
    )


    print()

    print(
        f"study: "
        f"{result['study_id']}"
    )

    print(
        f"  Изображений: "
        f"{result['num_images']}"
    )

    print(
        f"  TRUE: "
        f"{result['true_label']}"
    )

    print(
        f"  PRED: "
        f"{result['predicted_label']}"
    )

    print(
        f"  Среднее P0: "
        f"{result['mean_probability_0']:.3f}"
    )

    print(
        f"  Среднее P1: "
        f"{result['mean_probability_1']:.3f}"
    )

    print(
        f"  {status}"
    )


# ============================================================
# Общая статистика
# ============================================================

y_true = [
    result["true_label"]
    for result in study_results
]

y_pred = [
    result["predicted_label"]
    for result in study_results
]


total_studies = len(
    study_results
)

correct_studies = sum(
    result["correct"]
    for result in study_results
)

error_studies = (
    total_studies
    - correct_studies
)


accuracy = (
    correct_studies
    / total_studies
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = [
    [0, 0],
    [0, 0]
]

for true_label, predicted_label in zip(
    y_true,
    y_pred
):

    cm[true_label][predicted_label] += 1


# ============================================================
# Метрики
# ============================================================

precision = []
recall = []
f1 = []


for class_id in [0, 1]:

    true_positive = cm[class_id][class_id]

    false_positive = sum(
        cm[other_class][class_id]
        for other_class in [0, 1]
        if other_class != class_id
    )

    false_negative = sum(
        cm[class_id][other_class]
        for other_class in [0, 1]
        if other_class != class_id
    )


    # Precision

    if (
        true_positive + false_positive
        > 0
    ):

        class_precision = (
            true_positive
            /
            (
                true_positive
                + false_positive
            )
        )

    else:

        class_precision = 0.0


    # Recall

    if (
        true_positive + false_negative
        > 0
    ):

        class_recall = (
            true_positive
            /
            (
                true_positive
                + false_negative
            )
        )

    else:

        class_recall = 0.0


    # F1

    if (
        class_precision + class_recall
        > 0
    ):

        class_f1 = (
            2
            * class_precision
            * class_recall
            /
            (
                class_precision
                + class_recall
            )
        )

    else:

        class_f1 = 0.0


    precision.append(
        class_precision
    )

    recall.append(
        class_recall
    )

    f1.append(
        class_f1
    )


# ============================================================
# Итог
# ============================================================

print()
print("=" * 70)
print("ИТОГ — УРОВЕНЬ ИССЛЕДОВАНИЯ")
print("=" * 70)

print()

print(
    f"Исследований: "
    f"{total_studies}"
)

print(
    f"Правильных: "
    f"{correct_studies}"
)

print(
    f"Ошибок: "
    f"{error_studies}"
)

print()

print(
    f"STUDY-LEVEL ACCURACY: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# Confusion Matrix
# ============================================================

print()
print("CONFUSION MATRIX")
print()

print(
    "                 Предсказание"
)

print(
    "                 0       1"
)

print(
    f"Истина 0       "
    f"{cm[0][0]:<7}"
    f"{cm[0][1]}"
)

print(
    f"Истина 1       "
    f"{cm[1][0]:<7}"
    f"{cm[1][1]}"
)


# ============================================================
# Метрики
# ============================================================

print()
print("МЕТРИКИ ПО КЛАССАМ")

print()

print("Класс 0:")

print(
    f"  Precision: "
    f"{precision[0] * 100:.2f}%"
)

print(
    f"  Recall:    "
    f"{recall[0] * 100:.2f}%"
)

print(
    f"  F1:        "
    f"{f1[0] * 100:.2f}%"
)


print()

print("Класс 1:")

print(
    f"  Precision: "
    f"{precision[1] * 100:.2f}%"
)

print(
    f"  Recall:    "
    f"{recall[1] * 100:.2f}%"
)

print(
    f"  F1:        "
    f"{f1[1] * 100:.2f}%"
)


# ============================================================
# Сохраняем CSV
# ============================================================

result_csv = (
    RESULTS_DIR
    / "spine_quality_study_level.csv"
)


with open(
    result_csv,
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    fieldnames = [
        "study_id",
        "num_images",
        "true_label",
        "predicted_label",
        "mean_probability_0",
        "mean_probability_1",
        "correct",
    ]


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )


    writer.writeheader()


    for result in study_results:

        writer.writerow(
            result
        )


# ============================================================
# Финальный вывод
# ============================================================

print()
print("=" * 70)

print(
    f"Результаты сохранены:\n"
    f"{result_csv}"
)

print("=" * 70)