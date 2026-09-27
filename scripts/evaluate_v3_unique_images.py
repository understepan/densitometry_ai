from pathlib import Path
import csv
import hashlib
from collections import defaultdict

import pydicom


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(
    r"C:\hakaton\densitometry_ai"
)

PREDICTIONS_FILE = (
    PROJECT_DIR
    / "results"
    / "spine_quality_v3_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "spine_quality_v3_unique_predictions.csv"
)


# ============================================================
# HASH PIXEL DATA
# ============================================================

def get_pixel_hash(dicom_path):

    full_path = (
        PROJECT_DIR / dicom_path
    )

    ds = pydicom.dcmread(
        full_path
    )

    pixel_array = ds.pixel_array

    return hashlib.md5(
        pixel_array.tobytes()
    ).hexdigest()


# ============================================================
# ЧТЕНИЕ PREDICTIONS
# ============================================================

with open(
    PREDICTIONS_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)

    fieldnames = reader.fieldnames


print("=" * 80)
print("V3 — ОЦЕНКА НА УНИКАЛЬНЫХ ИЗОБРАЖЕНИЯХ")
print("=" * 80)

print()
print(
    f"Исходных строк predictions: "
    f"{len(rows)}"
)


# ============================================================
# УДАЛЕНИЕ ДУБЛИКАТОВ ВНУТРИ STUDY
# ============================================================

unique_rows = []

seen = set()

duplicates = []

for row in rows:

    study_id = row["study_id"]

    dicom_path = row["dicom_path"]

    pixel_hash = get_pixel_hash(
        dicom_path
    )

    key = (
        study_id,
        pixel_hash
    )

    if key in seen:

        duplicates.append(
            {
                "study_id": study_id,
                "dicom_path": dicom_path,
                "hash": pixel_hash,
            }
        )

        continue

    seen.add(key)

    row = dict(row)

    row["pixel_hash"] = pixel_hash

    unique_rows.append(row)


# ============================================================
# ИНФОРМАЦИЯ О ДУБЛИКАТАХ
# ============================================================

print()
print("=" * 80)
print("НАЙДЕННЫЕ ДУБЛИКАТЫ")
print("=" * 80)

if not duplicates:

    print(
        "Дубликатов нет."
    )

else:

    for item in duplicates:

        print()
        print(
            f"Study: "
            f"{item['study_id']}"
        )

        print(
            f"Duplicate: "
            f"{item['dicom_path']}"
        )

        print(
            f"Hash: "
            f"{item['hash']}"
        )


print()
print(
    f"Удалено дубликатов: "
    f"{len(duplicates)}"
)

print(
    f"Уникальных изображений: "
    f"{len(unique_rows)}"
)


# ============================================================
# МЕТРИКИ
# ============================================================

def calculate_metrics(
    data,
    title
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    total = len(data)

    correct = sum(
        int(row["true_label"])
        == int(row["predicted_label"])
        for row in data
    )

    accuracy = (
        correct / total
        if total
        else 0
    )

    print()
    print(
        f"Images: "
        f"{total}"
    )

    print(
        f"Correct: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    true0_pred0 = 0
    true0_pred1 = 0
    true1_pred0 = 0
    true1_pred1 = 0

    for row in data:

        true = int(
            row["true_label"]
        )

        pred = int(
            row["predicted_label"]
        )

        if true == 0 and pred == 0:
            true0_pred0 += 1

        elif true == 0 and pred == 1:
            true0_pred1 += 1

        elif true == 1 and pred == 0:
            true1_pred0 += 1

        elif true == 1 and pred == 1:
            true1_pred1 += 1

    print()
    print("Confusion matrix:")

    print(
        f"True 0: "
        f"pred0={true0_pred0}, "
        f"pred1={true0_pred1}"
    )

    print(
        f"True 1: "
        f"pred0={true1_pred0}, "
        f"pred1={true1_pred1}"
    )

    # --------------------------------------------------------
    # Class 0
    # --------------------------------------------------------

    precision0_den = (
        true0_pred0
        + true1_pred0
    )

    recall0_den = (
        true0_pred0
        + true0_pred1
    )

    precision0 = (
        true0_pred0
        / precision0_den
        if precision0_den
        else 0
    )

    recall0 = (
        true0_pred0
        / recall0_den
        if recall0_den
        else 0
    )

    f1_0_den = (
        precision0
        + recall0
    )

    f1_0 = (
        2
        * precision0
        * recall0
        / f1_0_den
        if f1_0_den
        else 0
    )

    # --------------------------------------------------------
    # Class 1
    # --------------------------------------------------------

    precision1_den = (
        true1_pred1
        + true0_pred1
    )

    recall1_den = (
        true1_pred1
        + true1_pred0
    )

    precision1 = (
        true1_pred1
        / precision1_den
        if precision1_den
        else 0
    )

    recall1 = (
        true1_pred1
        / recall1_den
        if recall1_den
        else 0
    )

    f1_1_den = (
        precision1
        + recall1
    )

    f1_1 = (
        2
        * precision1
        * recall1
        / f1_1_den
        if f1_1_den
        else 0
    )

    print()
    print("CLASS 0")

    print(
        f"Precision: "
        f"{precision0 * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall0 * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{f1_0 * 100:.2f}%"
    )

    print()
    print("CLASS 1")

    print(
        f"Precision: "
        f"{precision1 * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall1 * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{f1_1 * 100:.2f}%"
    )

    print()
    print(
        f"True class0: "
        f"{true0_pred0 + true0_pred1}"
    )

    print(
        f"True class1: "
        f"{true1_pred0 + true1_pred1}"
    )


# ============================================================
# IMAGE-LEVEL UNIQUE METRICS
# ============================================================

calculate_metrics(
    unique_rows,
    "IMAGE-LEVEL: UNIQUE PIXEL DATA"
)


# ============================================================
# STUDY-LEVEL
#
# Для каждого study берём среднюю вероятность P1
# по уникальным изображениям.
# ============================================================

study_groups = defaultdict(list)

for row in unique_rows:

    study_groups[
        row["study_id"]
    ].append(row)


study_rows = []

for study_id, study_data in study_groups.items():

    p1_values = [
        float(row["p1"])
        for row in study_data
    ]

    mean_p1 = (
        sum(p1_values)
        / len(p1_values)
    )

    predicted = (
        1
        if mean_p1 >= 0.5
        else 0
    )

    true_values = [
        int(row["true_label"])
        for row in study_data
    ]

    # Все изображения study должны иметь
    # одинаковую истинную метку.
    if len(set(true_values)) != 1:

        print()
        print(
            "ВНИМАНИЕ: "
            f"study {study_id} "
            "имеет разные true labels."
        )

        continue

    true_label = true_values[0]

    study_rows.append(
        {
            "study_id": study_id,
            "true_label": str(
                true_label
            ),
            "predicted_label": str(
                predicted
            ),
            "mean_p1": mean_p1,
            "num_unique_images": len(
                study_data
            ),
        }
    )


calculate_metrics(
    study_rows,
    "STUDY-LEVEL: UNIQUE IMAGES"
)


# ============================================================
# ПОДРОБНАЯ STUDY-LEVEL СТАТИСТИКА
# ============================================================

print()
print("=" * 80)
print("STUDY-LEVEL ПОДРОБНО")
print("=" * 80)

for row in sorted(
    study_rows,
    key=lambda x: (
        int(x["true_label"]),
        x["study_id"]
    )
):

    status = (
        "CORRECT"
        if (
            int(row["true_label"])
            == int(row["predicted_label"])
        )
        else "ERROR"
    )

    print()

    print(
        f"{status} | "
        f"TRUE={row['true_label']} "
        f"PRED={row['predicted_label']} "
        f"mean_P1={row['mean_p1']:.4f} "
        f"unique_images="
        f"{row['num_unique_images']}"
    )

    print(
        f"Study: "
        f"{row['study_id']}"
    )


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

output_fields = list(
    unique_rows[0].keys()
) if unique_rows else []

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=output_fields
    )

    writer.writeheader()

    writer.writerows(
        unique_rows
    )


print()
print("=" * 80)
print("РЕЗУЛЬТАТ СОХРАНЁН")
print("=" * 80)

print(
    OUTPUT_FILE
)