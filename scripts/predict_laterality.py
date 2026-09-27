from pathlib import Path
import csv

import numpy as np
import pydicom


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FINAL_MANIFEST = PROCESSED_DIR / "final_manifest.csv"

OUTPUT_FILE = PROCESSED_DIR / "laterality_predictions.csv"


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

# Граница угла, которую мы уже проверяли
# на контрольных парах.
ANGLE_THRESHOLD = 80.0


# ============================================================
# ЧТЕНИЕ CSV
# ============================================================

def read_csv_file(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        return list(reader)


# ============================================================
# НОРМАЛИЗАЦИЯ ИЗОБРАЖЕНИЯ
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum == minimum:
        return np.zeros_like(image)

    return (
        (image - minimum)
        / (maximum - minimum)
    )


# ============================================================
# PCA-УГОЛ
# ============================================================

def calculate_pca_angle(image):

    normalized = normalize_image(image)

    # Используем верхние 25% значений яркости,
    # как в предыдущем анализе.
    threshold = np.percentile(
        normalized,
        75
    )

    mask = normalized >= threshold

    coordinates = np.column_stack(
        np.where(mask)
    )

    # Для корректного PCA нужно минимум
    # несколько точек.
    if len(coordinates) < 2:
        return None

    # Центрируем координаты.
    centered = (
        coordinates
        - coordinates.mean(axis=0)
    )

    # PCA через SVD.
    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False
    )

    principal_vector = vh[0]

    y_component = principal_vector[0]
    x_component = principal_vector[1]

    angle = np.degrees(
        np.arctan2(
            y_component,
            x_component
        )
    )

    # Переводим угол в диапазон 0...180.
    angle = angle % 180

    return float(angle)


# ============================================================
# ПРЕДСКАЗАНИЕ СТОРОНЫ
# ============================================================

def predict_laterality(angle):

    if angle is None:
        return "unknown"

    if angle > ANGLE_THRESHOLD:
        return "left"

    return "right"


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 80)
    print("ПРЕДСКАЗАНИЕ LATERALITY ДЛЯ HIP")
    print("=" * 80)
    print()

    if not FINAL_MANIFEST.exists():

        raise FileNotFoundError(
            f"Не найден файл: {FINAL_MANIFEST}"
        )

    rows = read_csv_file(
        FINAL_MANIFEST
    )

    print(
        f"Строк в final_manifest: {len(rows)}"
    )

    print()

    # --------------------------------------------------------
    # Группируем HIP по исследованиям
    # --------------------------------------------------------

    hip_groups = {}

    for row in rows:

        if row["anatomy"] != "hip":
            continue

        study_id = row["study_id"]

        hip_groups.setdefault(
            study_id,
            []
        ).append(row)

    # --------------------------------------------------------
    # Обрабатываем исследования
    # --------------------------------------------------------

    predictions = []

    studies_with_two_hips = 0
    studies_with_other_count = 0

    errors = 0

    for study_id in sorted(hip_groups):

        hip_rows = hip_groups[study_id]

        # PCA-предсказание сейчас применяем
        # только к исследованиям ровно с двумя HIP.
        if len(hip_rows) != 2:

            studies_with_other_count += 1
            continue

        studies_with_two_hips += 1

        study_predictions = []

        for row in hip_rows:

            dicom_path = (
                PROJECT_DIR
                / row["dicom_path"]
            )

            try:

                dataset = pydicom.dcmread(
                    dicom_path
                )

                image = dataset.pixel_array

                angle = calculate_pca_angle(
                    image
                )

                laterality = predict_laterality(
                    angle
                )

                study_predictions.append({
                    "study_id": study_id,
                    "dicom_path": row["dicom_path"],
                    "rows": row["rows"],
                    "columns": row["columns"],
                    "pca_angle": (
                        ""
                        if angle is None
                        else f"{angle:.4f}"
                    ),
                    "predicted_laterality":
                        laterality,
                })

            except Exception as error:

                errors += 1

                print(
                    f"[ОШИБКА] {dicom_path}"
                )

                print(
                    f"  {error}"
                )

        # ----------------------------------------------------
        # Проверяем, что пара действительно разделилась
        # ----------------------------------------------------

        if len(study_predictions) == 2:

            predictions.extend(
                study_predictions
            )

    # --------------------------------------------------------
    # Сохраняем
    # --------------------------------------------------------

    fieldnames = [
        "study_id",
        "dicom_path",
        "rows",
        "columns",
        "pca_angle",
        "predicted_laterality",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            predictions
        )

    # --------------------------------------------------------
    # Статистика
    # --------------------------------------------------------

    left_count = sum(
        1
        for row in predictions
        if row["predicted_laterality"] == "left"
    )

    right_count = sum(
        1
        for row in predictions
        if row["predicted_laterality"] == "right"
    )

    unknown_count = sum(
        1
        for row in predictions
        if row["predicted_laterality"] == "unknown"
    )

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)
    print()

    print(
        f"HIP-исследований всего: "
        f"{len(hip_groups)}"
    )

    print(
        f"Исследований ровно с 2 HIP: "
        f"{studies_with_two_hips}"
    )

    print(
        f"Исследований с другим количеством HIP: "
        f"{studies_with_other_count}"
    )

    print()

    print(
        f"Получено предсказаний: "
        f"{len(predictions)}"
    )

    print(
        f"LEFT: {left_count}"
    )

    print(
        f"RIGHT: {right_count}"
    )

    print(
        f"UNKNOWN: {unknown_count}"
    )

    print()

    print(
        f"Ошибок: {errors}"
    )

    print()

    print(
        "Файл сохранён:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()