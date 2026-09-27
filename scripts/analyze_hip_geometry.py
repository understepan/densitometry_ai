from pathlib import Path
import csv
import math

import numpy as np
import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "laterality_ground_truth.csv"
)


def normalize_image(image):
    image = image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value == min_value:
        return np.zeros_like(image)

    return (
        (image - min_value)
        / (max_value - min_value)
    )


def analyze_image(image):
    normalized = normalize_image(image)

    # Используем тот же базовый порог,
    # который применяли при создании контрольных масок.
    threshold = np.percentile(
        normalized,
        75,
    )

    mask = normalized >= threshold

    height, width = mask.shape

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    # Центр яркой области.
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    # Нормированное положение центра:
    # 0.0 = край слева
    # 0.5 = центр
    # 1.0 = край справа
    centroid_x_norm = centroid_x / width

    centroid_y_norm = centroid_y / height

    # Насколько центр смещён относительно центра изображения.
    # < 0 = левее центра
    # > 0 = правее центра
    x_offset = centroid_x_norm - 0.5

    # Bounding box яркой области.
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    # Доля изображения, попавшая в маску.
    mask_ratio = float(mask.mean())

    # PCA — направление основной оси яркой области.
    points = np.column_stack(
        (
            xs.astype(np.float64),
            ys.astype(np.float64),
        )
    )

    centered = points - points.mean(axis=0)

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Берём собственный вектор
    # с максимальным собственным значением.
    main_vector = eigenvectors[
        :, np.argmax(eigenvalues)
    ]

    vx = float(main_vector[0])
    vy = float(main_vector[1])

    angle = math.degrees(
        math.atan2(vy, vx)
    )

    return {
        "height": height,
        "width": width,
        "threshold": threshold,
        "mask_ratio": mask_ratio,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_x_norm": centroid_x_norm,
        "centroid_y_norm": centroid_y_norm,
        "x_offset": x_offset,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "pca_angle": angle,
    }


def main():
    with GROUND_TRUTH_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    results = []

    print("=" * 100)
    print("АНАЛИЗ ГЕОМЕТРИИ КОНТРОЛЬНЫХ HIP-ИЗОБРАЖЕНИЙ")
    print("=" * 100)
    print()

    for row in rows:
        laterality = row["laterality"]

        if laterality == "unknown":
            continue

        study_id = row["study_id"]
        filename = row["dicom_filename"]

        study_dir = (
            PROJECT_DIR
            / "data"
            / "raw"
            / "training"
            / "Исследования"
            / study_id
        )

        matches = list(
            study_dir.rglob(filename)
        )

        if not matches:
            print(
                f"ОШИБКА: не найден "
                f"{study_id} / {filename}"
            )
            continue

        dicom_path = matches[0]

        ds = pydicom.dcmread(
            dicom_path,
        )

        image = ds.pixel_array

        features = analyze_image(image)

        if features is None:
            print(
                f"ОШИБКА: пустая маска "
                f"{study_id} / {filename}"
            )
            continue

        result = {
            "study_id": study_id,
            "dicom_filename": filename,
            "laterality": laterality,
            **features,
        }

        results.append(result)

        print("-" * 100)

        print(
            f"{study_id}"
        )

        print(
            f"{filename} | "
            f"TRUE={laterality}"
        )

        print(
            f"Размер: "
            f"{features['width']} x "
            f"{features['height']}"
        )

        print(
            f"Centroid X: "
            f"{features['centroid_x']:.2f}"
        )

        print(
            f"Centroid X normalized: "
            f"{features['centroid_x_norm']:.4f}"
        )

        print(
            f"X offset from center: "
            f"{features['x_offset']:+.4f}"
        )

        print(
            f"Centroid Y normalized: "
            f"{features['centroid_y_norm']:.4f}"
        )

        print(
            f"Mask ratio: "
            f"{features['mask_ratio']:.4f}"
        )

        print(
            f"Bounding box: "
            f"{features['bbox_width']} x "
            f"{features['bbox_height']}"
        )

        print(
            f"PCA angle: "
            f"{features['pca_angle']:.2f}°"
        )

    print()
    print("=" * 100)
    print("СРАВНЕНИЕ LEFT / RIGHT")
    print("=" * 100)
    print()

    left = [
        r
        for r in results
        if r["laterality"] == "left"
    ]

    right = [
        r
        for r in results
        if r["laterality"] == "right"
    ]

    def print_group(name, group):
        print(f"{name}: {len(group)} изображений")

        if not group:
            return

        centroid_values = [
            r["centroid_x_norm"]
            for r in group
        ]

        offset_values = [
            r["x_offset"]
            for r in group
        ]

        angle_values = [
            r["pca_angle"]
            for r in group
        ]

        print(
            f"  Centroid X normalized:"
            f" min={min(centroid_values):.4f},"
            f" max={max(centroid_values):.4f},"
            f" avg={np.mean(centroid_values):.4f}"
        )

        print(
            f"  X offset:"
            f" min={min(offset_values):+.4f},"
            f" max={max(offset_values):+.4f},"
            f" avg={np.mean(offset_values):+.4f}"
        )

        print(
            f"  PCA angle:"
            f" min={min(angle_values):.2f},"
            f" max={max(angle_values):.2f},"
            f" avg={np.mean(angle_values):.2f}"
        )

        print()

    print_group(
        "LEFT",
        left,
    )

    print_group(
        "RIGHT",
        right,
    )

    print("=" * 100)
    print("ПРОВЕРКА ПРОСТОГО ПРАВИЛА")
    print("=" * 100)
    print()

    print(
        "Правило-кандидат:"
    )

    print(
        "  centroid_x_norm > 0.5 -> LEFT"
    )

    print(
        "  centroid_x_norm < 0.5 -> RIGHT"
    )

    correct = 0
    total = 0

    for r in results:
        value = r["centroid_x_norm"]

        if value > 0.5:
            predicted = "left"
        else:
            predicted = "right"

        is_correct = (
            predicted == r["laterality"]
        )

        if is_correct:
            correct += 1

        total += 1

        print(
            f"{r['dicom_filename']:15s} "
            f"TRUE={r['laterality']:5s} "
            f"PRED={predicted:5s} "
            f"X={value:.4f} "
            f"{'OK' if is_correct else 'ERROR'}"
        )

    print()

    if total > 0:
        accuracy = (
            correct / total * 100
        )

        print(
            f"Правильно: "
            f"{correct} / {total}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.1f}%"
        )

    print()
    print("=" * 100)
    print("Анализ завершён.")
    print("=" * 100)


if __name__ == "__main__":
    main()