from pathlib import Path
import csv
import math

import numpy as np
import pydicom
import matplotlib.pyplot as plt


PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

PREDICTIONS_FILE = (
    PROJECT_DIR / "results" / "spine_quality_v3_unique_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "visualizations"
    / "contact_sheets"
    / "v3_borderline_4.png"
)

TARGET_STUDIES = {
    "2.25.126293418391331346694592071563510265596",
    "2.25.296380774?",  # заменится поиском ниже
    "2.25.489064774?",  # заменится поиском ниже
    "2.25.338439598607961312040815816824093323492",
}


def load_predictions():
    with open(PREDICTIONS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_target_rows(rows):
    """
    Ищем 126293 и 338439 по полному ID,
    а 296380 и 489064 — по окончанию ID,
    чтобы не пришлось вручную переписывать длинные идентификаторы.
    """
    result = []

    for row in rows:
        study_id = row["study_id"]

        if (
            "126293418391331346694592071563510265596" in study_id
            or "296380" in study_id
            or "489064" in study_id
            or "338439598607961312040815816824093323492" in study_id
        ):
            result.append(row)

    return result


def load_image(dicom_path):
    full_path = PROJECT_DIR / dicom_path

    ds = pydicom.dcmread(full_path)
    image = ds.pixel_array.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image)

    return image


def short_study_id(study_id):
    if "126293" in study_id:
        return "126293"
    if "296380" in study_id:
        return "296380"
    if "489064" in study_id:
        return "489064"
    if "338439" in study_id:
        return "338439"

    return study_id[-8:]


def main():
    rows = load_predictions()
    target_rows = find_target_rows(rows)

    # Оставляем только нужные 4 исследования
    selected = {}

    for row in target_rows:
        sid = short_study_id(row["study_id"])
        selected[sid] = row

    order = ["126293", "296380", "489064", "338439"]

    selected_rows = []

    for sid in order:
        if sid in selected:
            selected_rows.append(selected[sid])

    if len(selected_rows) != 4:
        print("ОШИБКА: найдено не 4 исследования.")
        print("Найдено:", [short_study_id(r["study_id"]) for r in selected_rows])
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, row in zip(axes, selected_rows):
        image = load_image(row["dicom_path"])

        true_label = int(row["true_label"])
        predicted_label = int(row["predicted_label"])
        p0 = float(row["p0"])
        p1 = float(row["p1"])

        ax.imshow(image, cmap="gray")
        ax.axis("off")

        sid = short_study_id(row["study_id"])

        title = (
            f"Study: {sid}\n"
            f"TRUE={true_label}   PRED={predicted_label}\n"
            f"P(class 0)={p0:.3f}   P(class 1)={p1:.3f}\n"
            f"Size: {image.shape[1]}×{image.shape[0]}"
        )

        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print("КОНТАКТНЫЙ ЛИСТ СОЗДАН")
    print()
    print(f"Файл: {OUTPUT_FILE}")
    print()

    for row in selected_rows:
        print(
            short_study_id(row["study_id"]),
            "|",
            "TRUE =", row["true_label"],
            "|",
            "PRED =", row["predicted_label"],
            "|",
            "P1 =", row["p1"],
            "|",
            row["dicom_path"],
        )


if __name__ == "__main__":
    main()