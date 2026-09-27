from pathlib import Path
import csv
import openpyxl
from collections import defaultdict


PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
)

ANNOTATION_FILE = (
    TRAINING_DIR
    / "разметка.xlsx"
)

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)


def normalize(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

    return value


def load_annotations():

    workbook = openpyxl.load_workbook(
        ANNOTATION_FILE,
        data_only=True
    )

    sheet = workbook["Калибровка"]

    annotations = {}

    for row in sheet.iter_rows(
        min_row=3,
        values_only=True
    ):

        study_id = normalize(row[1])

        if study_id is None:
            continue

        annotations[str(study_id)] = {
            "right_position": normalize(row[5]),
            "right_roi": normalize(row[6]),
            "left_position": normalize(row[7]),
            "left_roi": normalize(row[8]),
            "right_quality": normalize(row[10]),
            "left_quality": normalize(row[11]),
        }

    return annotations


def load_manifest():

    groups = defaultdict(list)

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:
            groups[row["study_id"]].append(row)

    return groups


def main():

    print("=" * 90)
    print("АНАЛИЗ HIP-ИЗОБРАЖЕНИЙ И РАЗМЕТКИ EXCEL")
    print("=" * 90)

    annotations = load_annotations()
    manifest = load_manifest()

    print()
    print(
        f"Исследований в Excel: {len(annotations)}"
    )

    print(
        f"Исследований в manifest: {len(manifest)}"
    )

    print()

    both_hips_labeled = []
    only_right_labeled = []
    only_left_labeled = []
    no_hip_annotation = []

    for study_id, annotation in annotations.items():

        right_exists = (
            annotation["right_position"] is not None
            or annotation["right_roi"] is not None
            or annotation["right_quality"] is not None
        )

        left_exists = (
            annotation["left_position"] is not None
            or annotation["left_roi"] is not None
            or annotation["left_quality"] is not None
        )

        if right_exists and left_exists:
            both_hips_labeled.append(study_id)

        elif right_exists:
            only_right_labeled.append(study_id)

        elif left_exists:
            only_left_labeled.append(study_id)

        else:
            no_hip_annotation.append(study_id)

    print("=" * 90)
    print("РАЗМЕТКА EXCEL")
    print("=" * 90)

    print(
        f"Есть правое И левое бедро: "
        f"{len(both_hips_labeled)}"
    )

    print(
        f"Есть только правое бедро: "
        f"{len(only_right_labeled)}"
    )

    print(
        f"Есть только левое бедро: "
        f"{len(only_left_labeled)}"
    )

    print(
        f"Нет hip-разметки: "
        f"{len(no_hip_annotation)}"
    )

    print()

    # ========================================================
    # СВЯЗЫВАЕМ С КОЛИЧЕСТВОМ HIP DICOM
    # ========================================================

    print("=" * 90)
    print("СВЯЗЬ EXCEL ↔ DICOM")
    print("=" * 90)

    interesting = []

    for study_id, annotation in annotations.items():

        images = manifest.get(
            study_id,
            []
        )

        hip_images = [
            row
            for row in images
            if row["anatomy"] == "hip"
        ]

        right_exists = (
            annotation["right_position"] is not None
            or annotation["right_roi"] is not None
            or annotation["right_quality"] is not None
        )

        left_exists = (
            annotation["left_position"] is not None
            or annotation["left_roi"] is not None
            or annotation["left_quality"] is not None
        )

        if right_exists or left_exists:

            interesting.append(
                (
                    study_id,
                    len(hip_images),
                    right_exists,
                    left_exists
                )
            )

    for item in interesting:

        study_id, hip_count, right, left = item

        print(
            f"{study_id} | "
            f"hip DICOM={hip_count} | "
            f"right={right} | "
            f"left={left}"
        )

    print()

    print("=" * 90)
    print("ИТОГ")
    print("=" * 90)

    print(
        f"Исследований с hip-разметкой: "
        f"{len(interesting)}"
    )


if __name__ == "__main__":
    main()