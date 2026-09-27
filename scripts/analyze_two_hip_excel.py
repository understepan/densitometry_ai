from pathlib import Path
import csv
from collections import defaultdict

import openpyxl


PROJECT_DIR = Path(__file__).resolve().parent.parent

EXCEL_FILE = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
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


def load_excel():
    workbook = openpyxl.load_workbook(
        EXCEL_FILE,
        data_only=True
    )

    sheet = workbook["Калибровка"]

    result = {}

    for row in sheet.iter_rows(
        min_row=3,
        values_only=True
    ):
        study_id = normalize(row[1])

        if study_id is None:
            continue

        result[str(study_id)] = {
            "right_position": normalize(row[5]),
            "right_roi": normalize(row[6]),
            "left_position": normalize(row[7]),
            "left_roi": normalize(row[8]),
            "right_quality": normalize(row[10]),
            "left_quality": normalize(row[11]),
        }

    return result


def has_right(annotation):
    return any(
        annotation[key] is not None
        for key in (
            "right_position",
            "right_roi",
            "right_quality",
        )
    )


def has_left(annotation):
    return any(
        annotation[key] is not None
        for key in (
            "left_position",
            "left_roi",
            "left_quality",
        )
    )


def load_manifest():
    groups = defaultdict(list)

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["anatomy"] == "hip":
                groups[row["study_id"]].append(row)

    return groups


def main():

    print("=" * 100)
    print("ИССЛЕДОВАНИЯ С РОВНО 2 HIP DICOM И EXCEL-РАЗМЕТКОЙ")
    print("=" * 100)

    annotations = load_excel()
    hip_groups = load_manifest()

    selected = []

    for study_id, images in hip_groups.items():

        if len(images) != 2:
            continue

        annotation = annotations.get(study_id)

        if annotation is None:
            continue

        right = has_right(annotation)
        left = has_left(annotation)

        selected.append(
            (
                study_id,
                images,
                right,
                left,
                annotation
            )
        )

    print()
    print(
        f"Всего исследований с 2 HIP DICOM: "
        f"{len(selected)}"
    )

    both = sum(
        1
        for item in selected
        if item[2] and item[3]
    )

    only_right = sum(
        1
        for item in selected
        if item[2] and not item[3]
    )

    only_left = sum(
        1
        for item in selected
        if not item[2] and item[3]
    )

    none = sum(
        1
        for item in selected
        if not item[2] and not item[3]
    )

    print()
    print("РАСПРЕДЕЛЕНИЕ")
    print("-" * 100)
    print(f"Правое + левое: {both}")
    print(f"Только правое:  {only_right}")
    print(f"Только левое:   {only_left}")
    print(f"Нет hip:        {none}")

    print()
    print("=" * 100)
    print("ДЕТАЛИ")
    print("=" * 100)

    for study_id, images, right, left, annotation in selected:

        print()
        print(f"STUDY: {study_id}")
        print(
            f"Excel: right={right}, left={left}"
        )

        print(
            "  RIGHT: "
            f"position={annotation['right_position']}, "
            f"roi={annotation['right_roi']}, "
            f"quality={annotation['right_quality']}"
        )

        print(
            "  LEFT:  "
            f"position={annotation['left_position']}, "
            f"roi={annotation['left_roi']}, "
            f"quality={annotation['left_quality']}"
        )

        for index, image in enumerate(images, start=1):

            print()
            print(f"  IMAGE {index}")
            print(f"    file: {image['dicom_path']}")
            print(f"    instance: {image['instance_number']}")
            print(
                f"    dimensions: "
                f"{image['rows']} x {image['columns']}"
            )

    print()
    print("=" * 100)
    print("ГОТОВО")
    print("=" * 100)


if __name__ == "__main__":
    main()