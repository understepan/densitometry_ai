from pathlib import Path
import csv
import hashlib
from collections import Counter, defaultdict

import pydicom


PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

SPLITS = {
    "TRAIN": PROJECT_DIR / "data" / "splits" / "train.csv",
    "VALIDATION": PROJECT_DIR / "data" / "splits" / "validation.csv",
    "TEST": PROJECT_DIR / "data" / "splits" / "test.csv",
}


def read_rows(csv_file):

    with open(
        csv_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = []

        for row in reader:

            # Используем только позвоночник
            if row["anatomy"] != "spine":
                continue

            # Только размеченные изображения
            if row["spine_quality"] not in {"0", "1"}:
                continue

            rows.append(row)

    return rows


def pixel_hash(path):

    dataset = pydicom.dcmread(path)

    return hashlib.sha256(
        dataset.pixel_array.tobytes()
    ).hexdigest()


def analyze_split(name, csv_file):

    rows = read_rows(csv_file)

    # hash -> первая строка
    unique = {}

    for row in rows:

        path = PROJECT_DIR / row["dicom_path"]

        h = pixel_hash(path)

        if h not in unique:
            unique[h] = row

    unique_rows = list(unique.values())

    studies = defaultdict(list)

    for row in unique_rows:
        studies[row["study_id"]].append(row)

    class_counts = Counter(
        row["spine_quality"]
        for row in unique_rows
    )

    study_classes = {}

    for study_id, study_rows in studies.items():

        classes = set(
            row["spine_quality"]
            for row in study_rows
        )

        study_classes[study_id] = classes

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Исходных строк:", len(rows))
    print("Уникальных PixelData:", len(unique_rows))
    print("Удалено дублей:", len(rows) - len(unique_rows))

    print()
    print("Уникальные изображения по классам:")

    print(
        "CLASS 0:",
        class_counts["0"]
    )

    print(
        "CLASS 1:",
        class_counts["1"]
    )

    print()
    print(
        "Уникальных исследований:",
        len(studies)
    )

    # Количество уникальных изображений на исследование
    image_per_study = Counter(
        len(items)
        for items in studies.values()
    )

    print()
    print("Уникальных изображений на исследование:")

    for count in sorted(image_per_study):
        print(
            f"  {count} изображение:",
            image_per_study[count],
            "исследований"
        )

    # Исследования по классам
    study_class_counts = Counter()

    for study_id, classes in study_classes.items():

        if classes == {"0"}:
            study_class_counts["0"] += 1

        elif classes == {"1"}:
            study_class_counts["1"] += 1

        else:
            study_class_counts["mixed"] += 1

    print()
    print("Исследования по классам:")

    print(
        "  CLASS 0:",
        study_class_counts["0"]
    )

    print(
        "  CLASS 1:",
        study_class_counts["1"]
    )

    print(
        "  MIXED:",
        study_class_counts["mixed"]
    )

    return unique_rows, studies


def main():

    print("=" * 70)
    print("АНАЛИЗ DATASET ДЛЯ V6")
    print("=" * 70)

    all_unique_hashes = {}

    for name, csv_file in SPLITS.items():

        unique_rows, studies = analyze_split(
            name,
            csv_file
        )

        for row in unique_rows:

            path = PROJECT_DIR / row["dicom_path"]

            h = pixel_hash(path)

            if h in all_unique_hashes:

                print()
                print(
                    "!!! ВНИМАНИЕ: одинаковый PixelData "
                    "между split !!!"
                )

                print(
                    "Первый:",
                    all_unique_hashes[h]
                )

                print(
                    "Второй:",
                    name,
                    row["dicom_path"]
                )

            else:

                all_unique_hashes[h] = (
                    name,
                    row["dicom_path"]
                )

    print()
    print("=" * 70)
    print("ИТОГ")
    print("=" * 70)

    print(
        "Всего уникальных PixelData:",
        len(all_unique_hashes)
    )

    print()
    print(
        "Если предупреждений о межсплитовых "
        "дублях нет — leakage не обнаружен."
    )


if __name__ == "__main__":
    main()