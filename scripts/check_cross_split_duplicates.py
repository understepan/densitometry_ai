from pathlib import Path
import csv
import hashlib
import pydicom


PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

SPLITS = {
    "TRAIN": PROJECT_DIR / "data" / "splits" / "train.csv",
    "VALIDATION": PROJECT_DIR / "data" / "splits" / "validation.csv",
    "TEST": PROJECT_DIR / "data" / "splits" / "test.csv",
}


def get_pixel_hash(dicom_path):
    """
    Возвращает SHA256-хэш PixelData.
    Одинаковый PixelData -> одинаковый хэш.
    """

    dataset = pydicom.dcmread(dicom_path, stop_before_pixels=False)

    if "PixelData" not in dataset:
        return None

    pixel_data = dataset.PixelData

    return hashlib.sha256(pixel_data).hexdigest()


def load_split(split_name, csv_path):
    """
    Загружает только изображения позвоночника
    с метками 0/1.
    """

    rows = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:

            if row.get("anatomy") != "spine":
                continue

            if row.get("spine_quality") not in {"0", "1"}:
                continue

            dicom_path = PROJECT_DIR / row["dicom_path"]

            if not dicom_path.exists():
                print(f"ОШИБКА: файл не найден: {dicom_path}")
                continue

            rows.append({
                "study_id": row["study_id"],
                "dicom_path": row["dicom_path"],
                "label": row["spine_quality"],
            })

    return rows


def main():

    print("=" * 70)
    print("ПРОВЕРКА ДУБЛИКАТОВ МЕЖДУ TRAIN / VALIDATION / TEST")
    print("=" * 70)

    split_rows = {}

    # ---------------------------------------------------------
    # 1. Загружаем выборки
    # ---------------------------------------------------------

    for split_name, csv_path in SPLITS.items():

        rows = load_split(split_name, csv_path)

        split_rows[split_name] = rows

        print()
        print(f"{split_name}:")
        print(f"  строк после фильтрации spine: {len(rows)}")

    # ---------------------------------------------------------
    # 2. Считаем PixelData hash
    # ---------------------------------------------------------

    hash_map = {
        "TRAIN": {},
        "VALIDATION": {},
        "TEST": {},
    }

    print()
    print("-" * 70)
    print("ВЫЧИСЛЕНИЕ SHA256 ДЛЯ PixelData")
    print("-" * 70)

    for split_name, rows in split_rows.items():

        for row in rows:

            dicom_path = PROJECT_DIR / row["dicom_path"]

            try:
                pixel_hash = get_pixel_hash(dicom_path)

            except Exception as error:
                print(
                    f"ОШИБКА чтения: {dicom_path}\n"
                    f"  {error}"
                )
                continue

            if pixel_hash is None:
                continue

            if pixel_hash not in hash_map[split_name]:
                hash_map[split_name][pixel_hash] = []

            hash_map[split_name][pixel_hash].append(row)

    # ---------------------------------------------------------
    # 3. Количество уникальных изображений
    # ---------------------------------------------------------

    print()
    print("-" * 70)
    print("УНИКАЛЬНЫЕ PixelData")
    print("-" * 70)

    for split_name in ["TRAIN", "VALIDATION", "TEST"]:

        total = len(split_rows[split_name])
        unique = len(hash_map[split_name])

        print(
            f"{split_name}: "
            f"{total} строк, "
            f"{unique} уникальных PixelData"
        )

    # ---------------------------------------------------------
    # 4. Проверяем пары split
    # ---------------------------------------------------------

    pairs = [
        ("TRAIN", "VALIDATION"),
        ("TRAIN", "TEST"),
        ("VALIDATION", "TEST"),
    ]

    all_cross_duplicates = []

    print()
    print("=" * 70)
    print("ПРОВЕРКА МЕЖДУ SPLIT")
    print("=" * 70)

    for split_a, split_b in pairs:

        hashes_a = set(hash_map[split_a].keys())
        hashes_b = set(hash_map[split_b].keys())

        common_hashes = hashes_a & hashes_b

        print()
        print(f"{split_a} ↔ {split_b}")
        print(f"Совпадающих PixelData: {len(common_hashes)}")

        if not common_hashes:
            print("  Дубликатов НЕ найдено.")
            continue

        print("  ВНИМАНИЕ: найдены одинаковые изображения!")

        for pixel_hash in sorted(common_hashes):

            rows_a = hash_map[split_a][pixel_hash]
            rows_b = hash_map[split_b][pixel_hash]

            for row_a in rows_a:
                for row_b in rows_b:

                    item = {
                        "split_a": split_a,
                        "study_a": row_a["study_id"],
                        "path_a": row_a["dicom_path"],
                        "label_a": row_a["label"],

                        "split_b": split_b,
                        "study_b": row_b["study_id"],
                        "path_b": row_b["dicom_path"],
                        "label_b": row_b["label"],

                        "hash": pixel_hash,
                    }

                    all_cross_duplicates.append(item)

                    print()
                    print(f"  HASH: {pixel_hash}")
                    print(
                        f"  {split_a}: "
                        f"{row_a['study_id']} | "
                        f"{row_a['dicom_path']} | "
                        f"label={row_a['label']}"
                    )
                    print(
                        f"  {split_b}: "
                        f"{row_b['study_id']} | "
                        f"{row_b['dicom_path']} | "
                        f"label={row_b['label']}"
                    )

    # ---------------------------------------------------------
    # 5. Сохраняем отчёт
    # ---------------------------------------------------------

    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        results_dir /
        "cross_split_duplicates.csv"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        fieldnames = [
            "split_a",
            "study_a",
            "path_a",
            "label_a",
            "split_b",
            "study_b",
            "path_b",
            "label_b",
            "hash",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in all_cross_duplicates:
            writer.writerow(item)

    # ---------------------------------------------------------
    # 6. Финальный результат
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ИТОГ")
    print("=" * 70)

    if len(all_cross_duplicates) == 0:

        print()
        print("✓ МЕЖДУ TRAIN / VALIDATION / TEST")
        print("  ТОЧНЫХ PixelData-ДУБЛИКАТОВ НЕ НАЙДЕНО.")
        print()
        print("✓ УТЕЧКИ ОДИНАКОВЫХ ИЗОБРАЖЕНИЙ")
        print("  МЕЖДУ SPLIT НЕ ОБНАРУЖЕНО.")

    else:

        print()
        print(
            f"⚠ НАЙДЕНО ДУБЛИКАТОВ: "
            f"{len(all_cross_duplicates)}"
        )

        print()
        print(
            "Нужно исключить дубликаты между "
            "разными split перед финальным обучением."
        )

    print()
    print(f"Отчёт сохранён:")
    print(output_file)


if __name__ == "__main__":
    main()