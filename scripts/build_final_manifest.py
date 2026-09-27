from pathlib import Path
import csv


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TRAINING_MANIFEST = PROCESSED_DIR / "training_manifest.csv"
IMAGE_MANIFEST = PROCESSED_DIR / "image_manifest.csv"

OUTPUT_FILE = PROCESSED_DIR / "final_manifest.csv"


# ============================================================
# ЗАГРУЗКА CSV
# ============================================================

def read_csv_file(file_path):
    """Читает CSV и возвращает список словарей."""

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        return list(reader)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 80)
    print("ПОСТРОЕНИЕ FINAL MANIFEST")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # 1. Проверяем файлы
    # --------------------------------------------------------

    if not TRAINING_MANIFEST.exists():
        raise FileNotFoundError(
            f"Не найден: {TRAINING_MANIFEST}"
        )

    if not IMAGE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Не найден: {IMAGE_MANIFEST}"
        )

    # --------------------------------------------------------
    # 2. Загружаем manifest
    # --------------------------------------------------------

    training_rows = read_csv_file(
        TRAINING_MANIFEST
    )

    image_rows = read_csv_file(
        IMAGE_MANIFEST
    )

    print(
        f"Строк в training_manifest: "
        f"{len(training_rows)}"
    )

    print(
        f"Строк в image_manifest: "
        f"{len(image_rows)}"
    )

    print()

    # --------------------------------------------------------
    # 3. Индексируем image_manifest
    # --------------------------------------------------------

    image_by_path = {}

    for row in image_rows:

        dicom_path = row["dicom_path"]

        if dicom_path in image_by_path:

            raise ValueError(
                "Дублирующийся dicom_path "
                f"в image_manifest: {dicom_path}"
            )

        image_by_path[dicom_path] = row

    # --------------------------------------------------------
    # 4. Объединяем данные
    # --------------------------------------------------------

    final_rows = []

    missing_in_image_manifest = []

    for training_row in training_rows:

        dicom_path = training_row["dicom_path"]

        image_row = image_by_path.get(
            dicom_path
        )

        if image_row is None:

            missing_in_image_manifest.append(
                dicom_path
            )

            continue

        # Берём информацию о конкретном изображении
        # из image_manifest.

        final_row = dict(training_row)

        final_row["anatomy"] = image_row["anatomy"]

        final_row["laterality"] = image_row["laterality"]

        final_rows.append(final_row)

    # --------------------------------------------------------
    # 5. Проверка
    # --------------------------------------------------------

    print("=" * 80)
    print("ПРОВЕРКА ОБЪЕДИНЕНИЯ")
    print("=" * 80)
    print()

    print(
        f"Успешно объединено: "
        f"{len(final_rows)}"
    )

    print(
        f"Не найдено в image_manifest: "
        f"{len(missing_in_image_manifest)}"
    )

    if missing_in_image_manifest:

        print()
        print("Первые отсутствующие файлы:")

        for path in missing_in_image_manifest[:10]:

            print(path)

    # --------------------------------------------------------
    # 6. Проверяем anatomy
    # --------------------------------------------------------

    anatomy_counts = {}

    for row in final_rows:

        anatomy = row["anatomy"]

        anatomy_counts[anatomy] = (
            anatomy_counts.get(anatomy, 0) + 1
        )

    print()
    print("=" * 80)
    print("ANATOMY")
    print("=" * 80)
    print()

    for anatomy, count in sorted(
        anatomy_counts.items()
    ):

        print(
            f"{anatomy}: {count}"
        )

    # --------------------------------------------------------
    # 7. Проверяем laterality
    # --------------------------------------------------------

    laterality_counts = {}

    for row in final_rows:

        laterality = row["laterality"]

        if laterality == "":
            laterality = "unknown"

        laterality_counts[laterality] = (
            laterality_counts.get(
                laterality,
                0
            ) + 1
        )

    print()
    print("=" * 80)
    print("LATERALITY")
    print("=" * 80)
    print()

    for laterality, count in sorted(
        laterality_counts.items()
    ):

        print(
            f"{laterality}: {count}"
        )

    # --------------------------------------------------------
    # 8. Проверяем количество исследований
    # --------------------------------------------------------

    study_ids = {
        row["study_id"]
        for row in final_rows
    }

    print()
    print("=" * 80)
    print("ИССЛЕДОВАНИЯ")
    print("=" * 80)
    print()

    print(
        f"Уникальных исследований: "
        f"{len(study_ids)}"
    )

    # --------------------------------------------------------
    # 9. Сохраняем
    # --------------------------------------------------------

    fieldnames = list(
        training_rows[0].keys()
    )

    fieldnames.extend([
        "anatomy",
        "laterality",
    ])

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

        writer.writerows(final_rows)

    # --------------------------------------------------------
    # 10. Итог
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)
    print()

    print(
        f"training_manifest: "
        f"{len(training_rows)} строк"
    )

    print(
        f"image_manifest: "
        f"{len(image_rows)} строк"
    )

    print(
        f"final_manifest: "
        f"{len(final_rows)} строк"
    )

    print()

    if (
        len(training_rows) == 499
        and len(image_rows) == 499
        and len(final_rows) == 499
        and len(missing_in_image_manifest) == 0
        and len(study_ids) == 100
    ):

        print(
            "ВСЕ ПРОВЕРКИ УСПЕШНО ПРОЙДЕНЫ."
        )

    else:

        print(
            "ЕСТЬ НЕСООТВЕТСТВИЯ — "
            "ОБУЧЕНИЕ ПОКА НЕ НАЧИНАЕМ."
        )

    print()
    print(
        f"Файл сохранён:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()