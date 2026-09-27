from pathlib import Path
import csv


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

FINAL_MANIFEST = PROCESSED_DIR / "final_manifest.csv"
LATERALITY_FILE = PROCESSED_DIR / "laterality_predictions.csv"

OUTPUT_FILE = PROCESSED_DIR / "labeled_manifest.csv"


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
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 80)
    print("ПОСТРОЕНИЕ LABELED MANIFEST")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # 1. Проверяем файлы
    # --------------------------------------------------------

    if not FINAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Не найден: {FINAL_MANIFEST}"
        )

    if not LATERALITY_FILE.exists():
        raise FileNotFoundError(
            f"Не найден: {LATERALITY_FILE}"
        )

    # --------------------------------------------------------
    # 2. Загружаем данные
    # --------------------------------------------------------

    manifest_rows = read_csv_file(
        FINAL_MANIFEST
    )

    laterality_rows = read_csv_file(
        LATERALITY_FILE
    )

    print(
        f"Строк в final_manifest: "
        f"{len(manifest_rows)}"
    )

    print(
        f"Строк в laterality_predictions: "
        f"{len(laterality_rows)}"
    )

    print()

    # --------------------------------------------------------
    # 3. Индекс laterality
    # --------------------------------------------------------

    laterality_by_path = {}

    for row in laterality_rows:

        dicom_path = row["dicom_path"]

        if dicom_path in laterality_by_path:

            raise ValueError(
                "Дублирующийся dicom_path "
                f"в laterality_predictions: "
                f"{dicom_path}"
            )

        laterality_by_path[dicom_path] = row

    # --------------------------------------------------------
    # 4. Формируем labels
    # --------------------------------------------------------

    labeled_rows = []

    label_source_counts = {
        "excel": 0,
        "pca_pair": 0,
        "unknown": 0,
    }

    anatomy_counts = {
        "spine": 0,
        "hip": 0,
        "unknown": 0,
    }

    laterality_counts = {
        "left": 0,
        "right": 0,
        "unknown": 0,
    }

    skipped_rows = 0

    for row in manifest_rows:

        anatomy = row["anatomy"]

        # ----------------------------------------------------
        # ПОЗВОНОЧНИК
        # ----------------------------------------------------

        if anatomy == "spine":

            position = row["spine_position"]
            axis_or_roi = row["spine_axis"]
            artifacts = row["spine_artifacts"]
            quality = row["spine_quality"]

            laterality = "not_applicable"

            label_source = "excel"

            anatomy_counts["spine"] += 1

        # ----------------------------------------------------
        # БЕДРО
        # ----------------------------------------------------

        elif anatomy == "hip":

            prediction = laterality_by_path.get(
                row["dicom_path"]
            )

            if prediction is None:

                laterality = "unknown"

                position = ""
                axis_or_roi = ""
                artifacts = ""
                quality = ""

                label_source = "unknown"

            else:

                laterality = prediction[
                    "predicted_laterality"
                ]

                if laterality == "left":

                    position = row[
                        "left_hip_position"
                    ]

                    axis_or_roi = row[
                        "left_hip_roi"
                    ]

                    artifacts = ""

                    quality = row[
                        "left_hip_quality"
                    ]

                    label_source = "pca_pair"

                elif laterality == "right":

                    position = row[
                        "right_hip_position"
                    ]

                    axis_or_roi = row[
                        "right_hip_roi"
                    ]

                    artifacts = ""

                    quality = row[
                        "right_hip_quality"
                    ]

                    label_source = "pca_pair"

                else:

                    position = ""
                    axis_or_roi = ""
                    artifacts = ""
                    quality = ""

                    label_source = "unknown"

            anatomy_counts["hip"] += 1

        # ----------------------------------------------------
        # НЕИЗВЕСТНАЯ АНАТОМИЯ
        # ----------------------------------------------------

        else:

            anatomy_counts["unknown"] += 1

            laterality = "unknown"

            position = ""
            axis_or_roi = ""
            artifacts = ""
            quality = ""

            label_source = "unknown"

        # ----------------------------------------------------
        # Счётчики
        # ----------------------------------------------------

        laterality_counts[
            laterality
            if laterality in laterality_counts
            else "unknown"
        ] += 1

        label_source_counts[
            label_source
        ] += 1

        # ----------------------------------------------------
        # Создаём строку
        # ----------------------------------------------------

        labeled_row = dict(row)

        labeled_row[
            "label_anatomy"
        ] = anatomy

        labeled_row[
            "label_laterality"
        ] = laterality

        labeled_row[
            "label_position"
        ] = position

        labeled_row[
            "label_axis_or_roi"
        ] = axis_or_roi

        labeled_row[
            "label_artifacts"
        ] = artifacts

        labeled_row[
            "label_quality"
        ] = quality

        labeled_row[
            "label_source"
        ] = label_source

        labeled_rows.append(
            labeled_row
        )

    # --------------------------------------------------------
    # 5. Вывод статистики
    # --------------------------------------------------------

    print("=" * 80)
    print("ANATOMY")
    print("=" * 80)
    print()

    for key, value in anatomy_counts.items():

        print(
            f"{key}: {value}"
        )

    print()

    print("=" * 80)
    print("LATERALITY")
    print("=" * 80)
    print()

    for key, value in laterality_counts.items():

        print(
            f"{key}: {value}"
        )

    print()

    print("=" * 80)
    print("ИСТОЧНИК LABEL")
    print("=" * 80)
    print()

    for key, value in label_source_counts.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # 6. Сохраняем
    # --------------------------------------------------------

    fieldnames = list(
        manifest_rows[0].keys()
    )

    fieldnames.extend([
        "label_anatomy",
        "label_laterality",
        "label_position",
        "label_axis_or_roi",
        "label_artifacts",
        "label_quality",
        "label_source",
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

        writer.writerows(
            labeled_rows
        )

    # --------------------------------------------------------
    # 7. Финальная проверка
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ФИНАЛЬНАЯ ПРОВЕРКА")
    print("=" * 80)
    print()

    print(
        f"Входных строк: "
        f"{len(manifest_rows)}"
    )

    print(
        f"Выходных строк: "
        f"{len(labeled_rows)}"
    )

    print(
        f"Пропущено строк: "
        f"{skipped_rows}"
    )

    print()

    if (
        len(manifest_rows) == 499
        and len(labeled_rows) == 499
        and skipped_rows == 0
        and anatomy_counts["spine"] == 166
        and anatomy_counts["hip"] == 333
        and label_source_counts["pca_pair"] == 60
    ):

        print(
            "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ."
        )

    else:

        print(
            "ЕСТЬ НЕСООТВЕТСТВИЯ."
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