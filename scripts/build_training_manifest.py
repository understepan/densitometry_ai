from pathlib import Path
import csv

import openpyxl
import pydicom


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = PROJECT_DIR / "data" / "raw" / "training"
STUDIES_DIR = TRAINING_DIR / "Исследования"
ANNOTATION_FILE = TRAINING_DIR / "разметка.xlsx"

OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "training_manifest.csv"


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def normalize_value(value):
    """
    Приводим пустые значения Excel к None.
    Остальные значения оставляем как есть.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return None

    return value


def load_annotations():
    """
    Загружает разметку из Excel.

    Возвращает словарь:

    {
        study_id: {
            ...
        }
    }
    """

    workbook = openpyxl.load_workbook(
        ANNOTATION_FILE,
        data_only=True
    )

    sheet = workbook["Калибровка"]

    annotations = {}

    # Строки 1-2 — заголовки.
    # Данные начинаются с 3-й строки.
    for row in sheet.iter_rows(min_row=3, values_only=True):

        study_id = normalize_value(row[1])

        if study_id is None:
            continue

        annotations[str(study_id)] = {
            "spine_position": normalize_value(row[2]),
            "spine_axis": normalize_value(row[3]),
            "spine_artifacts": normalize_value(row[4]),

            "right_hip_position": normalize_value(row[5]),
            "right_hip_roi": normalize_value(row[6]),

            "left_hip_position": normalize_value(row[7]),
            "left_hip_roi": normalize_value(row[8]),

            "spine_quality": normalize_value(row[9]),
            "right_hip_quality": normalize_value(row[10]),
            "left_hip_quality": normalize_value(row[11]),

            "comment": normalize_value(row[12]),
        }

    return annotations


def read_dicom_info(file_path):
    """
    Читает основные данные одного DICOM.
    """

    dataset = pydicom.dcmread(file_path)

    return {
        "dicom_study_uid": str(
            getattr(dataset, "StudyInstanceUID", "")
        ),

        "series_uid": str(
            getattr(dataset, "SeriesInstanceUID", "")
        ),

        "image_uid": str(
            getattr(dataset, "SOPInstanceUID", "")
        ),

        "modality": str(
            getattr(dataset, "Modality", "")
        ),

        "rows": getattr(dataset, "Rows", None),

        "columns": getattr(dataset, "Columns", None),

        "instance_number": getattr(
            dataset,
            "InstanceNumber",
            None
        ),
    }


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 80)
    print("ПОСТРОЕНИЕ TRAINING MANIFEST")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Проверяем наличие файлов
    # --------------------------------------------------------

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл разметки: {ANNOTATION_FILE}"
        )

    if not STUDIES_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка исследований: {STUDIES_DIR}"
        )

    # --------------------------------------------------------
    # 2. Загружаем Excel
    # --------------------------------------------------------

    annotations = load_annotations()

    print()
    print(f"Размеченных исследований в Excel: {len(annotations)}")

    # --------------------------------------------------------
    # 3. Собираем строки manifest
    # --------------------------------------------------------

    rows = []

    total_dicom = 0
    errors = 0

    study_dirs = sorted(
        [
            path
            for path in STUDIES_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print(f"Папок исследований: {len(study_dirs)}")

    print()

    # --------------------------------------------------------
    # 4. Обрабатываем каждое исследование
    # --------------------------------------------------------

    for study_dir in study_dirs:

        study_id = study_dir.name

        # Получаем разметку исследования
        annotation = annotations.get(study_id)

        if annotation is None:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] "
                f"Нет разметки для исследования: {study_id}"
            )

        dicom_files = sorted(
            study_dir.rglob("*.dcm")
        )

        for dicom_path in dicom_files:

            total_dicom += 1

            try:

                dicom_info = read_dicom_info(
                    dicom_path
                )

                row = {
                    "study_id": study_id,

                    "dicom_study_uid":
                        dicom_info["dicom_study_uid"],

                    "series_uid":
                        dicom_info["series_uid"],

                    "image_uid":
                        dicom_info["image_uid"],

                    "dicom_path":
                        str(
                            dicom_path.relative_to(
                                PROJECT_DIR
                            )
                        ),

                    "modality":
                        dicom_info["modality"],

                    "rows":
                        dicom_info["rows"],

                    "columns":
                        dicom_info["columns"],

                    "instance_number":
                        dicom_info["instance_number"],
                }

                # ------------------------------------------------
                # Добавляем разметку
                # ------------------------------------------------

                if annotation is not None:

                    row.update(annotation)

                else:

                    row.update({
                        "spine_position": None,
                        "spine_axis": None,
                        "spine_artifacts": None,

                        "right_hip_position": None,
                        "right_hip_roi": None,

                        "left_hip_position": None,
                        "left_hip_roi": None,

                        "spine_quality": None,
                        "right_hip_quality": None,
                        "left_hip_quality": None,

                        "comment": None,
                    })

                rows.append(row)

            except Exception as error:

                errors += 1

                print(
                    f"[ОШИБКА] {dicom_path}"
                )

                print(
                    f"  {error}"
                )

    # --------------------------------------------------------
    # 5. Сохраняем CSV
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "study_id",
        "dicom_study_uid",
        "series_uid",
        "image_uid",
        "dicom_path",
        "modality",
        "rows",
        "columns",
        "instance_number",

        "spine_position",
        "spine_axis",
        "spine_artifacts",

        "right_hip_position",
        "right_hip_roi",

        "left_hip_position",
        "left_hip_roi",

        "spine_quality",
        "right_hip_quality",
        "left_hip_quality",

        "comment",
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

        writer.writerows(rows)

    # --------------------------------------------------------
    # 6. Итог
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    print(
        f"Исследований в Excel: {len(annotations)}"
    )

    print(
        f"Папок исследований: {len(study_dirs)}"
    )

    print(
        f"DICOM-файлов обработано: {total_dicom}"
    )

    print(
        f"Ошибок: {errors}"
    )

    print()
    print(
        f"Manifest сохранён:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 80)


if __name__ == "__main__":
    main()