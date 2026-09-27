from pathlib import Path
import csv

import pydicom
from openpyxl import load_workbook


PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = PROJECT_DIR / "data" / "raw" / "training"
STUDIES_DIR = TRAINING_DIR / "Исследования"
ANNOTATIONS_FILE = TRAINING_DIR / "разметка.xlsx"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "study_manifest.csv"


def normalize(value):
    if value is None:
        return ""

    return str(value).strip()


def load_annotations():
    """
    Загружает разметку из Excel.

    Ключ:
        study = имя папки исследования
    """

    workbook = load_workbook(
        ANNOTATIONS_FILE,
        read_only=True,
        data_only=True
    )

    sheet = workbook["Калибровка"]

    annotations = {}

    for row in sheet.iter_rows(
        min_row=3,
        values_only=True
    ):
        study = normalize(row[1])

        if not study:
            continue

        annotations[study] = {
            "number": normalize(row[0]),

            # Позвоночник
            "spine_position": normalize(row[2]),
            "spine_axis": normalize(row[3]),
            "spine_artifacts": normalize(row[4]),

            # Правое бедро
            "right_hip_position": normalize(row[5]),
            "right_hip_roi": normalize(row[6]),

            # Левое бедро
            "left_hip_position": normalize(row[7]),
            "left_hip_roi": normalize(row[8]),

            # Итоги
            "spine_result": normalize(row[9]),
            "right_hip_result": normalize(row[10]),
            "left_hip_result": normalize(row[11]),

            # Комментарий
            "comment": normalize(row[12]),

            # Числовые оценки
            "spine_score": normalize(row[14]),
            "right_hip_score": normalize(row[15]),
            "left_hip_score": normalize(row[16]),

            # Норма / Патология
            "overall_label": normalize(row[17]),

            # Общая оценка
            "overall_score": normalize(row[18]),
        }

    workbook.close()

    return annotations


def get_dicom_information(study_dir):
    """
    Читает все DICOM внутри одного исследования.
    """

    dicom_files = sorted(
        study_dir.rglob("*.dcm")
    )

    if not dicom_files:
        return None

    first_dataset = pydicom.dcmread(
        dicom_files[0],
        stop_before_pixels=True
    )

    dicom_study_uid = normalize(
        getattr(
            first_dataset,
            "StudyInstanceUID",
            None
        )
    )

    relative_paths = []

    for file_path in dicom_files:
        relative_path = file_path.relative_to(
            PROJECT_DIR
        )

        relative_paths.append(
            str(relative_path)
        )

    return {
        "dicom_count": len(dicom_files),
        "dicom_study_uid": dicom_study_uid,
        "dicom_paths": " | ".join(relative_paths),
    }


def main():
    print("=" * 80)
    print("СОЗДАНИЕ STUDY MANIFEST")
    print("=" * 80)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 1. Загружаем Excel
    # ---------------------------------------------------------

    annotations = load_annotations()

    print(
        f"Исследований в Excel: "
        f"{len(annotations)}"
    )

    print()

    # ---------------------------------------------------------
    # 2. Находим папки исследований
    # ---------------------------------------------------------

    study_dirs = sorted(
        path
        for path in STUDIES_DIR.iterdir()
        if path.is_dir()
    )

    print(
        f"Папок исследований: "
        f"{len(study_dirs)}"
    )

    print()

    # ---------------------------------------------------------
    # 3. Заголовки CSV
    # ---------------------------------------------------------

    fieldnames = [
        "study_id",
        "dicom_study_uid",
        "dicom_count",
        "dicom_paths",

        "spine_position",
        "spine_axis",
        "spine_artifacts",

        "right_hip_position",
        "right_hip_roi",

        "left_hip_position",
        "left_hip_roi",

        "spine_result",
        "right_hip_result",
        "left_hip_result",

        "comment",

        "spine_score",
        "right_hip_score",
        "left_hip_score",

        "overall_label",
        "overall_score",
    ]

    rows = []

    missing_annotations = []
    missing_dicom = []

    # ---------------------------------------------------------
    # 4. Собираем строки
    # ---------------------------------------------------------

    for study_dir in study_dirs:

        study_id = study_dir.name

        if study_id not in annotations:
            missing_annotations.append(
                study_id
            )
            continue

        dicom_info = get_dicom_information(
            study_dir
        )

        if dicom_info is None:
            missing_dicom.append(
                study_id
            )
            continue

        annotation = annotations[study_id]

        row = {
            "study_id": study_id,

            "dicom_study_uid":
                dicom_info["dicom_study_uid"],

            "dicom_count":
                dicom_info["dicom_count"],

            "dicom_paths":
                dicom_info["dicom_paths"],
        }

        row.update(annotation)

        rows.append(row)

    # ---------------------------------------------------------
    # 5. Сохраняем CSV
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # ---------------------------------------------------------
    # 6. Итог
    # ---------------------------------------------------------

    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)
    print()

    print(
        f"Создано строк: "
        f"{len(rows)}"
    )

    print(
        f"Не найдено разметок: "
        f"{len(missing_annotations)}"
    )

    print(
        f"Не найдено DICOM: "
        f"{len(missing_dicom)}"
    )

    print()

    print(
        f"CSV сохранён:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    if missing_annotations:
        print("Исследования без разметки:")

        for study_id in missing_annotations:
            print(f"  {study_id}")

        print()

    if missing_dicom:
        print("Исследования без DICOM:")

        for study_id in missing_dicom:
            print(f"  {study_id}")

        print()

    print("=" * 80)


if __name__ == "__main__":
    main()