from pathlib import Path
import csv

import pydicom


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)


# ============================================================
# ПРЕДВАРИТЕЛЬНАЯ КЛАССИФИКАЦИЯ АНАТОМИИ
# ============================================================

def guess_anatomy(rows, columns):
    """
    Предварительная классификация анатомической области.

    Для данного набора данных:
    - ширина 300 наблюдается у изображений позвоночника;
    - ширина 280 наблюдается у изображений бедра;
    - два визуально проверенных изображения исследования
      2.25.12798473087614376830819854616447908612
      имеют размеры 401x248 и 405x248 и также относятся к бедру.

    Это правило предназначено для подготовки датасета,
    а не является медицинским алгоритмом.
    """

    # Позвоночник
    if columns == 300:
        return "spine"

    # Основная группа изображений бедра
    if columns == 280:
        return "hip"

    # Визуально проверенные исключения
    if (rows, columns) in {
        (401, 248),
        (405, 248),
    }:
        return "hip"

    return "unknown"


# ============================================================
# ЧТЕНИЕ DICOM
# ============================================================

def read_dicom_info(dicom_path):

    dataset = pydicom.dcmread(dicom_path)

    rows = getattr(dataset, "Rows", None)
    columns = getattr(dataset, "Columns", None)

    if rows is not None:
        rows = int(rows)

    if columns is not None:
        columns = int(columns)

    return {
        "series_uid": str(
            getattr(dataset, "SeriesInstanceUID", "")
        ),
        "image_uid": str(
            getattr(dataset, "SOPInstanceUID", "")
        ),
        "instance_number": getattr(
            dataset,
            "InstanceNumber",
            None
        ),
        "rows": rows,
        "columns": columns,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ПОСТРОЕНИЕ IMAGE MANIFEST")
    print("=" * 80)

    if not STUDIES_DIR.exists():

        raise FileNotFoundError(
            f"Не найдена папка исследований:\n"
            f"{STUDIES_DIR}"
        )

    study_dirs = sorted(
        [
            path
            for path in STUDIES_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print()
    print(
        f"Исследований найдено: {len(study_dirs)}"
    )

    rows = []

    total_dicom = 0
    errors = 0

    anatomy_counts = {
        "spine": 0,
        "hip": 0,
        "unknown": 0,
    }

    # ========================================================
    # ОБХОД ИССЛЕДОВАНИЙ
    # ========================================================

    for study_dir in study_dirs:

        study_id = study_dir.name

        dicom_files = sorted(
            study_dir.rglob("*.dcm")
        )

        print()
        print(
            f"{study_id}: "
            f"{len(dicom_files)} DICOM"
        )

        for dicom_path in dicom_files:

            total_dicom += 1

            try:

                info = read_dicom_info(
                    dicom_path
                )

                anatomy = guess_anatomy(
                    info["rows"],
                    info["columns"]
                )

                anatomy_counts[anatomy] += 1

                row = {
                    "study_id": study_id,

                    "dicom_path": str(
                        dicom_path.relative_to(
                            PROJECT_DIR
                        )
                    ),

                    "series_uid": info[
                        "series_uid"
                    ],

                    "image_uid": info[
                        "image_uid"
                    ],

                    "instance_number": info[
                        "instance_number"
                    ],

                    "rows": info["rows"],

                    "columns": info[
                        "columns"
                    ],

                    "anatomy": anatomy,

                    # Пока НЕ определяем
                    # правое/левое бедро.
                    "laterality": "",
                }

                rows.append(row)

            except Exception as error:

                errors += 1

                print(
                    f"[ОШИБКА] {dicom_path}"
                )

                print(
                    f"         {error}"
                )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "study_id",
        "dicom_path",
        "series_uid",
        "image_uid",
        "instance_number",
        "rows",
        "columns",
        "anatomy",
        "laterality",
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

    # ========================================================
    # ИТОГ
    # ========================================================

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    print(
        f"Исследований: {len(study_dirs)}"
    )

    print(
        f"DICOM-файлов: {total_dicom}"
    )

    print(
        f"Ошибок: {errors}"
    )

    print()

    print(
        f"spine:   {anatomy_counts['spine']}"
    )

    print(
        f"hip:     {anatomy_counts['hip']}"
    )

    print(
        f"unknown: {anatomy_counts['unknown']}"
    )

    print()

    print(
        "Manifest сохранён:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 80)


if __name__ == "__main__":
    main()