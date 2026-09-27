from pathlib import Path

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def main():
    dicom_files = sorted(DATA_DIR.rglob("*.dcm"))

    print(f"Всего DICOM-файлов: {len(dicom_files)}")
    print()

    for file_path in dicom_files:
        print("=" * 70)
        print(f"Файл: {file_path.name}")

        dataset = pydicom.dcmread(
            file_path,
            stop_before_pixels=True,
        )

        study_uid = getattr(
            dataset,
            "StudyInstanceUID",
            "нет данных",
        )

        series_uid = getattr(
            dataset,
            "SeriesInstanceUID",
            "нет данных",
        )

        sop_uid = getattr(
            dataset,
            "SOPInstanceUID",
            "нет данных",
        )

        modality = getattr(
            dataset,
            "Modality",
            "нет данных",
        )

        rows = getattr(
            dataset,
            "Rows",
            "нет данных",
        )

        columns = getattr(
            dataset,
            "Columns",
            "нет данных",
        )

        print(f"Modality: {modality}")
        print(f"StudyInstanceUID: {study_uid}")
        print(f"SeriesInstanceUID: {series_uid}")
        print(f"SOPInstanceUID: {sop_uid}")
        print(f"Rows: {rows}")
        print(f"Columns: {columns}")

    print()
    print("=" * 70)
    print("Проверка завершена.")


if __name__ == "__main__":
    main()