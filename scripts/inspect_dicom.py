from pathlib import Path

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def main():
    dicom_files = list(DATA_DIR.rglob("*.dcm"))

    print(f"Найдено DICOM-файлов: {len(dicom_files)}")

    if not dicom_files:
        print("DICOM-файлы не найдены.")
        return

    file_path = dicom_files[0]

    print()
    print(f"Читаем файл:")
    print(file_path)

    dataset = pydicom.dcmread(file_path)

    print()
    print("DICOM успешно прочитан.")

    print()
    print("Основная информация:")

    print(
        f"Modality: "
        f"{getattr(dataset, 'Modality', 'нет данных')}"
    )

    print(
        f"StudyInstanceUID: "
        f"{getattr(dataset, 'StudyInstanceUID', 'нет данных')}"
    )

    print(
        f"SeriesInstanceUID: "
        f"{getattr(dataset, 'SeriesInstanceUID', 'нет данных')}"
    )

    print(
        f"SOPInstanceUID: "
        f"{getattr(dataset, 'SOPInstanceUID', 'нет данных')}"
    )

    print(
        f"Rows: "
        f"{getattr(dataset, 'Rows', 'нет данных')}"
    )

    print(
        f"Columns: "
        f"{getattr(dataset, 'Columns', 'нет данных')}"
    )


if __name__ == "__main__":
    main()