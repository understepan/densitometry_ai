from pathlib import Path

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def main():
    dicom_files = sorted(DATA_DIR.rglob("*.dcm"))

    print(f"Всего DICOM-файлов: {len(dicom_files)}")
    print()

    tags_to_check = [
        "Modality",
        "PhotometricInterpretation",
        "SamplesPerPixel",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "RescaleIntercept",
        "RescaleSlope",
        "WindowCenter",
        "WindowWidth",
        "PixelSpacing",
        "ImageOrientationPatient",
        "ImagePositionPatient",
    ]

    for file_path in dicom_files:
        print("=" * 70)
        print(f"Файл: {file_path.name}")

        dataset = pydicom.dcmread(file_path)

        for tag_name in tags_to_check:
            value = getattr(dataset, tag_name, "ОТСУТСТВУЕТ")
            print(f"{tag_name}: {value}")

        print()

    print("=" * 70)
    print("Проверка метаданных завершена.")


if __name__ == "__main__":
    main()