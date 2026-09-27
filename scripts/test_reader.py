from pathlib import Path

from src.dicom.reader import read_dicom_pixels


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def main():
    dicom_files = sorted(DATA_DIR.rglob("*.dcm"))

    print(f"Найдено DICOM-файлов: {len(dicom_files)}")
    print()

    for file_path in dicom_files:
        pixels = read_dicom_pixels(str(file_path))

        print("=" * 70)
        print(f"Файл: {file_path.name}")
        print(f"Shape: {pixels.shape}")
        print(f"Dtype: {pixels.dtype}")
        print(f"Min: {pixels.min()}")
        print(f"Max: {pixels.max()}")

    print()
    print("=" * 70)
    print("Тест загрузчика завершён.")


if __name__ == "__main__":
    main()