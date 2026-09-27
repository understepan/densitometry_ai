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

        dataset = pydicom.dcmread(file_path)

        print(f"Rows: {dataset.Rows}")
        print(f"Columns: {dataset.Columns}")

        try:
            pixels = dataset.pixel_array

            print(f"Тип данных: {pixels.dtype}")
            print(f"Размер массива: {pixels.shape}")
            print(f"Минимальное значение: {pixels.min()}")
            print(f"Максимальное значение: {pixels.max()}")

        except Exception as error:
            print(f"Не удалось получить Pixel Array: {error}")

    print()
    print("=" * 70)
    print("Проверка Pixel Data завершена.")


if __name__ == "__main__":
    main()