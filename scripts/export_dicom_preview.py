from pathlib import Path

import pydicom
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "examples"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dicom_files = sorted(DATA_DIR.rglob("*.dcm"))

    print(f"Найдено DICOM-файлов: {len(dicom_files)}")
    print()

    for file_path in dicom_files:
        print("=" * 70)
        print(f"Файл: {file_path.name}")

        dataset = pydicom.dcmread(file_path)
        pixels = dataset.pixel_array

        output_path = OUTPUT_DIR / f"{file_path.stem}.png"

        image = Image.fromarray(pixels)
        image.save(output_path)

        print(f"PNG сохранён: {output_path}")

    print()
    print("=" * 70)
    print("Экспорт завершён.")


if __name__ == "__main__":
    main()