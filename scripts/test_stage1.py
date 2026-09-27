from pathlib import Path

from src.dicom.reader import read_dicom_pixels
from src.dicom.study_loader import load_study


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Для теста"


def main():
    print("=" * 70)
    print("ФИНАЛЬНАЯ ПРОВЕРКА STAGE 1")
    print("=" * 70)
    print()

    # 1. Поиск и группировка исследований
    studies = load_study(str(DATA_DIR))

    print(f"Найдено исследований: {len(studies)}")
    print()

    total_images = 0

    # 2. Проверка каждого изображения
    for study_uid, images in studies.items():
        print("-" * 70)
        print(f"StudyInstanceUID: {study_uid}")
        print(f"Изображений в исследовании: {len(images)}")
        print()

        for image in images:
            file_path = Path(image["file_path"])

            pixels = read_dicom_pixels(str(file_path))

            print(f"Файл: {file_path.name}")
            print(f"  Shape: {pixels.shape}")
            print(f"  Dtype: {pixels.dtype}")
            print(f"  Min: {pixels.min()}")
            print(f"  Max: {pixels.max()}")

            if pixels.ndim != 2:
                raise ValueError(
                    f"Изображение не является 2D: {file_path.name}"
                )

            if pixels.size == 0:
                raise ValueError(
                    f"Изображение пустое: {file_path.name}"
                )

            total_images += 1
            print("  Проверка: OK")
            print()

    print("=" * 70)
    print("ИТОГ")
    print("=" * 70)
    print(f"Исследований: {len(studies)}")
    print(f"Изображений: {total_images}")
    print()
    print("Все проверки Stage 1 пройдены.")
    print("=" * 70)


if __name__ == "__main__":
    main()