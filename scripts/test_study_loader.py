from pathlib import Path

from src.dicom.study_loader import load_study


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Для теста"


def main():
    studies = load_study(str(DATA_DIR))

    print(f"Найдено исследований: {len(studies)}")
    print()

    for study_uid, images in studies.items():
        print("=" * 70)
        print(f"StudyInstanceUID: {study_uid}")
        print(f"Количество изображений: {len(images)}")

        for image in images:
            print(f"  Файл: {Path(image['file_path']).name}")
            print(f"  SeriesInstanceUID: {image['series_uid']}")
            print(f"  SOPInstanceUID: {image['sop_uid']}")
            print()

    print("=" * 70)
    print("Тест загрузки исследований завершён.")


if __name__ == "__main__":
    main()