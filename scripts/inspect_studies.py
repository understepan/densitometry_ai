from pathlib import Path

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"


def main():
    dicom_files = list(DATA_DIR.rglob("*.dcm"))

    print(f"Всего DICOM-файлов: {len(dicom_files)}")

    if not dicom_files:
        print("DICOM-файлы не найдены.")
        return

    studies = {}

    for file_path in dicom_files:
        try:
            dataset = pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
            )

            study_uid = getattr(
                dataset,
                "StudyInstanceUID",
                None,
            )

            series_uid = getattr(
                dataset,
                "SeriesInstanceUID",
                None,
            )

            sop_uid = getattr(
                dataset,
                "SOPInstanceUID",
                None,
            )

            if study_uid is None:
                print(f"Нет StudyInstanceUID: {file_path}")
                continue

            if study_uid not in studies:
                studies[study_uid] = []

            studies[study_uid].append(
                {
                    "file": file_path,
                    "series_uid": series_uid,
                    "sop_uid": sop_uid,
                }
            )

        except Exception as error:
            print(
                f"Ошибка при чтении "
                f"{file_path}: {error}"
            )

    print()
    print(f"Найдено исследований: {len(studies)}")
    print()

    for study_uid, images in studies.items():
        print(f"StudyInstanceUID:")
        print(study_uid)

        print(
            f"Количество DICOM-файлов: "
            f"{len(images)}"
        )

        for image in images:
            print()
            print(f"  Файл: {image['file']}")
            print(
                f"  SeriesInstanceUID: "
                f"{image['series_uid']}"
            )
            print(
                f"  SOPInstanceUID: "
                f"{image['sop_uid']}"
            )

        print()
        print("-" * 70)


if __name__ == "__main__":
    main()