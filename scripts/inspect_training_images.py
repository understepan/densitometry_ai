from pathlib import Path
from collections import defaultdict

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = PROJECT_DIR / "data" / "raw" / "training"
STUDIES_DIR = TRAINING_DIR / "Исследования"


def get_value(dataset, name):
    value = getattr(dataset, name, None)

    if value is None:
        return "<ОТСУТСТВУЕТ>"

    return str(value)


def main():
    print("=" * 80)
    print("ПРОВЕРКА МЕТАДАННЫХ ОБУЧАЮЩИХ DICOM")
    print("=" * 80)
    print()

    study_dirs = sorted(
        path
        for path in STUDIES_DIR.iterdir()
        if path.is_dir()
    )

    print(
        f"Исследований найдено: "
        f"{len(study_dirs)}"
    )

    print()

    # Берём первые 10 исследований,
    # чтобы сначала спокойно изучить структуру.
    for study_dir in study_dirs[:10]:

        dicom_files = sorted(
            study_dir.rglob("*.dcm")
        )

        print("=" * 80)
        print(
            f"ИССЛЕДОВАНИЕ: "
            f"{study_dir.name}"
        )
        print(
            f"DICOM-файлов: "
            f"{len(dicom_files)}"
        )
        print("=" * 80)

        for file_path in dicom_files:

            try:
                dataset = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=True
                )

                print()
                print(
                    f"Файл: "
                    f"{file_path.name}"
                )

                print(
                    f"  Modality: "
                    f"{get_value(dataset, 'Modality')}"
                )

                print(
                    f"  SOPInstanceUID: "
                    f"{get_value(dataset, 'SOPInstanceUID')}"
                )

                print(
                    f"  SeriesInstanceUID: "
                    f"{get_value(dataset, 'SeriesInstanceUID')}"
                )

                print(
                    f"  SeriesNumber: "
                    f"{get_value(dataset, 'SeriesNumber')}"
                )

                print(
                    f"  InstanceNumber: "
                    f"{get_value(dataset, 'InstanceNumber')}"
                )

                print(
                    f"  Rows: "
                    f"{get_value(dataset, 'Rows')}"
                )

                print(
                    f"  Columns: "
                    f"{get_value(dataset, 'Columns')}"
                )

                print(
                    f"  ImageType: "
                    f"{get_value(dataset, 'ImageType')}"
                )

                print(
                    f"  BodyPartExamined: "
                    f"{get_value(dataset, 'BodyPartExamined')}"
                )

                print(
                    f"  ViewPosition: "
                    f"{get_value(dataset, 'ViewPosition')}"
                )

                print(
                    f"  PatientPosition: "
                    f"{get_value(dataset, 'PatientPosition')}"
                )

            except Exception as error:

                print()
                print(
                    f"ОШИБКА: "
                    f"{file_path}"
                )

                print(
                    f"  {error}"
                )

        print()

    print("=" * 80)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80)


if __name__ == "__main__":
    main()