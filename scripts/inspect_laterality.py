from pathlib import Path

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)


STUDY_NAMES = [
    "1.2.840.113619.2.110.512719.20250403090444",
    "1.2.840.113619.2.110.512719.20250403092052",
    "2.25.114887542067602662452692698814728642117",
]


def get_value(ds, name):
    value = getattr(ds, name, None)

    if value is None:
        return "<нет>"

    return str(value)


def main():
    print("=" * 100)
    print("ПРОВЕРКА DICOM-ТЕГОВ ДЛЯ ОПРЕДЕЛЕНИЯ СТОРОНЫ")
    print("=" * 100)

    for study_name in STUDY_NAMES:

        study_dir = STUDIES_DIR / study_name

        print()
        print("=" * 100)
        print(f"ИССЛЕДОВАНИЕ: {study_name}")
        print("=" * 100)

        dicom_files = sorted(study_dir.rglob("*.dcm"))

        for dicom_path in dicom_files:

            ds = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=True
            )

            print()
            print(f"Файл: {dicom_path.name}")
            print(
                f"InstanceNumber: "
                f"{get_value(ds, 'InstanceNumber')}"
            )
            print(
                f"ImageLaterality: "
                f"{get_value(ds, 'ImageLaterality')}"
            )
            print(
                f"Laterality: "
                f"{get_value(ds, 'Laterality')}"
            )
            print(
                f"PatientPosition: "
                f"{get_value(ds, 'PatientPosition')}"
            )
            print(
                f"ViewPosition: "
                f"{get_value(ds, 'ViewPosition')}"
            )
            print(
                f"BodyPartExamined: "
                f"{get_value(ds, 'BodyPartExamined')}"
            )

    print()
    print("=" * 100)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 100)


if __name__ == "__main__":
    main()