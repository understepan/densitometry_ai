from pathlib import Path
import csv
import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "laterality_ground_truth.csv"
)


def read_dicom_info(study_id, dicom_filename, laterality):
    study_dir = (
        PROJECT_DIR
        / "data"
        / "raw"
        / "training"
        / "Исследования"
        / study_id
    )

    matches = list(study_dir.rglob(dicom_filename))

    if not matches:
        print(f"ФАЙЛ НЕ НАЙДЕН: {study_id} / {dicom_filename}")
        return

    path = matches[0]

    ds = pydicom.dcmread(path, stop_before_pixels=True)

    print("=" * 80)
    print(f"Study:       {study_id}")
    print(f"DICOM:       {dicom_filename}")
    print(f"Laterality:  {laterality}")
    print(f"Path:        {path.relative_to(PROJECT_DIR)}")
    print()

    tags = [
        ("PatientOrientation", "PatientOrientation"),
        ("ImageOrientationPatient", "ImageOrientationPatient"),
        ("ImagePositionPatient", "ImagePositionPatient"),
        ("SliceLocation", "SliceLocation"),
        ("InstanceNumber", "InstanceNumber"),
        ("SeriesNumber", "SeriesNumber"),
        ("SeriesInstanceUID", "SeriesInstanceUID"),
        ("StudyInstanceUID", "StudyInstanceUID"),
        ("Rows", "Rows"),
        ("Columns", "Columns"),
        ("PixelSpacing", "PixelSpacing"),
        ("Laterality", "Laterality"),
        ("ImageLaterality", "ImageLaterality"),
        ("ViewPosition", "ViewPosition"),
        ("BodyPartExamined", "BodyPartExamined"),
    ]

    for label, attribute in tags:
        value = getattr(ds, attribute, "<нет тега>")
        print(f"{label}: {value}")


def main():
    with GROUND_TRUTH_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    print(f"Контрольных записей: {len(rows)}")
    print()

    for row in rows:
        if row["laterality"] == "unknown":
            continue

        read_dicom_info(
            study_id=row["study_id"],
            dicom_filename=row["dicom_filename"],
            laterality=row["laterality"],
        )

    print()
    print("=" * 80)
    print("Анализ завершён.")


if __name__ == "__main__":
    main()