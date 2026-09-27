from pathlib import Path
import csv
from collections import defaultdict

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)


def main():
    print("=" * 100)
    print("АНАЛИЗ ПАР HIP-ИЗОБРАЖЕНИЙ ПО PATIENT ORIENTATION")
    print("=" * 100)

    groups = defaultdict(list)

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["anatomy"] == "hip":
                groups[row["study_id"]].append(row)

    # Берём только исследования ровно с двумя hip DICOM.
    pairs = []

    for study_id, images in groups.items():
        if len(images) == 2:
            pairs.append(
                (study_id, images)
            )

    print()
    print(f"Исследований с ровно 2 HIP DICOM: {len(pairs)}")
    print()

    print("=" * 100)
    print("ПАРЫ")
    print("=" * 100)

    for study_id, images in pairs:

        print()
        print(f"STUDY: {study_id}")

        for index, row in enumerate(images, start=1):

            path = Path(row["dicom_path"])

            try:
                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True
                )

                orientation = getattr(
                    ds,
                    "PatientOrientation",
                    None
                )

                instance_number = getattr(
                    ds,
                    "InstanceNumber",
                    None
                )

                series_number = getattr(
                    ds,
                    "SeriesNumber",
                    None
                )

                print()
                print(f"  IMAGE {index}")
                print(f"    file: {path.name}")
                print(f"    instance: {instance_number}")
                print(f"    series: {series_number}")
                print(f"    orientation: {orientation}")
                print(
                    f"    dimensions: "
                    f"{row['rows']} x {row['columns']}"
                )

            except Exception as error:
                print()
                print(f"    ОШИБКА: {error}")

    print()
    print("=" * 100)
    print("ГОТОВО")
    print("=" * 100)


if __name__ == "__main__":
    main()