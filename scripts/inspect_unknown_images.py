from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)


def main():

    print("=" * 80)
    print("ПОИСК UNKNOWN ИЗОБРАЖЕНИЙ")
    print("=" * 80)

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        unknown = [
            row
            for row in reader
            if row["anatomy"] == "unknown"
        ]

    print()
    print(
        f"Найдено unknown: {len(unknown)}"
    )

    print()

    for number, row in enumerate(
        unknown,
        start=1
    ):

        print("-" * 80)

        print(
            f"UNKNOWN #{number}"
        )

        print(
            f"study_id:        {row['study_id']}"
        )

        print(
            f"dicom_path:      {row['dicom_path']}"
        )

        print(
            f"series_uid:      {row['series_uid']}"
        )

        print(
            f"image_uid:       {row['image_uid']}"
        )

        print(
            f"instance_number: {row['instance_number']}"
        )

        print(
            f"dimensions:      "
            f"{row['rows']} x {row['columns']}"
        )

        print()

    print("=" * 80)


if __name__ == "__main__":
    main()