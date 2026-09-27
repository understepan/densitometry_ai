from pathlib import Path
import csv
from collections import defaultdict


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)


def main():

    print("=" * 100)
    print("АНАЛИЗ ГРУПП ИЗОБРАЖЕНИЙ БЕДРА")
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

            groups[row["study_id"]].append(row)

    total_studies = 0

    for study_id, images in groups.items():

        hip_images = [
            row
            for row in images
            if row["anatomy"] == "hip"
        ]

        if not hip_images:
            continue

        total_studies += 1

        print()
        print("=" * 100)
        print(f"STUDY: {study_id}")
        print("=" * 100)

        print(
            f"Всего изображений: {len(images)}"
        )

        print(
            f"Позвоночник: "
            f"{sum(row['anatomy'] == 'spine' for row in images)}"
        )

        print(
            f"Бедро: {len(hip_images)}"
        )

        print()
        print(
            "HIP ИЗОБРАЖЕНИЯ:"
        )

        for number, row in enumerate(
            hip_images,
            start=1
        ):

            print(
                f"  #{number:02d} | "
                f"size={row['rows']}x{row['columns']} | "
                f"Instance={row['instance_number']} | "
                f"Series={row['series_uid'][-12:]} | "
                f"{row['dicom_path']}"
            )

    print()
    print("=" * 100)
    print("ИТОГ")
    print("=" * 100)

    print(
        f"Исследований с изображениями бедра: "
        f"{total_studies}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()