from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)

CONTROL_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "laterality_control.csv"
)


def main():
    with CONTROL_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        control_rows = list(csv.DictReader(f))

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        manifest_rows = list(csv.DictReader(f))

    print(f"Контрольных записей: {len(control_rows)}")
    print(f"Записей в image_manifest: {len(manifest_rows)}")
    print()

    for control in control_rows:
        study_id = control["study_id"]
        image_order = int(control["image_order"])
        laterality = control["laterality"]

        study_images = [
            row
            for row in manifest_rows
            if row["study_id"] == study_id
            and row["anatomy"] == "hip"
        ]

        study_images.sort(
            key=lambda row: (
                int(row["instance_number"])
                if row["instance_number"].isdigit()
                else 999999
            )
        )

        print("=" * 80)
        print(f"Study: {study_id}")
        print(f"Контроль: IMAGE {image_order} -> {laterality}")
        print()

        print("HIP DICOM в image_manifest:")

        for index, row in enumerate(study_images, start=1):
            print(
                f"  IMAGE {index}: "
                f"{Path(row['dicom_path']).name} | "
                f"InstanceNumber={row['instance_number']} | "
                f"{row['rows']}x{row['columns']} | "
                f"anatomy={row['anatomy']}"
            )

    print()
    print("=" * 80)
    print("Проверка завершена.")


if __name__ == "__main__":
    main()