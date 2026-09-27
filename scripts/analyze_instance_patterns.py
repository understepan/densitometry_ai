from pathlib import Path
from collections import defaultdict

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)


def classify_by_columns(ds):
    columns = getattr(ds, "Columns", None)

    if columns == 300:
        return "spine"

    if columns == 280:
        return "hip"

    return "unknown"


def main():

    print("=" * 100)
    print("АНАЛИЗ INSTANCE NUMBER")
    print("=" * 100)

    study_dirs = sorted(
        path
        for path in STUDIES_DIR.iterdir()
        if path.is_dir()
    )

    for index, study_dir in enumerate(
        study_dirs,
        start=1
    ):

        images = []

        for dicom_path in sorted(
            study_dir.rglob("*.dcm")
        ):

            ds = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=True
            )

            instance = getattr(
                ds,
                "InstanceNumber",
                None
            )

            region = classify_by_columns(ds)

            images.append(
                (
                    instance,
                    region,
                    dicom_path.name,
                    getattr(ds, "Rows", None),
                    getattr(ds, "Columns", None)
                )
            )

        images.sort(
            key=lambda x: (
                x[0] is None,
                x[0] if x[0] is not None else 999999
            )
        )

        print()
        print(
            f"[{index:3}/100] "
            f"{study_dir.name}"
        )

        print(
            f"  DICOM: {len(images)}"
        )

        print(
            "  InstanceNumber:"
        )

        for (
            instance,
            region,
            filename,
            rows,
            columns
        ) in images:

            print(
                f"    {instance:>3} | "
                f"{region:<7} | "
                f"{rows}x{columns} | "
                f"{filename}"
            )

    print()
    print("=" * 100)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 100)


if __name__ == "__main__":
    main()