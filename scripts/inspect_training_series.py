from pathlib import Path
import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent
STUDIES_DIR = PROJECT_DIR / "data" / "raw" / "training" / "Исследования"


def main():
    print("=" * 100)
    print("АНАЛИЗ СЕРИЙ TRAINING DATA")
    print("=" * 100)

    study_dirs = sorted(
        path for path in STUDIES_DIR.iterdir()
        if path.is_dir()
    )

    print(f"Исследований найдено: {len(study_dirs)}")
    print()

    for study_index, study_dir in enumerate(study_dirs, start=1):

        dicom_files = sorted(study_dir.rglob("*.dcm"))

        print("-" * 100)
        print(f"[{study_index}/{len(study_dirs)}] {study_dir.name}")
        print(f"DICOM-файлов: {len(dicom_files)}")

        for dicom_path in dicom_files:
            try:
                ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

                series_uid = str(
                    getattr(ds, "SeriesInstanceUID", "")
                )

                series_number = getattr(
                    ds,
                    "SeriesNumber",
                    None
                )

                instance_number = getattr(
                    ds,
                    "InstanceNumber",
                    None
                )

                rows = getattr(ds, "Rows", None)
                columns = getattr(ds, "Columns", None)

                print(
                    f"  {dicom_path.name:30} "
                    f"SeriesNumber={series_number!s:4} "
                    f"InstanceNumber={instance_number!s:4} "
                    f"Размер={rows}x{columns}"
                )

            except Exception as error:
                print(f"  ОШИБКА: {dicom_path}")
                print(f"  {error}")

    print()
    print("=" * 100)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 100)


if __name__ == "__main__":
    main()