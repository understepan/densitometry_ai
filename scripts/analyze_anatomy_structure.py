from pathlib import Path
from collections import Counter

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)


def provisional_region(ds):
    """
    Предварительная классификация по размеру изображения.

    По визуальной проверке:
    Columns=300 -> позвоночник
    Columns=280 -> бедро

    Остальные варианты пока считаем UNKNOWN.
    """

    columns = getattr(ds, "Columns", None)

    if columns == 300:
        return "spine_candidate"

    if columns == 280:
        return "hip_candidate"

    return "unknown"


def main():

    print("=" * 100)
    print("АНАЛИЗ АНАТОМИЧЕСКОЙ СТРУКТУРЫ TRAINING DATA")
    print("=" * 100)

    study_dirs = sorted(
        path
        for path in STUDIES_DIR.iterdir()
        if path.is_dir()
    )

    print()
    print(f"Исследований: {len(study_dirs)}")

    total_dicom = 0

    region_counter = Counter()
    columns_counter = Counter()
    study_count_counter = Counter()

    unknown_images = []

    print()

    for study_index, study_dir in enumerate(
        study_dirs,
        start=1
    ):

        dicom_files = sorted(
            study_dir.rglob("*.dcm")
        )

        study_count_counter[len(dicom_files)] += 1

        study_regions = []

        for dicom_path in dicom_files:

            try:

                ds = pydicom.dcmread(
                    dicom_path,
                    stop_before_pixels=True
                )

                total_dicom += 1

                columns = getattr(
                    ds,
                    "Columns",
                    None
                )

                rows = getattr(
                    ds,
                    "Rows",
                    None
                )

                instance_number = getattr(
                    ds,
                    "InstanceNumber",
                    None
                )

                region = provisional_region(ds)

                columns_counter[columns] += 1
                region_counter[region] += 1

                study_regions.append(
                    (
                        instance_number,
                        region,
                        rows,
                        columns,
                        dicom_path.name
                    )
                )

                if region == "unknown":

                    unknown_images.append(
                        (
                            study_dir.name,
                            dicom_path.name,
                            instance_number,
                            rows,
                            columns
                        )
                    )

            except Exception as error:

                print()
                print("ОШИБКА:")
                print(dicom_path)
                print(error)

        print(
            f"[{study_index:3}/{len(study_dirs)}] "
            f"{study_dir.name} | "
            f"DICOM={len(dicom_files)} | "
            f"spine_candidate="
            f"{sum(1 for x in study_regions if x[1] == 'spine_candidate')} | "
            f"hip_candidate="
            f"{sum(1 for x in study_regions if x[1] == 'hip_candidate')} | "
            f"unknown="
            f"{sum(1 for x in study_regions if x[1] == 'unknown')}"
        )

    print()
    print("=" * 100)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 100)

    print()
    print(f"Всего DICOM: {total_dicom}")

    print()
    print("Распределение Columns:")

    for columns, count in sorted(
        columns_counter.items(),
        key=lambda x: (
            x[0] is None,
            x[0] if x[0] is not None else -1
        )
    ):
        print(
            f"  Columns={columns}: {count}"
        )

    print()
    print("Предварительная классификация:")

    for region, count in region_counter.items():
        print(
            f"  {region}: {count}"
        )

    print()
    print("Количество DICOM на исследование:")

    for count, studies_count in sorted(
        study_count_counter.items()
    ):
        print(
            f"  {count} DICOM: "
            f"{studies_count} исследований"
        )

    print()
    print("=" * 100)
    print("UNKNOWN ИЗОБРАЖЕНИЯ")
    print("=" * 100)

    if not unknown_images:

        print()
        print("UNKNOWN изображений нет.")

    else:

        print()

        for (
            study_id,
            filename,
            instance_number,
            rows,
            columns
        ) in unknown_images:

            print(
                f"{study_id} | "
                f"{filename} | "
                f"Instance={instance_number} | "
                f"Размер={rows}x{columns}"
            )

    print()
    print("=" * 100)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 100)


if __name__ == "__main__":
    main()