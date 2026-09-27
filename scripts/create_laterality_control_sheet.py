from pathlib import Path

import csv
import matplotlib.pyplot as plt
import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "visualizations"
    / "laterality_control"
)

CONTROL_STUDIES = [
    "2.25.126293418391331346694592071563510265596",
    "2.25.145848300160680194368420189079847088556",
    "2.25.164566280810634620154538016595892212884",
    "2.25.39068163399801173920284179500049527111",
    "2.25.78999454687125053757525542837745968726",
]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    groups = {}

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["study_id"] not in CONTROL_STUDIES:
                continue

            if row["anatomy"] != "hip":
                continue

            groups.setdefault(
                row["study_id"],
                []
            ).append(row)

    for study_id in CONTROL_STUDIES:

        images = groups.get(
            study_id,
            []
        )

        if len(images) != 2:
            print(
                f"ПРОПУСК: {study_id} "
                f"найдено изображений: {len(images)}"
            )
            continue

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 7)
        )

        fig.suptitle(
            f"STUDY {study_id}",
            fontsize=12
        )

        for index, (ax, row) in enumerate(
            zip(axes, images),
            start=1
        ):

            path = Path(row["dicom_path"])

            ds = pydicom.dcmread(path)

            image = ds.pixel_array

            ax.imshow(
                image,
                cmap="gray"
            )

            ax.set_title(
                f"IMAGE {index}\n"
                f"{path.name}\n"
                f"Instance={row['instance_number']}\n"
                f"{row['rows']}x{row['columns']}",
                fontsize=9
            )

            ax.axis("off")

        output_file = (
            OUTPUT_DIR
            / f"{study_id}.png"
        )

        plt.tight_layout()
        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        print(
            f"Создано: {output_file}"
        )

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)
    print(
        f"Папка: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()