from pathlib import Path
import csv
import math

import pydicom
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "laterality_ground_truth.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "visualizations"
    / "laterality_pairs"
)


def load_image(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with GROUND_TRUTH_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    studies = {}

    for row in rows:
        studies.setdefault(row["study_id"], []).append(row)

    for study_id, study_rows in studies.items():
        images = []

        for row in study_rows:
            filename = row["dicom_filename"]
            laterality = row["laterality"]

            study_dir = (
                PROJECT_DIR
                / "data"
                / "raw"
                / "training"
                / "Исследования"
                / study_id
            )

            matches = list(study_dir.rglob(filename))

            if not matches:
                print(f"Не найден: {study_id} / {filename}")
                continue

            path = matches[0]
            pixel_array = load_image(path)

            images.append(
                {
                    "filename": filename,
                    "laterality": laterality,
                    "array": pixel_array,
                }
            )

        if not images:
            continue

        fig, axes = plt.subplots(
            1,
            len(images),
            figsize=(10 * len(images), 8),
        )

        if len(images) == 1:
            axes = [axes]

        for ax, image in zip(axes, images):
            ax.imshow(
                image["array"],
                cmap="gray",
            )

            ax.set_title(
                f'{image["filename"]}\n'
                f'laterality = {image["laterality"]}',
                fontsize=12,
            )

            ax.axis("off")

        fig.suptitle(
            f"Study: {study_id}",
            fontsize=14,
        )

        output_file = OUTPUT_DIR / f"{study_id}.png"

        plt.tight_layout()
        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Создано: {output_file}")

    print()
    print("Готово.")


if __name__ == "__main__":
    main()