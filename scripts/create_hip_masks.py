from pathlib import Path
import csv

import numpy as np
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
    / "hip_masks"
)


def normalize_image(image):
    image = image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value == min_value:
        return np.zeros_like(image)

    image = (
        (image - min_value)
        / (max_value - min_value)
    )

    return image


def create_mask(image):
    image = normalize_image(image)

    # Пока используем простой диагностический порог.
    threshold = np.percentile(image, 75)

    mask = image >= threshold

    return mask


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with GROUND_TRUTH_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        laterality = row["laterality"]

        if laterality == "unknown":
            continue

        study_id = row["study_id"]
        filename = row["dicom_filename"]

        study_dir = (
            PROJECT_DIR
            / "data"
            / "raw"
            / "training"
            / "Исследования"
            / study_id
        )

        matches = list(
            study_dir.rglob(filename)
        )

        if not matches:
            print(
                f"Не найден файл: "
                f"{study_id} / {filename}"
            )
            continue

        dicom_path = matches[0]

        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array

        mask = create_mask(image)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
        )

        axes[0].imshow(
            image,
            cmap="gray",
        )

        axes[0].set_title(
            f"{filename}\n"
            f"laterality = {laterality}"
        )

        axes[0].axis("off")

        axes[1].imshow(
            mask,
            cmap="gray",
        )

        axes[1].set_title(
            "Бинарная маска"
        )

        axes[1].axis("off")

        fig.suptitle(
            f"Study: {study_id}",
            fontsize=11,
        )

        output_file = (
            OUTPUT_DIR
            / f"{study_id}_{filename[:-4]}.png"
        )

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=120,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Создано: {output_file}"
        )

    print()
    print("Готово.")


if __name__ == "__main__":
    main()