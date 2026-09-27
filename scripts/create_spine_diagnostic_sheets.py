from pathlib import Path
import math

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

SPLITS_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "visualizations"
    / "spine_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def get_dicom_path(row):

    possible_columns = [
        "path",
        "dicom_path",
        "file",
        "filename",
        "dicom_file",
        "file_name",
    ]

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                path = Path(str(value))

                if not path.is_absolute():
                    path = PROJECT_ROOT / path

                if path.exists():
                    return path

    study_id = str(row["study_id"])

    study_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "training"
        / "Исследования"
        / study_id
    )

    # Ищем DICOM внутри папки исследования.
    dicom_files = list(study_dir.rglob("*"))

    dicom_files = [
        p for p in dicom_files
        if p.is_file()
    ]

    # Если в исследовании несколько изображений,
    # ищем по имени, если оно есть.
    possible_name_columns = [
        "instance",
        "instance_number",
        "name",
        "filename",
        "file_name",
    ]

    for column in possible_name_columns:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                value = str(value)

                for p in dicom_files:

                    if p.name == value:
                        return p

    # Последний вариант:
    # сравниваем размеры изображения.
    if len(dicom_files) == 1:
        return dicom_files[0]

    raise FileNotFoundError(
        f"Не удалось определить DICOM для study={study_id}"
    )


def normalize_image(image):

    image = image.astype(np.float32)

    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 > p1:

        image = np.clip(
            image,
            p1,
            p99
        )

        image = (
            image - p1
        ) / (
            p99 - p1
        )

    else:

        image -= image.min()

        if image.max() > 0:
            image /= image.max()

    return image


def load_image(path):

    ds = pydicom.dcmread(
        str(path)
    )

    image = ds.pixel_array

    return normalize_image(image)


def create_sheet(
    df,
    title,
    output_name,
    max_images=100
):

    df = df.head(max_images).copy()

    if len(df) == 0:
        print(
            f"Нет изображений для: {title}"
        )
        return

    columns = 5

    rows = math.ceil(
        len(df) / columns
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 4)
    )

    axes = np.array(axes).reshape(
        -1
    )

    for ax in axes:
        ax.axis("off")

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        try:

            path = get_dicom_path(row)

            image = load_image(path)

            axes[i].imshow(
                image,
                cmap="gray"
            )

            true_label = int(
                row["label_quality"]
            )

            study_id = str(
                row["study_id"]
            )

            short_id = study_id[-8:]

            axes[i].set_title(
                f"study ...{short_id}\n"
                f"TRUE={true_label}",
                fontsize=9
            )

        except Exception as e:

            axes[i].text(
                0.5,
                0.5,
                f"ERROR\n{e}",
                ha="center",
                va="center"
            )

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Создано: {output_path}"
    )


def main():

    print("=" * 70)
    print("SPINE DATASET DIAGNOSTIC")
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest["study_id"] = (
        manifest["study_id"]
        .astype(str)
    )

    # Только позвоночник
    df = manifest[
        manifest["anatomy"] == "spine"
    ].copy()

    # Только известная разметка
    df = df[
        df["label_quality"].isin(
            [0, 1]
        )
    ].copy()

    print()
    print(
        f"Всего spine строк: {len(df)}"
    )

    print(
        f"Class 0: "
        f"{(df['label_quality'] == 0).sum()}"
    )

    print(
        f"Class 1: "
        f"{(df['label_quality'] == 1).sum()}"
    )

    # --------------------------------------------------------
    # Убираем точные PixelData-дубликаты
    # --------------------------------------------------------

    print()
    print(
        "Проверка уникальности изображений..."
    )

    hashes = []

    import hashlib

    for _, row in df.iterrows():

        path = get_dicom_path(row)

        ds = pydicom.dcmread(
            str(path)
        )

        hashes.append(
            hashlib.md5(
                ds.PixelData
            ).hexdigest()
        )

    df["pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    ).copy()

    print(
        f"Уникальных изображений: {len(df)}"
    )

    # --------------------------------------------------------
    # READ SPLITS
    # --------------------------------------------------------

    split_files = {
        "TRAIN": "train.csv",
        "VALIDATION": "validation.csv",
        "TEST": "test.csv",
    }

    for split_name, filename in split_files.items():

        split_path = (
            SPLITS_DIR
            / filename
        )

        split_df = pd.read_csv(
            split_path
        )

        split_ids = set(
            split_df["study_id"]
            .astype(str)
        )

        part = df[
            df["study_id"].isin(
                split_ids
            )
        ].copy()

        print()
        print(
            f"{split_name}: "
            f"{len(part)} изображений"
        )

        print(
            f"  class 0: "
            f"{(part['label_quality'] == 0).sum()}"
        )

        print(
            f"  class 1: "
            f"{(part['label_quality'] == 1).sum()}"
        )

        # ----------------------------------------------------
        # Entire split
        # ----------------------------------------------------

        create_sheet(
            part,
            f"{split_name} — ALL SPINE",
            f"{split_name.lower()}_all.png"
        )

        # ----------------------------------------------------
        # Class 0
        # ----------------------------------------------------

        class0 = part[
            part["label_quality"] == 0
        ]

        create_sheet(
            class0,
            f"{split_name} — CLASS 0",
            f"{split_name.lower()}_class0.png"
        )

        # ----------------------------------------------------
        # Class 1
        # ----------------------------------------------------

        class1 = part[
            part["label_quality"] == 1
        ]

        create_sheet(
            class1,
            f"{split_name} — CLASS 1",
            f"{split_name.lower()}_class1.png"
        )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)

    print()
    print(
        "Папка с изображениями:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()