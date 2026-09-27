from pathlib import Path
import math

import numpy as np
import pandas as pd
import pydicom

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "visualizations"
    / "spine_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 220

COLUMNS = 5

MAX_IMAGES_PER_CLASS = 30


# ============================================================
# LOAD DICOM
# ============================================================

def load_dicom_image(path):

    ds = pydicom.dcmread(
        str(path),
        force=True
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    # Percentile normalization
    p1 = np.percentile(
        image,
        1
    )

    p99 = np.percentile(
        image,
        99
    )

    if p99 > p1:

        image = (
            image - p1
        ) / (
            p99 - p1
        )

    else:

        image = np.zeros_like(
            image
        )

    image = np.clip(
        image,
        0,
        1
    )

    image = (
        image * 255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        image,
        mode="L"
    )


# ============================================================
# CREATE TILE
# ============================================================

def make_tile(
    image,
    study_id,
    label
):

    # сохраняем пропорции
    image.thumbnail(
        (
            TILE_SIZE - 10,
            TILE_SIZE - 35
        )
    )

    canvas = Image.new(
        "L",
        (
            TILE_SIZE,
            TILE_SIZE
        ),
        0
    )

    x = (
        TILE_SIZE - image.width
    ) // 2

    y = 5

    canvas.paste(
        image,
        (
            x,
            y
        )
    )

    # переводим в RGB для текста
    canvas = canvas.convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        canvas
    )

    text = (
        f"class={label}  "
        f"study={study_id[-12:]}"
    )

    draw.rectangle(
        (
            0,
            TILE_SIZE - 30,
            TILE_SIZE,
            TILE_SIZE
        ),
        fill="black"
    )

    draw.text(
        (
            5,
            TILE_SIZE - 25
        ),
        text,
        fill="white"
    )

    return canvas


# ============================================================
# CONTACT SHEET
# ============================================================

def make_contact_sheet(
    dataframe,
    label,
    output_path
):

    dataframe = dataframe[
        dataframe["spine_quality"] == label
    ].copy()

    dataframe = dataframe.head(
        MAX_IMAGES_PER_CLASS
    )

    tiles = []

    for _, row in dataframe.iterrows():

        path = (
            PROJECT_ROOT
            / row["dicom_path"]
        )

        try:

            image = load_dicom_image(
                path
            )

            tile = make_tile(
                image,
                str(row["study_id"]),
                label
            )

            tiles.append(
                tile
            )

        except Exception as e:

            print(
                "Ошибка:",
                path,
                e
            )

    if not tiles:

        print(
            "Нет изображений для класса",
            label
        )

        return

    rows = math.ceil(
        len(tiles) / COLUMNS
    )

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * TILE_SIZE,
            rows * TILE_SIZE
        ),
        "gray"
    )

    for index, tile in enumerate(
        tiles
    ):

        x = (
            index % COLUMNS
        ) * TILE_SIZE

        y = (
            index // COLUMNS
        ) * TILE_SIZE

        sheet.paste(
            tile,
            (
                x,
                y
            )
        )

    sheet.save(
        output_path
    )

    print(
        "Создано:",
        output_path
    )

    print(
        "Изображений:",
        len(tiles)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "SPINE CONTACT SHEET"
    )

    print(
        "=" * 70
    )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    # только позвоночник
    df = df[
        df["anatomy"] == "spine"
    ].copy()

    # итоговая метка
    df["spine_quality"] = pd.to_numeric(
        df["spine_quality"],
        errors="coerce"
    )

    df = df[
        df["spine_quality"].isin(
            [0, 1]
        )
    ].copy()

    # --------------------------------------------------------
    # Используем только TRAIN
    # --------------------------------------------------------

    train_split = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "splits"
        / "train.csv"
    )

    train_ids = set(
        train_split[
            "study_id"
        ].astype(str)
    )

    df["study_id"] = df[
        "study_id"
    ].astype(str)

    df = df[
        df["study_id"].isin(
            train_ids
        )
    ].copy()

    print(
        "TRAIN spine:",
        len(df)
    )

    # --------------------------------------------------------
    # Убираем точные PixelData-дубликаты
    # --------------------------------------------------------

    hashes = []

    print(
        "Проверка PixelData..."
    )

    for _, row in df.iterrows():

        path = (
            PROJECT_ROOT
            / row["dicom_path"]
        )

        ds = pydicom.dcmread(
            str(path),
            force=True
        )

        hashes.append(
            hash(ds.PixelData)
        )

    df["pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=[
            "pixel_hash"
        ],
        keep="first"
    ).copy()

    print(
        "Уникальных TRAIN:",
        len(df)
    )

    print()

    print(
        "Класс 0:",
        (
            df["spine_quality"] == 0
        ).sum()
    )

    print(
        "Класс 1:",
        (
            df["spine_quality"] == 1
        ).sum()
    )

    # --------------------------------------------------------
    # Создаём контактные листы
    # --------------------------------------------------------

    make_contact_sheet(
        df,
        0,
        OUTPUT_DIR
        / "train_class_0.png"
    )

    make_contact_sheet(
        df,
        1,
        OUTPUT_DIR
        / "train_class_1.png"
    )

    print()

    print(
        "Готово."
    )

    print(
        "Папка:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()