from pathlib import Path
import math

import numpy as np
import pydicom
from PIL import Image, ImageOps, ImageDraw, ImageFont


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "visualizations"
    / "contact_sheets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "v3_six_class1_studies.png"
)


# ============================================================
# 6 ИССЛЕДОВАНИЙ
# ============================================================

STUDIES = [
    {
        "id": "2.25.102755089973625799055786462646268820450",
        "true": 1,
        "pred": 0,
        "status": "ERROR"
    },
    {
        "id": "2.25.114887542067602662452692698814728642117",
        "true": 1,
        "pred": 0,
        "status": "ERROR"
    },
    {
        "id": "2.25.269648774413746842548908998641183395622",
        "true": 1,
        "pred": 0,
        "status": "ERROR"
    },
    {
        "id": "2.25.338439598607961312040815816824093323492",
        "true": 1,
        "pred": 0,
        "status": "ERROR"
    },
    {
        "id": "2.25.29638026465755240167985424461715024381",
        "true": 1,
        "pred": 1,
        "status": "CORRECT"
    },
    {
        "id": "2.25.48906424954135283790354958243772558962",
        "true": 1,
        "pred": 1,
        "status": "CORRECT"
    },
]


# ============================================================
# РАЗМЕРЫ
# ============================================================

CELL_WIDTH = 360
CELL_HEIGHT = 330

IMAGE_SIZE = 280

HEADER_HEIGHT = 70

BACKGROUND = 30


# ============================================================
# ПОИСК DICOM
# ============================================================

def find_dicoms(study_id):

    study_dir = STUDIES_DIR / study_id

    if not study_dir.exists():
        print()
        print("ПАПКА НЕ НАЙДЕНА:")
        print(study_dir)
        return []

    dicom_files = []

    for path in study_dir.rglob("*"):

        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            if hasattr(ds, "SOPInstanceUID"):
                dicom_files.append(path)

        except Exception:
            continue

    return sorted(dicom_files)


# ============================================================
# DICOM → PIL
# ============================================================

def dicom_to_pil(path):

    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(
        np.float32
    )

    # Нормализация
    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:

        image = (
            (image - minimum)
            / (maximum - minimum)
            * 255.0
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32
        )

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    pil = Image.fromarray(
        image,
        mode="L"
    )

    # Подгоняем без искажения пропорций
    pil.thumbnail(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        BACKGROUND
    )

    x = (
        IMAGE_SIZE - pil.width
    ) // 2

    y = (
        IMAGE_SIZE - pil.height
    ) // 2

    canvas.paste(
        pil,
        (x, y)
    )

    return canvas.convert("RGB")


# ============================================================
# ШРИФТ
# ============================================================

def get_font(size):

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]

    for font_path in candidates:

        path = Path(font_path)

        if path.exists():

            try:
                return ImageFont.truetype(
                    str(path),
                    size
                )
            except Exception:
                pass

    return ImageFont.load_default()


FONT_TITLE = get_font(22)
FONT_TEXT = get_font(17)
FONT_SMALL = get_font(14)


# ============================================================
# СОЗДАНИЕ КОНТАКТ-ЛИСТА
# ============================================================

def create_contact_sheet():

    study_images = []

    print("=" * 70)
    print("V3 — CONTACT SHEET FOR 6 CLASS-1 STUDIES")
    print("=" * 70)

    for study in STUDIES:

        study_id = study["id"]

        print()
        print(
            f"Study: {study_id}"
        )

        dicoms = find_dicoms(
            study_id
        )

        print(
            f"DICOM: {len(dicoms)}"
        )

        images = []

        for dicom_path in dicoms:

            try:

                image = dicom_to_pil(
                    dicom_path
                )

                images.append(
                    (
                        dicom_path,
                        image
                    )
                )

            except Exception as error:

                print(
                    f"Ошибка: "
                    f"{dicom_path}"
                )

                print(error)

        study_images.append(
            (
                study,
                images
            )
        )

    # --------------------------------------------------------
    # Сколько ячеек
    # --------------------------------------------------------

    total_cells = sum(
        max(1, len(images))
        for _, images in study_images
    )

    columns = 3

    rows = math.ceil(
        total_cells / columns
    )

    width = (
        columns * CELL_WIDTH
    )

    height = (
        rows * CELL_HEIGHT
    )

    sheet = Image.new(
        "RGB",
        (width, height),
        (BACKGROUND, BACKGROUND, BACKGROUND)
    )

    draw = ImageDraw.Draw(sheet)

    cell_index = 0

    # --------------------------------------------------------
    # Рисуем исследования
    # --------------------------------------------------------

    for study, images in study_images:

        study_id = study["id"]

        short_id = (
            study_id[-18:]
        )

        header_text = (
            f"TRUE={study['true']}  "
            f"PRED={study['pred']}  "
            f"{study['status']}"
        )

        for image_number, (
            dicom_path,
            image
        ) in enumerate(images):

            column = (
                cell_index % columns
            )

            row = (
                cell_index // columns
            )

            x0 = (
                column * CELL_WIDTH
            )

            y0 = (
                row * CELL_HEIGHT
            )

            # Рамка
            draw.rectangle(
                [
                    x0,
                    y0,
                    x0 + CELL_WIDTH - 1,
                    y0 + CELL_HEIGHT - 1
                ],
                outline=(120, 120, 120),
                width=2
            )

            # Заголовок
            draw.text(
                (
                    x0 + 10,
                    y0 + 8
                ),
                header_text,
                fill=(255, 255, 255),
                font=FONT_TITLE
            )

            # ID
            draw.text(
                (
                    x0 + 10,
                    y0 + 35
                ),
                f"...{short_id}",
                fill=(190, 190, 190),
                font=FONT_SMALL
            )

            # Изображение
            image_x = (
                x0
                + (CELL_WIDTH - IMAGE_SIZE) // 2
            )

            image_y = (
                y0
                + HEADER_HEIGHT
            )

            sheet.paste(
                image,
                (
                    image_x,
                    image_y
                )
            )

            # Номер изображения
            draw.text(
                (
                    x0 + 10,
                    y0 + CELL_HEIGHT - 24
                ),
                f"Image {image_number + 1}",
                fill=(220, 220, 220),
                font=FONT_SMALL
            )

            # Имя DICOM
            draw.text(
                (
                    x0 + 100,
                    y0 + CELL_HEIGHT - 24
                ),
                dicom_path.name[:28],
                fill=(160, 160, 160),
                font=FONT_SMALL
            )

            cell_index += 1

    sheet.save(
        OUTPUT_FILE,
        quality=95
    )

    print()
    print("=" * 70)
    print("CONTACT SHEET CREATED")
    print("=" * 70)

    print()
    print(
        f"Файл:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_contact_sheet()