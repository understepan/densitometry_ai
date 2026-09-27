from pathlib import Path

import pydicom
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent.parent

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


STUDY_NAMES = [
    "1.2.840.113619.2.110.512719.20250403090444",
    "1.2.840.113619.2.110.512719.20250403092052",
    "2.25.114887542067602662452692698814728642117",
]


def normalize_pixels(ds):
    pixels = ds.pixel_array

    min_value = pixels.min()
    max_value = pixels.max()

    if max_value == min_value:
        return pixels.astype("uint8")

    normalized = (
        (pixels - min_value)
        / (max_value - min_value)
        * 255
    )

    return normalized.astype("uint8")


def create_thumbnail(ds, width=220):
    image_array = normalize_pixels(ds)

    image = Image.fromarray(image_array)

    original_width, original_height = image.size

    scale = width / original_width
    height = int(original_height * scale)

    image = image.resize(
        (width, height)
    )

    return image


def create_contact_sheet(study_name, dicom_files):

    thumbnails = []

    for dicom_path in dicom_files:

        ds = pydicom.dcmread(dicom_path)

        thumbnail = create_thumbnail(ds)

        instance_number = getattr(
            ds,
            "InstanceNumber",
            "?"
        )

        label = (
            f"Instance {instance_number}\n"
            f"{dicom_path.name}\n"
            f"{ds.Rows}x{ds.Columns}"
        )

        thumbnails.append(
            (
                thumbnail,
                label
            )
        )

    cell_width = 260
    cell_height = 300

    columns = 4

    rows = (
        len(thumbnails) + columns - 1
    ) // columns

    sheet = Image.new(
        "RGB",
        (
            cell_width * columns,
            cell_height * rows
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, (thumbnail, label) in enumerate(
        thumbnails
    ):

        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        image_x = (
            x
            + (cell_width - thumbnail.width) // 2
        )

        sheet.paste(
            thumbnail,
            (image_x, y + 10)
        )

        draw.text(
            (x + 10, y + thumbnail.height + 20),
            label,
            fill="black"
        )

    return sheet


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("СОЗДАНИЕ CONTACT SHEETS")
    print("=" * 80)

    for study_name in STUDY_NAMES:

        study_dir = STUDIES_DIR / study_name

        if not study_dir.exists():
            print()
            print(f"Не найдена папка: {study_name}")
            continue

        dicom_files = sorted(
            study_dir.rglob("*.dcm")
        )

        if not dicom_files:
            continue

        sheet = create_contact_sheet(
            study_name,
            dicom_files
        )

        output_file = (
            OUTPUT_DIR
            / f"{study_name}.png"
        )

        sheet.save(output_file)

        print()
        print(f"Исследование: {study_name}")
        print(f"DICOM: {len(dicom_files)}")
        print(f"Сохранено: {output_file}")

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)


if __name__ == "__main__":
    main()