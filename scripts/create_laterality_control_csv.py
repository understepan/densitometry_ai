from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "laterality_control.csv"
)


CONTROL_ROWS = [
    {
        "study_id": "2.25.39068163399801173920284179500049527111",
        "image_order": 1,
        "laterality": "left",
        "note": "IMAGE 1: левый тазобедренный сустав",
    },
    {
        "study_id": "2.25.39068163399801173920284179500049527111",
        "image_order": 2,
        "laterality": "right",
        "note": "IMAGE 2: правый тазобедренный сустав",
    },

    {
        "study_id": "2.25.78999454687125053757525542837745968726",
        "image_order": 1,
        "laterality": "right",
        "note": "IMAGE 1: правый тазобедренный сустав",
    },
    {
        "study_id": "2.25.78999454687125053757525542837745968726",
        "image_order": 2,
        "laterality": "left",
        "note": "IMAGE 2: левый тазобедренный сустав",
    },

    {
        "study_id": "2.25.126293418391331346694592071563510265596",
        "image_order": 1,
        "laterality": "left",
        "note": "IMAGE 1: левый тазобедренный сустав",
    },
    {
        "study_id": "2.25.126293418391331346694592071563510265596",
        "image_order": 2,
        "laterality": "unknown",
        "note": (
            "IMAGE 2: правое крыло подвздошной кости и верхняя "
            "часть вертлужной впадины; полноценный тазобедренный "
            "сустав практически отсутствует в кадре"
        ),
    },

    {
        "study_id": "2.25.145848300160680194368420189079847088556",
        "image_order": 1,
        "laterality": "left",
        "note": "IMAGE 1: левый тазобедренный сустав",
    },
    {
        "study_id": "2.25.145848300160680194368420189079847088556",
        "image_order": 2,
        "laterality": "right",
        "note": "IMAGE 2: правый тазобедренный сустав",
    },

    {
        "study_id": "2.25.164566280810634620154538016595892212884",
        "image_order": 1,
        "laterality": "left",
        "note": "IMAGE 1: левый тазобедренный сустав",
    },
    {
        "study_id": "2.25.164566280810634620154538016595892212884",
        "image_order": 2,
        "laterality": "right",
        "note": "IMAGE 2: правый тазобедренный сустав",
    },
]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "study_id",
        "image_order",
        "laterality",
        "note",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(CONTROL_ROWS)

    print(f"Файл создан: {OUTPUT_FILE}")
    print(f"Записей: {len(CONTROL_ROWS)}")


if __name__ == "__main__":
    main()