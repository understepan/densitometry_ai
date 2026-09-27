from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "laterality_ground_truth.csv"
)


GROUND_TRUTH = [
    {
        "study_id": "2.25.39068163399801173920284179500049527111",
        "dicom_filename": "CR000000.dcm",
        "laterality": "left",
        "source": "manual_visual_check",
    },
    {
        "study_id": "2.25.39068163399801173920284179500049527111",
        "dicom_filename": "CR000001.dcm",
        "laterality": "right",
        "source": "manual_visual_check",
    },

    {
        "study_id": "2.25.78999454687125053757525542837745968726",
        "dicom_filename": "CR000000.dcm",
        "laterality": "right",
        "source": "manual_visual_check",
    },
    {
        "study_id": "2.25.78999454687125053757525542837745968726",
        "dicom_filename": "CR000001.dcm",
        "laterality": "left",
        "source": "manual_visual_check",
    },

    {
        "study_id": "2.25.126293418391331346694592071563510265596",
        "dicom_filename": "CR000001.dcm",
        "laterality": "left",
        "source": "manual_visual_check",
    },
    {
        "study_id": "2.25.126293418391331346694592071563510265596",
        "dicom_filename": "CR000002.dcm",
        "laterality": "unknown",
        "source": "manual_visual_check",
    },

    {
        "study_id": "2.25.145848300160680194368420189079847088556",
        "dicom_filename": "CR000000.dcm",
        "laterality": "left",
        "source": "manual_visual_check",
    },
    {
        "study_id": "2.25.145848300160680194368420189079847088556",
        "dicom_filename": "CR000001.dcm",
        "laterality": "right",
        "source": "manual_visual_check",
    },

    {
        "study_id": "2.25.164566280810634620154538016595892212884",
        "dicom_filename": "CR000000.dcm",
        "laterality": "left",
        "source": "manual_visual_check",
    },
    {
        "study_id": "2.25.164566280810634620154538016595892212884",
        "dicom_filename": "CR000002.dcm",
        "laterality": "right",
        "source": "manual_visual_check",
    },
]


def main():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "study_id",
        "dicom_filename",
        "laterality",
        "source",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(GROUND_TRUTH)

    print(f"Файл создан: {OUTPUT_FILE}")
    print(f"Записей: {len(GROUND_TRUTH)}")


if __name__ == "__main__":
    main()