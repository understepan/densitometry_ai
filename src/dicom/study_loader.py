from pathlib import Path

import pydicom


def load_study(study_dir: str):
    study_path = Path(study_dir)

    if not study_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {study_path}")

    dicom_files = sorted(study_path.rglob("*.dcm"))

    studies = {}

    for file_path in dicom_files:
        dataset = pydicom.dcmread(file_path)

        study_uid = str(dataset.StudyInstanceUID)

        if study_uid not in studies:
            studies[study_uid] = []

        studies[study_uid].append(
            {
                "file_path": str(file_path),
                "series_uid": str(dataset.SeriesInstanceUID),
                "sop_uid": str(dataset.SOPInstanceUID),
            }
        )

    return studies