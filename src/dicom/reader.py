from pathlib import Path

import numpy as np
import pydicom


def read_dicom(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    dataset = pydicom.dcmread(path)

    return dataset


def read_dicom_pixels(file_path: str) -> np.ndarray:
    dataset = read_dicom(file_path)

    pixels = dataset.pixel_array

    return pixels