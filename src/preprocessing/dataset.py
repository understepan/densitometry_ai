from pathlib import Path
import csv

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class SpineQualityDataset(Dataset):
    def __init__(
            self,
            csv_file,
            project_dir,
            image_size=224,
            transform=None,
    ):
        self.csv_file = Path(csv_file)
        self.project_dir = Path(project_dir)
        self.image_size = image_size
        self.transform = transform

        # ----------------------------------------------------
        # Загружаем CSV
        # ----------------------------------------------------

        with open(
            self.csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            all_rows = list(reader)

        # ----------------------------------------------------
        # Оставляем только:
        #
        # 1. spine
        # 2. spine_quality = 0 или 1
        #
        # Пропуски исключаем.
        # ----------------------------------------------------

        self.rows = []

        for row in all_rows:

            if row["anatomy"] != "spine":
                continue

            label = row["spine_quality"]

            if label not in {"0", "1"}:
                continue

            self.rows.append(row)

        if not self.rows:

            raise ValueError(
                "После фильтрации не осталось "
                "изображений для Dataset."
            )

    # ========================================================
    # Длина Dataset
    # ========================================================

    def __len__(self):

        return len(self.rows)

    # ========================================================
    # Загрузка DICOM
    # ========================================================

    def load_image(self, dicom_path):

        full_path = (
            self.project_dir
            / dicom_path
        )

        if not full_path.exists():

            raise FileNotFoundError(
                f"Не найден DICOM: {full_path}"
            )

        dataset = pydicom.dcmread(
            full_path
        )

        image = dataset.pixel_array

        # ----------------------------------------------------
        # Переводим в float32
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Min-Max normalization
        #
        # 0 -> 0
        # max -> 1
        # ----------------------------------------------------

        min_value = image.min()
        max_value = image.max()

        if max_value > min_value:

            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Tensor
        #
        # [H, W]
        # ->
        # [1, H, W]
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        )

        image = image.unsqueeze(0)

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False,
        )

        image = image.squeeze(0)

        return image

    # ========================================================
    # Получение одного элемента
    # ========================================================

    def __getitem__(self, index):
        row = self.rows[index]

        image = self.load_image(row["dicom_path"])

        # Аугментация применяется только тогда,
        # когда transform передан в Dataset.
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            int(row["spine_quality"]),
            dtype=torch.long,
        )

        return {
            "image": image,
            "label": label,
            "study_id": row["study_id"],
            "dicom_path": row["dicom_path"],
        }