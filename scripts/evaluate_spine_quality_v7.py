from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_manifest.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "spine_quality_resnet18_v7_best.pth"
OUTPUT_PATH = PROJECT_ROOT / "results" / "spine_quality_v7_exact_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224


# ============================================================
# MODEL
# ============================================================

class SpineQualityResNet18V7(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None)

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device
        ).view(1, 3, 1, 1)

        x = (x - mean) / std

        return self.model(x)


# ============================================================
# PREPROCESSING
# EXACTLY AS IN V7
# ============================================================

def percentile_normalize(image):

    image = image.astype(np.float32)

    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 <= p1:
        image = image - image.min()

        max_value = image.max()

        if max_value > 0:
            image = image / max_value

        return image

    image = np.clip(image, p1, p99)

    image = (image - p1) / (p99 - p1)

    return image


def resize_with_padding(image, target_size=224):

    h, w = image.shape

    scale = min(
        target_size / h,
        target_size / w
    )

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil = Image.fromarray(
        (image * 255).astype(np.uint8)
    )

    pil = pil.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR
    )

    resized = np.asarray(
        pil,
        dtype=np.float32
    ) / 255.0

    canvas = np.zeros(
        (target_size, target_size),
        dtype=np.float32
    )

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2

    canvas[
        top:top + new_h,
        left:left + new_w
    ] = resized

    return canvas


def preprocess_image(image):

    image = percentile_normalize(image)

    image = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    image = torch.from_numpy(
        image
    ).float()

    image = image.unsqueeze(0)

    return image


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = get_dicom_path(row)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        import pydicom

        ds = pydicom.dcmread(str(path))

        image = ds.pixel_array

        image = preprocess_image(image)

        label = int(row["label_quality"])

        return {
            "image": image,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "study_id": str(row["study_id"]),
            "file_path": str(path),
        }


# ============================================================
# REMOVE EXACT DUPLICATES
# BY PIXEL DATA
# ============================================================

def pixel_hash(file_path):

    import pydicom

    ds = pydicom.dcmread(str(file_path))

    pixel_data = ds.PixelData

    return hashlib.md5(pixel_data).hexdigest()


def get_dicom_path(row):

    # В manifest путь обычно хранится в одной из этих колонок.
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

    # Если отдельного пути нет,
    # собираем его из study_id + имени файла.
    study_id = str(row["study_id"])

    possible_name_columns = [
        "instance",
        "instance_number",
        "sop_instance",
        "sop_instance_uid",
        "name",
        "filename",
        "file_name",
    ]

    for column in possible_name_columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value):

                value = str(value)

                study_dir = (
                    PROJECT_ROOT
                    / "data"
                    / "raw"
                    / "training"
                    / "Исследования"
                    / study_id
                )

                candidate = study_dir / value

                if candidate.exists():
                    return candidate

    raise FileNotFoundError(
        "Не удалось определить путь к DICOM.\n"
        f"study_id={study_id}\n"
        f"Колонки: {list(row.index)}"
    )


def remove_exact_duplicates(df):

    print()
    print("Проверка точных дубликатов PixelData...")

    hashes = []

    for _, row in df.iterrows():

        path = get_dicom_path(row)

        hashes.append(
            pixel_hash(path)
        )

    df = df.copy()

    df["pixel_hash"] = hashes

    before = len(df)

    df = df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    ).reset_index(drop=True)

    after = len(df)

    print(f"До удаления: {before}")
    print(f"После удаления: {after}")
    print(f"Удалено дубликатов: {before - after}")

    return df


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_dataframe():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_ids = pd.read_csv(
        SPLIT_PATH
    )

    # В test.csv может быть как study_id,
    # так и полный набор колонок.
    if "study_id" in test_ids.columns:

        study_ids = set(
            test_ids["study_id"]
            .astype(str)
        )

    else:

        study_ids = set(
            test_ids.iloc[:, 0]
            .astype(str)
        )

    df = manifest.copy()

    df["study_id"] = (
        df["study_id"]
        .astype(str)
    )

    # Только TEST studies
    df = df[
        df["study_id"].isin(study_ids)
    ].copy()

    # Только позвоночник
    df = df[
        df["anatomy"] == "spine"
    ].copy()

    # Только известная разметка
    df = df[
        df["label_quality"].isin([0, 1])
    ].copy()

    print()
    print("TEST после фильтрации spine:")
    print(f"Строк: {len(df)}")
    print(
        "Class 0:",
        int((df["label_quality"] == 0).sum())
    )
    print(
        "Class 1:",
        int((df["label_quality"] == 1).sum())
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EVALUATION V7 — EXACT UNIQUE TEST")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_test_dataframe()

    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    df = remove_exact_duplicates(df)

    print()
    print("Финальный TEST:")
    print(f"Уникальных изображений: {len(df)}")

    # --------------------------------------------------------
    # CHECK STUDIES
    # --------------------------------------------------------

    print(
        f"Уникальных исследований: "
        f"{df['study_id'].nunique()}"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = TestDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Загрузка модели:")
    print(MODEL_PATH)

    model = SpineQualityResNet18V7()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Поддерживаем несколько вариантов
    # сохранения checkpoint.

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    print("Модель успешно загружена.")

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    y_true = []
    y_pred = []
    probabilities = []

    studies = []
    files = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                DEVICE
            )

            labels = batch["label"].numpy()

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            ).cpu().numpy()

            p1 = probs[:, 1].cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            probabilities.extend(p1.tolist())

            studies.extend(
                batch["study_id"]
            )

            files.extend(
                batch["file_path"]
            )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS V7")
    print("=" * 70)

    print()
    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Class 1 precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Class 1 recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Class 1 F1: "
        f"{f1 * 100:.2f}%"
    )

    print()
    print("Confusion matrix:")
    print()
    print("                 Pred 0    Pred 1")
    print(
        f"True 0          "
        f"{cm[0, 0]:8d}  "
        f"{cm[0, 1]:8d}"
    )
    print(
        f"True 1          "
        f"{cm[1, 0]:8d}  "
        f"{cm[1, 1]:8d}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print("Classification report:")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Class 0",
                "Class 1"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # DETAILED PREDICTIONS
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "study_id": studies,
        "file_path": files,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability_class1": probabilities
    })

    result_df["correct"] = (
        result_df["true_label"]
        ==
        result_df["predicted_label"]
    )

    result_df = result_df.sort_values(
        ["study_id", "file_path"]
    ).reset_index(drop=True)

    print()
    print("=" * 70)
    print("DETAILED PREDICTIONS")
    print("=" * 70)

    for _, row in result_df.iterrows():

        status = (
            "OK"
            if row["correct"]
            else "ERROR"
        )

        print(
            f"{status:5s} | "
            f"study={row['study_id']} | "
            f"true={int(row['true_label'])} | "
            f"pred={int(row['predicted_label'])} | "
            f"P(class1)={row['probability_class1']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("Файл сохранён:")
    print(OUTPUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()