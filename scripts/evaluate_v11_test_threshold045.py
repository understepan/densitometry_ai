from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_manifest.csv"
TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "spine_quality_resnet18_v11_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "spine_quality_v11_test_threshold045.csv"

THRESHOLD = 0.45


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(path_value):
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# PIXEL HASH
# ============================================================

def get_pixel_hash(dicom_path):
    path = resolve_path(dicom_path)

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array

    return hash(arr.tobytes())


# ============================================================
# V11 PREPROCESSING
# ============================================================

def crop_central_roi(arr):
    """
    Берём центральные 70% изображения по ширине.
    Высота сохраняется полностью.
    """

    height, width = arr.shape

    left = int(width * 0.15)
    right = int(width * 0.85)

    return arr[:, left:right]


def percentile_normalize(arr):
    """
    Нормализация по 1-99 перцентилям.
    """

    arr = arr.astype(np.float32)

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = np.clip(arr, p1, p99)
    arr = (arr - p1) / (p99 - p1)
    arr = (arr * 255).astype(np.uint8)

    return arr


def resize_with_padding(image):
    """
    Сохраняем пропорции и дополняем до 224x224.
    """

    image = Image.fromarray(image)

    width, height = image.size

    scale = min(224 / width, 224 / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "L",
        (224, 224),
        0
    )

    left = (224 - new_width) // 2
    top = (224 - new_height) // 2

    canvas.paste(image, (left, top))

    return canvas


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

class V11Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        features = self.backbone(x)

        logits = self.classifier(features)

        return logits


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Загрузка V11 model...")

    model = V11Model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # V11 checkpoint сохранён как словарь
    # с model_state_dict + метрики обучения.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

        print("Checkpoint format: model_state_dict")

        if "epoch" in checkpoint:
            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_accuracy" in checkpoint:
            print(
                f"Checkpoint val accuracy: "
                f"{checkpoint['val_accuracy']}"
            )

        if "val_f1" in checkpoint:
            print(
                f"Checkpoint val F1: "
                f"{checkpoint['val_f1']}"
            )

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

        print("Checkpoint format: state_dict")

    else:
        state_dict = checkpoint

        print("Checkpoint format: raw state_dict")

    # На случай DataParallel
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    state_dict = cleaned_state_dict

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing or unexpected:

        print()
        print("ОШИБКА ЗАГРУЗКИ MODEL")

        if missing:
            print("Missing keys:")
            for key in missing:
                print(" ", key)

        if unexpected:
            print()
            print("Unexpected keys:")
            for key in unexpected:
                print(" ", key)

        raise RuntimeError(
            "Архитектура checkpoint не совпадает с V11."
        )

    model.to(DEVICE)

    model.eval()

    print("Модель загружена.")

    return model


# ============================================================
# READ IMAGE
# ============================================================

def prepare_image(dicom_path):

    path = resolve_path(dicom_path)

    ds = pydicom.dcmread(path)

    arr = ds.pixel_array

    # V11 central crop
    arr = crop_central_roi(arr)

    # V11 percentile normalization
    arr = percentile_normalize(arr)

    # resize + padding
    image = resize_with_padding(arr)

    # grayscale -> 3 channels
    image = np.stack(
        [np.array(image)] * 3,
        axis=-1
    )

    image = Image.fromarray(image)

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# EXACT DEDUP
# ============================================================

def exact_dedup(df):

    print("Удаление точных дубликатов...")

    hashes = []

    for _, row in df.iterrows():

        try:
            pixel_hash = get_pixel_hash(
                row["dicom_path"]
            )

        except Exception as e:

            print(
                f"Ошибка чтения: "
                f"{row['dicom_path']}"
            )

            raise e

        hashes.append(pixel_hash)

    df = df.copy()

    df["_pixel_hash"] = hashes

    df = df.drop_duplicates(
        subset=["_pixel_hash"]
    ).copy()

    df = df.drop(
        columns=["_pixel_hash"]
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V11 FINAL TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}")

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print()
    print(
        f"Исходных строк manifest: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # TEST SPLIT
    # --------------------------------------------------------

    test_split = pd.read_csv(
        TEST_SPLIT_PATH
    )

    print(
        f"Строк в test.csv: "
        f"{len(test_split)}"
    )

    # --------------------------------------------------------
    # STUDY IDs
    # --------------------------------------------------------

    if "study_id" not in test_split.columns:

        raise RuntimeError(
            "В test.csv нет колонки study_id"
        )

    test_studies = set(
        test_split["study_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # SPINE
    # --------------------------------------------------------

    spine = manifest[
        manifest["anatomy"].astype(str).str.lower()
        == "spine"
    ].copy()

    print(
        f"Spine строк: {len(spine)}"
    )

    # --------------------------------------------------------
    # TEST ONLY
    # --------------------------------------------------------

    test_df = spine[
        spine["study_id"]
        .astype(str)
        .isin(test_studies)
    ].copy()

    print(
        f"TEST spine before dedup: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # VALID LABELS
    # --------------------------------------------------------

    test_df = test_df[
        test_df["spine_quality"].notna()
    ].copy()

    # --------------------------------------------------------
    # EXACT DEDUP
    # --------------------------------------------------------

    test_df = exact_dedup(
        test_df
    )

    print()
    print("TEST exact:")
    print(
        f"Images: {len(test_df)}"
    )

    print(
        f"Studies: "
        f"{test_df['study_id'].nunique()}"
    )

    print()
    print("Class distribution:")

    print(
        test_df["spine_quality"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print()
    print("Получение predictions...")

    y_true = []
    probabilities = []
    study_ids = []
    dicom_paths = []

    with torch.no_grad():

        for _, row in test_df.iterrows():

            tensor = prepare_image(
                row["dicom_path"]
            )

            tensor = tensor.to(DEVICE)

            logits = model(tensor)

            probs = torch.softmax(
                logits,
                dim=1
            )

            probability_class1 = (
                probs[0, 1]
                .item()
            )

            y_true.append(
                int(row["spine_quality"])
            )

            probabilities.append(
                probability_class1
            )

            study_ids.append(
                row["study_id"]
            )

            dicom_paths.append(
                row["dicom_path"]
            )

    y_true = np.array(
        y_true,
        dtype=int
    )

    probabilities = np.array(
        probabilities,
        dtype=float
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

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
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"V11 TEST @ THRESHOLD {THRESHOLD:.2f}"
    )
    print("=" * 70)

    print(
        f"Accuracy:  {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall:    {recall * 100:.2f}%"
    )

    print(
        f"F1:        {f1 * 100:.2f}%"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    # --------------------------------------------------------
    # DETAILED PREDICTIONS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST PREDICTIONS")
    print("=" * 70)

    result_df = pd.DataFrame({
        "study_id": study_ids,
        "dicom_path": dicom_paths,
        "true_label": y_true,
        "probability_class1": probabilities,
        "prediction": y_pred
    })

    result_df = result_df.sort_values(
        "probability_class1"
    ).reset_index(drop=True)

    for i, row in result_df.iterrows():

        status = (
            "OK"
            if row["true_label"]
            == row["prediction"]
            else "ERROR"
        )

        print(
            f"{i + 1:02d}. "
            f"study={row['study_id']} | "
            f"true={int(row['true_label'])} | "
            f"pred={int(row['prediction'])} | "
            f"P(class1)="
            f"{row['probability_class1']:.6f} | "
            f"{status}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"Результаты сохранены:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()