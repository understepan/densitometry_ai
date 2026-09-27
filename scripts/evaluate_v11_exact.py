from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "processed" / "labeled_manifest.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "spine_quality_resnet18_v11_best.pth"
RESULT_PATH = PROJECT_ROOT / "results" / "spine_quality_v11_exact_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224


# ============================================================
# V11 PREPROCESSING
# ============================================================

def crop_central_roi(image):
    """
    V11:
    оставляем центральные 70% ширины изображения,
    высоту сохраняем полностью.

    Это технический эксперимент по уменьшению
    влияния краёв/фона.
    """

    width, height = image.size

    left = int(width * 0.15)
    right = int(width * 0.85)

    if right <= left:
        return image

    return image.crop((left, 0, right, height))


def percentile_normalize(array):
    """
    Нормализация по 1-99 перцентилям.
    """

    array = array.astype(np.float32)

    p1 = np.percentile(array, 1)
    p99 = np.percentile(array, 99)

    if p99 <= p1:
        return np.zeros_like(array, dtype=np.float32)

    array = (array - p1) / (p99 - p1)
    array = np.clip(array, 0.0, 1.0)

    return array


def resize_with_padding(image, size=224):
    """
    Сохраняем aspect ratio и дополняем изображение
    до квадрата.
    """

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Некорректный размер изображения")

    scale = min(size / width, size / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image = image.resize(
        (new_width, new_height),
        Image.BILINEAR
    )

    canvas = Image.new(
        "L",
        (size, size),
        0
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    canvas.paste(image, (left, top))

    return canvas


def preprocess_dicom(dicom_path):
    """
    Полный preprocessing V11.
    """

    import pydicom

    ds = pydicom.dcmread(dicom_path)

    array = ds.pixel_array

    # float32
    array = array.astype(np.float32)

    # 1-99 percentile normalization
    array = percentile_normalize(array)

    # 0-255
    array = (array * 255.0).astype(np.uint8)

    image = Image.fromarray(array, mode="L")

    # V11 central ROI
    image = crop_central_roi(image)

    # resize + padding
    image = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    # [0,1]
    array = np.asarray(image).astype(np.float32) / 255.0

    # H,W -> 1,H,W
    tensor = torch.from_numpy(array).unsqueeze(0)

    # 1 channel -> 3 channels
    tensor = tensor.repeat(3, 1, 1)

    # ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor


# ============================================================
# EXACT PIXEL DEDUPLICATION
# ============================================================

def pixel_hash(dicom_path):
    """
    Создаёт hash PixelData.

    Используется для удаления точных дубликатов,
    как при предыдущей оценке V9.
    """

    import pydicom
    import hashlib

    ds = pydicom.dcmread(
        dicom_path,
        stop_before_pixels=False
    )

    pixel_bytes = ds.PixelData

    return hashlib.md5(pixel_bytes).hexdigest()


# ============================================================
# DATASET
# ============================================================

class SpineTestDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        path = PROJECT_ROOT / row["dicom_path"]

        image = preprocess_dicom(path)

        label = int(row["spine_quality"])

        return (
            image,
            label,
            str(row["study_id"]),
            str(row["dicom_path"]),
        )


# ============================================================
# MODEL
# ============================================================

def build_v11_model():

    weights = ResNet18_Weights.DEFAULT

    backbone = models.resnet18(
        weights=weights
    )

    num_features = backbone.fc.in_features

    # Убираем стандартный classifier ResNet18
    backbone.fc = nn.Identity()

    # Именно такая структура соответствует
    # сохранённому V11 checkpoint:
    #
    # backbone.*
    # classifier.0.*
    # classifier.3.*

    classifier = nn.Sequential(
        nn.Linear(
            num_features,
            128
        ),

        nn.ReLU(),

        nn.Dropout(
            0.30
        ),

        nn.Linear(
            128,
            2
        )
    )

    class V11Model(nn.Module):

        def __init__(
            self,
            backbone,
            classifier
        ):
            super().__init__()

            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):

            features = self.backbone(x)

            logits = self.classifier(
                features
            )

            return logits

    model = V11Model(
        backbone=backbone,
        classifier=classifier
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("LOADING V11 MODEL")
    print("=" * 60)

    print("Model:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена:\n{MODEL_PATH}"
        )

    model = build_v11_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Поддерживаем несколько вариантов сохранения checkpoint
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Иногда checkpoint сам является state_dict
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Если ключи имеют префикс module.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


# ============================================================
# BUILD EXACT TEST SET
# ============================================================

def prepare_exact_test():

    print()
    print("=" * 60)
    print("PREPARING EXACT TEST")
    print("=" * 60)

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Не найден test.csv:\n{TEST_CSV}"
        )

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Не найден labeled_manifest.csv:\n{MANIFEST_CSV}"
        )

    test_df = pd.read_csv(TEST_CSV)

    manifest = pd.read_csv(
        MANIFEST_CSV
    )

    print(
        "Строк в test.csv:",
        len(test_df)
    )

    # --------------------------------------------------------
    # Определяем study_id
    # --------------------------------------------------------

    if "study_id" not in test_df.columns:
        raise ValueError(
            "В test.csv нет столбца study_id"
        )

    # --------------------------------------------------------
    # Берём spine из полного manifest
    # --------------------------------------------------------

    spine = manifest[
        manifest["anatomy"].astype(str).str.lower() == "spine"
    ].copy()

    print(
        "Spine строк в manifest:",
        len(spine)
    )

    # --------------------------------------------------------
    # Оставляем только TEST studies
    # --------------------------------------------------------

    test_df = spine[
        spine["study_id"].isin(
            test_df["study_id"]
        )
    ].copy()

    print(
        "TEST spine до dedup:",
        len(test_df)
    )

    # --------------------------------------------------------
    # Только валидные labels
    # --------------------------------------------------------

    test_df = test_df[
        test_df["spine_quality"].notna()
    ].copy()

    test_df["spine_quality"] = (
        test_df["spine_quality"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Проверяем существование файлов
    # --------------------------------------------------------

    existing_rows = []

    for _, row in test_df.iterrows():

        path = PROJECT_ROOT / row["dicom_path"]

        if path.exists():
            existing_rows.append(row)

    test_df = pd.DataFrame(
        existing_rows
    )

    print(
        "TEST после проверки файлов:",
        len(test_df)
    )

    # --------------------------------------------------------
    # EXACT PIXEL DEDUP
    # --------------------------------------------------------

    print()
    print("Удаляем точные PixelData-дубликаты...")

    hashes = []

    for _, row in test_df.iterrows():

        path = PROJECT_ROOT / row["dicom_path"]

        try:
            h = pixel_hash(path)

        except Exception as e:

            print(
                "Ошибка PixelData:",
                path,
                e
            )

            h = None

        hashes.append(h)

    test_df["pixel_hash"] = hashes

    before = len(test_df)

    # Сохраняем первое вхождение каждого PixelData
    test_df = test_df.drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    ).copy()

    after = len(test_df)

    print(
        "До dedup:",
        before
    )

    print(
        "После dedup:",
        after
    )

    print(
        "Удалено:",
        before - after
    )

    # --------------------------------------------------------
    # Финальные проверки
    # --------------------------------------------------------

    if test_df["study_id"].duplicated().any():

        duplicated_studies = (
            test_df[
                test_df["study_id"].duplicated(
                    keep=False
                )
            ]["study_id"]
            .unique()
        )

        print()
        print(
            "ВНИМАНИЕ: несколько изображений "
            "из одного study:"
        )

        for study in duplicated_studies:
            print(study)

    test_df = test_df.sort_values(
        "study_id"
    ).reset_index(drop=True)

    print()
    print(
        "ИТОГОВЫЙ EXACT TEST:",
        len(test_df)
    )

    print(
        "Уникальных studies:",
        test_df["study_id"].nunique()
    )

    print()
    print("Class distribution:")

    print(
        test_df["spine_quality"]
        .value_counts()
        .sort_index()
    )

    if len(test_df) != 15:
        print()
        print(
            "ПРЕДУПРЕЖДЕНИЕ:"
        )

        print(
            "Ожидалось 15 уникальных TEST "
            "изображений, но получено:",
            len(test_df)
        )

    return test_df


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataframe):

    dataset = SpineTestDataset(
        dataframe
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    y_pred = []
    probabilities = []
    studies = []
    paths = []

    print()
    print("=" * 60)
    print("RUNNING V11 TEST")
    print("=" * 60)

    for images, labels, study_ids, dicom_paths in loader:

        images = images.to(
            DEVICE
        )

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        probability_class1 = (
            probs[:, 1]
            .item()
        )

        prediction = (
            torch.argmax(
                probs,
                dim=1
            )
            .item()
        )

        true_label = labels.item()

        y_true.append(
            true_label
        )

        y_pred.append(
            prediction
        )

        probabilities.append(
            probability_class1
        )

        studies.append(
            study_ids[0]
        )

        paths.append(
            dicom_paths[0]
        )

    return (
        y_true,
        y_pred,
        probabilities,
        studies,
        paths,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    dataframe,
    y_true,
    y_pred,
    probabilities,
    studies,
    paths,
):

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

    print()
    print("=" * 60)
    print("V11 EXACT TEST RESULTS")
    print("=" * 60)

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
    print()

    print(
        "                 Predicted"
    )

    print(
        "               0        1"
    )

    print(
        f"True 0      {cm[0,0]:3d}      {cm[0,1]:3d}"
    )

    print(
        f"True 1      {cm[1,0]:3d}      {cm[1,1]:3d}"
    )

    # --------------------------------------------------------
    # Detailed table
    # --------------------------------------------------------

    results = pd.DataFrame({
        "study_id": studies,
        "dicom_path": paths,
        "spine_quality": y_true,
        "prediction": y_pred,
        "probability_class1": probabilities,
    })

    results["correct"] = (
        results["spine_quality"]
        ==
        results["prediction"]
    )

    results = results.sort_values(
        "study_id"
    ).reset_index(drop=True)

    print()
    print("=" * 60)
    print("ALL TEST PREDICTIONS")
    print("=" * 60)

    for i, row in results.iterrows():

        true_label = int(
            row["spine_quality"]
        )

        prediction = int(
            row["prediction"]
        )

        probability = float(
            row["probability_class1"]
        )

        status = (
            "OK"
            if true_label == prediction
            else "ERROR"
        )

        print()
        print(
            f"{i + 1:02d}. {status}"
        )

        print(
            "Study:",
            row["study_id"]
        )

        print(
            "TRUE:",
            true_label
        )

        print(
            "PRED:",
            prediction
        )

        print(
            f"P(class1): {probability:.6f}"
        )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    errors = results[
        ~results["correct"]
    ].copy()

    print()
    print("=" * 60)
    print(
        f"ERRORS: {len(errors)}"
    )
    print("=" * 60)

    if len(errors) == 0:

        print(
            "Ошибок нет."
        )

    else:

        for _, row in errors.iterrows():

            print()
            print(
                "Study:",
                row["study_id"]
            )

            print(
                "TRUE:",
                int(row["spine_quality"])
            )

            print(
                "PRED:",
                int(row["prediction"])
            )

            print(
                f"P(class1): "
                f"{row['probability_class1']:.6f}"
            )

            print(
                "DICOM:",
                row["dicom_path"]
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        RESULT_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("RESULT FILE")
    print("=" * 60)

    print(
        RESULT_PATH
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("V11 EXACT EVALUATION")
    print(
        "Device:",
        DEVICE
    )

    test_df = prepare_exact_test()

    model = load_model()

    (
        y_true,
        y_pred,
        probabilities,
        studies,
        paths,
    ) = evaluate(
        model,
        test_df
    )

    print_results(
        test_df,
        y_true,
        y_pred,
        probabilities,
        studies,
        paths,
    )


if __name__ == "__main__":
    main()