from pathlib import Path
import csv
import hashlib

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TEST_CSV = PROJECT_DIR / "data" / "splits" / "test.csv"

V3_MODEL_FILE = PROJECT_DIR / "models" / "spine_quality_resnet18_v3_best.pth"
V5_MODEL_FILE = PROJECT_DIR / "models" / "spine_quality_resnet18_v5_best.pth"

OUTPUT_FILE = PROJECT_DIR / "results" / "compare_v3_v5.csv"

IMAGE_SIZE = 224


# ============================================================
# МОДЕЛЬ V3
# ============================================================

class SpineQualityResNet18V3(nn.Module):

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
# МОДЕЛЬ V5
# ============================================================

class SpineQualityResNet18V5(nn.Module):

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
# PIXEL HASH
# ============================================================

def get_pixel_hash(dataset):

    if "PixelData" not in dataset:
        return None

    return hashlib.sha256(
        dataset.PixelData
    ).hexdigest()


# ============================================================
# ПОДГОТОВКА ИЗОБРАЖЕНИЯ
# ============================================================

def load_image(dicom_path):

    dataset = pydicom.dcmread(dicom_path)

    image = dataset.pixel_array.astype(np.float32)

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

    image = torch.from_numpy(image)

    image = image.unsqueeze(0)

    image = F.interpolate(
        image.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    image = image.squeeze(0)

    return image


# ============================================================
# ПОЛУЧЕНИЕ УНИКАЛЬНЫХ TEST ИЗОБРАЖЕНИЙ
# ============================================================

def load_unique_test_images():

    rows = []

    with open(
        TEST_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("anatomy") != "spine":
                continue

            if row.get("spine_quality") not in {"0", "1"}:
                continue

            dicom_path = PROJECT_DIR / row["dicom_path"]

            if not dicom_path.exists():
                print(
                    f"Файл не найден: {dicom_path}"
                )
                continue

            dataset = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=False
            )

            pixel_hash = get_pixel_hash(dataset)

            if pixel_hash is None:
                continue

            rows.append({
                "study_id": row["study_id"],
                "dicom_path": row["dicom_path"],
                "label": int(row["spine_quality"]),
                "hash": pixel_hash
            })

    # Удаляем повторяющийся PixelData
    unique = {}

    for row in rows:

        if row["hash"] not in unique:
            unique[row["hash"]] = row

    return list(unique.values())


# ============================================================
# ПРЕДСКАЗАНИЕ
# ============================================================

def predict(model, image, device):

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(image)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    p0 = float(probabilities[0].item())
    p1 = float(probabilities[1].item())

    return predicted_class, p0, p1


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model(model_class, model_file, device):

    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # В некоторых версиях модели checkpoint содержит префикс
    # "base." перед названием основной модели.
    #
    # Например:
    # base.model.conv1.weight
    #
    # А текущий класс ожидает:
    # model.conv1.weight
    # --------------------------------------------------------

    if any(
        key.startswith("base.")
        for key in state_dict.keys()
    ):

        state_dict = {
            key[len("base."):]: value
            for key, value in state_dict.items()
        }

    model = model_class().to(device)

    model.load_state_dict(state_dict)

    model.eval()

    return model
# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("СРАВНЕНИЕ V3 И V5")
    print("=" * 70)

    device = torch.device("cpu")

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Загружаем TEST
    # --------------------------------------------------------

    test_rows = load_unique_test_images()

    print()
    print(
        f"Уникальных TEST изображений: "
        f"{len(test_rows)}"
    )

    # --------------------------------------------------------
    # Загружаем модели
    # --------------------------------------------------------

    print()
    print("Загрузка V3...")

    model_v3 = load_model(
        SpineQualityResNet18V3,
        V3_MODEL_FILE,
        device
    )

    print("V3 загружена.")

    print()
    print("Загрузка V5...")

    model_v5 = load_model(
        SpineQualityResNet18V5,
        V5_MODEL_FILE,
        device
    )

    print("V5 загружена.")

    # --------------------------------------------------------
    # Предсказания
    # --------------------------------------------------------

    results = []

    for row in test_rows:

        dicom_path = PROJECT_DIR / row["dicom_path"]

        image = load_image(dicom_path)

        v3_pred, v3_p0, v3_p1 = predict(
            model_v3,
            image,
            device
        )

        v5_pred, v5_p0, v5_p1 = predict(
            model_v5,
            image,
            device
        )

        true_label = row["label"]

        v3_correct = v3_pred == true_label
        v5_correct = v5_pred == true_label

        if v3_correct and v5_correct:
            category = "Обе правильно"

        elif v3_correct and not v5_correct:
            category = "V3 правильно, V5 ошибка"

        elif not v3_correct and v5_correct:
            category = "V5 правильно, V3 ошибка"

        else:
            category = "Обе ошиблись"

        results.append({
            "study_id": row["study_id"],
            "dicom_path": row["dicom_path"],
            "true_label": true_label,

            "v3_pred": v3_pred,
            "v3_p0": v3_p0,
            "v3_p1": v3_p1,

            "v5_pred": v5_pred,
            "v5_p0": v5_p0,
            "v5_p1": v5_p1,

            "category": category
        })

    # --------------------------------------------------------
    # Таблица
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ПОДРОБНОЕ СРАВНЕНИЕ")
    print("=" * 70)

    for i, row in enumerate(results, start=1):

        print()
        print(
            f"{i:02d}. "
            f"TRUE={row['true_label']} | "
            f"V3={row['v3_pred']} "
            f"(P1={row['v3_p1']:.3f}) | "
            f"V5={row['v5_pred']} "
            f"(P1={row['v5_p1']:.3f})"
        )

        print(
            f"    Study: {row['study_id']}"
        )

        print(
            f"    DICOM: {row['dicom_path']}"
        )

        print(
            f"    {row['category']}"
        )

    # --------------------------------------------------------
    # Группы
    # --------------------------------------------------------

    categories = {
        "Обе правильно": [],
        "V3 правильно, V5 ошибка": [],
        "V5 правильно, V3 ошибка": [],
        "Обе ошиблись": []
    }

    for row in results:
        categories[row["category"]].append(row)

    print()
    print("=" * 70)
    print("ГРУППЫ")
    print("=" * 70)

    for category, rows in categories.items():

        print()
        print(
            f"{category}: {len(rows)}"
        )

        for row in rows:

            print(
                f"  Study={row['study_id']} | "
                f"TRUE={row['true_label']} | "
                f"V3={row['v3_pred']} "
                f"P1={row['v3_p1']:.3f} | "
                f"V5={row['v5_pred']} "
                f"P1={row['v5_p1']:.3f}"
            )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    v3_correct_count = sum(
        row["v3_pred"] == row["true_label"]
        for row in results
    )

    v5_correct_count = sum(
        row["v5_pred"] == row["true_label"]
        for row in results
    )

    total = len(results)

    print()
    print("=" * 70)
    print("ИТОГ")
    print("=" * 70)

    print()
    print(
        f"V3: {v3_correct_count}/{total} = "
        f"{100 * v3_correct_count / total:.2f}%"
    )

    print(
        f"V5: {v5_correct_count}/{total} = "
        f"{100 * v5_correct_count / total:.2f}%"
    )

    print()
    print(
        f"V3 правильно, V5 ошибка: "
        f"{len(categories['V3 правильно, V5 ошибка'])}"
    )

    print(
        f"V5 правильно, V3 ошибка: "
        f"{len(categories['V5 правильно, V3 ошибка'])}"
    )

    print(
        f"Обе правильно: "
        f"{len(categories['Обе правильно'])}"
    )

    print(
        f"Обе ошиблись: "
        f"{len(categories['Обе ошиблись'])}"
    )

    # --------------------------------------------------------
    # Сохранение CSV
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "study_id",
        "dicom_path",
        "true_label",
        "v3_pred",
        "v3_p0",
        "v3_p1",
        "v5_pred",
        "v5_p0",
        "v5_p1",
        "category"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print(
        f"Результаты сохранены:"
    )
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()