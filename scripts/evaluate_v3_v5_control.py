from pathlib import Path
import csv
import hashlib

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torchvision.models import resnet18


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TEST_CSV = PROJECT_DIR / "data" / "splits" / "test.csv"

V3_CHECKPOINT = PROJECT_DIR / "models" / "spine_quality_resnet18_best.pth"
V5_CHECKPOINT = PROJECT_DIR / "models" / "spine_quality_resnet18_v5_best.pth"

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")


# ============================================================
# МОДЕЛЬ
# ============================================================

class SpineQualityResNet18(nn.Module):

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
# ЗАГРУЗКА CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path):

    print()
    print("Загрузка:", checkpoint_path.name)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    # Иногда checkpoint V3 содержит base.model.*
    # Приводим его к model.*
    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("base."):
            new_key = key[len("base."):]

        else:
            new_key = key

        new_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        new_state_dict,
        strict=False
    )

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("Первые missing keys:")
        for key in missing[:5]:
            print(" ", key)

    if unexpected:
        print("Первые unexpected keys:")
        for key in unexpected[:5]:
            print(" ", key)

    model.eval()

    return model


# ============================================================
# ЧТЕНИЕ TEST
# ============================================================

def read_test_rows():

    with open(
        TEST_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = []

        for row in reader:

            if row["anatomy"] != "spine":
                continue

            if row["spine_quality"] not in {"0", "1"}:
                continue

            rows.append(row)

    return rows


# ============================================================
# УНИКАЛЬНЫЕ PIXEL DATA
# ============================================================

def get_pixel_hash(dicom_path):

    dataset = pydicom.dcmread(dicom_path)

    pixel_array = dataset.pixel_array

    return hashlib.sha256(
        pixel_array.tobytes()
    ).hexdigest()


def make_unique_rows(rows):

    unique_rows = []
    seen = set()

    for row in rows:

        full_path = PROJECT_DIR / row["dicom_path"]

        pixel_hash = get_pixel_hash(full_path)

        if pixel_hash in seen:
            continue

        seen.add(pixel_hash)

        row = dict(row)
        row["_pixel_hash"] = pixel_hash

        unique_rows.append(row)

    return unique_rows


# ============================================================
# ПОДГОТОВКА ИЗОБРАЖЕНИЯ
# ============================================================

def load_image(dicom_path):

    dataset = pydicom.dcmread(dicom_path)

    image = dataset.pixel_array.astype(
        np.float32
    )

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

    image = torch.from_numpy(
        image
    ).unsqueeze(0)

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    image = image.squeeze(0)

    return image


# ============================================================
# ПРЕДСКАЗАНИЕ
# ============================================================

@torch.no_grad()
def predict(model, image):

    image = image.unsqueeze(0)

    output = model(image)

    probability = torch.softmax(
        output,
        dim=1
    )[0, 1].item()

    prediction = int(
        torch.argmax(output, dim=1).item()
    )

    return prediction, probability


# ============================================================
# МЕТРИКИ
# ============================================================

def calculate_metrics(results):

    total = len(results)

    correct = sum(
        r["true"] == r["pred"]
        for r in results
    )

    accuracy = correct / total * 100

    tp = sum(
        r["true"] == 1 and r["pred"] == 1
        for r in results
    )

    tn = sum(
        r["true"] == 0 and r["pred"] == 0
        for r in results
    )

    fp = sum(
        r["true"] == 0 and r["pred"] == 1
        for r in results
    )

    fn = sum(
        r["true"] == 1 and r["pred"] == 0
        for r in results
    )

    precision = (
        tp / (tp + fp) * 100
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn) * 100
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    print()
    print("Всего:", total)
    print("Правильных:", correct)
    print(f"Accuracy: {accuracy:.2f}%")

    print()
    print("Confusion matrix:")
    print("TRUE 0 -> PRED 0:", tn)
    print("TRUE 0 -> PRED 1:", fp)
    print("TRUE 1 -> PRED 0:", fn)
    print("TRUE 1 -> PRED 1:", tp)

    print()
    print("Класс 1:")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1:        {f1:.2f}%")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# ОЦЕНКА МОДЕЛИ
# ============================================================

def evaluate_model(model, rows, model_name):

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    results = []

    for row in rows:

        full_path = PROJECT_DIR / row["dicom_path"]

        image = load_image(full_path)

        prediction, probability = predict(
            model,
            image
        )

        true_label = int(
            row["spine_quality"]
        )

        results.append({
            "study_id": row["study_id"],
            "dicom_path": row["dicom_path"],
            "true": true_label,
            "pred": prediction,
            "probability": probability
        })

        print(
            f'{row["study_id"]} | '
            f'TRUE={true_label} | '
            f'PRED={prediction} | '
            f'P1={probability:.4f}'
        )

    metrics = calculate_metrics(results)

    return results, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("КОНТРОЛЬНАЯ ПРОВЕРКА V3 vs V5")
    print("=" * 70)

    print()
    print("TEST:", TEST_CSV)

    # --------------------------------------------------------
    # 1. Читаем TEST
    # --------------------------------------------------------

    rows = read_test_rows()

    print()
    print("TEST строк до удаления дублей:", len(rows))

    # --------------------------------------------------------
    # 2. Убираем точные PixelData-дубли
    # --------------------------------------------------------

    unique_rows = make_unique_rows(rows)

    print(
        "Уникальных изображений:",
        len(unique_rows)
    )

    print(
        "Удалено дублей:",
        len(rows) - len(unique_rows)
    )

    # --------------------------------------------------------
    # 3. Загружаем V3
    # --------------------------------------------------------

    model_v3 = SpineQualityResNet18()

    model_v3 = load_checkpoint(
        model_v3,
        V3_CHECKPOINT
    )

    # --------------------------------------------------------
    # 4. Загружаем V5
    # --------------------------------------------------------

    model_v5 = SpineQualityResNet18()

    model_v5 = load_checkpoint(
        model_v5,
        V5_CHECKPOINT
    )

    # --------------------------------------------------------
    # 5. Оцениваем V3
    # --------------------------------------------------------

    results_v3, metrics_v3 = evaluate_model(
        model_v3,
        unique_rows,
        "V3"
    )

    # --------------------------------------------------------
    # 6. Оцениваем V5
    # --------------------------------------------------------

    results_v5, metrics_v5 = evaluate_model(
        model_v5,
        unique_rows,
        "V5"
    )

    # --------------------------------------------------------
    # 7. Сравнение
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("СРАВНЕНИЕ V3 И V5")
    print("=" * 70)

    print(
        f'V3 Accuracy: '
        f'{metrics_v3["accuracy"]:.2f}%'
    )

    print(
        f'V5 Accuracy: '
        f'{metrics_v5["accuracy"]:.2f}%'
    )

    print()

    v3_correct = 0
    v5_correct = 0
    both_correct = 0
    v3_only = 0
    v5_only = 0
    both_error = 0

    print(
        "TRUE | V3 | V5 | Study"
    )

    print("-" * 70)

    for r3, r5 in zip(
        results_v3,
        results_v5
    ):

        true = r3["true"]

        v3 = r3["pred"]
        v5 = r5["pred"]

        v3_ok = v3 == true
        v5_ok = v5 == true

        if v3_ok:
            v3_correct += 1

        if v5_ok:
            v5_correct += 1

        if v3_ok and v5_ok:
            both_correct += 1

        elif v3_ok and not v5_ok:
            v3_only += 1

        elif v5_ok and not v3_ok:
            v5_only += 1

        else:
            both_error += 1

        print(
            f'{true:^4} | '
            f'{v3:^2} | '
            f'{v5:^2} | '
            f'{r3["study_id"]}'
        )

    print()
    print("V3 правильно:", v3_correct)
    print("V5 правильно:", v5_correct)
    print("Обе правильно:", both_correct)
    print("V3 правильно / V5 ошибка:", v3_only)
    print("V5 правильно / V3 ошибка:", v5_only)
    print("Обе ошиблись:", both_error)

    print()
    print("=" * 70)
    print("КОНТРОЛЬНАЯ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()