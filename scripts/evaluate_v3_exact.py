from pathlib import Path
import csv
import hashlib

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torchvision.models import resnet18


PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TEST_CSV = PROJECT_DIR / "data" / "splits" / "test.csv"
MODEL_FILE = PROJECT_DIR / "models" / "spine_quality_resnet18_best.pth"

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")


# ============================================================
# ТОЧНАЯ АРХИТЕКТУРА V3
# ============================================================

class SpineQualityResNet18V3(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=None)

        num_features = self.model.fc.in_features

        # ВАЖНО:
        # V3 checkpoint содержит обычный Linear,
        # а не Dropout + Linear.
        self.model.fc = nn.Linear(
            num_features,
            2
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
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

def load_model():

    print()
    print("Загрузка V3:")
    print(MODEL_FILE)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
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

    # Если checkpoint был сохранён внутри объекта base
    if any(
        key.startswith("base.")
        for key in state_dict.keys()
    ):
        state_dict = {
            key[len("base."):]: value
            for key, value in state_dict.items()
        }

    model = SpineQualityResNet18V3()

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print()
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("ERROR — missing:")
        for key in missing:
            print(" ", key)

    if unexpected:
        print("ERROR — unexpected:")
        for key in unexpected:
            print(" ", key)

    if not missing and not unexpected:
        print("✓ Checkpoint загружен полностью")

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# TEST
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
# PIXEL HASH
# ============================================================

def get_pixel_hash(path):

    dataset = pydicom.dcmread(path)

    return hashlib.sha256(
        dataset.pixel_array.tobytes()
    ).hexdigest()


# ============================================================
# УДАЛЕНИЕ ДУБЛИКАТОВ
# ============================================================

def make_unique_rows(rows):

    unique_rows = []
    seen = set()

    for row in rows:

        path = PROJECT_DIR / row["dicom_path"]

        pixel_hash = get_pixel_hash(path)

        if pixel_hash in seen:
            continue

        seen.add(pixel_hash)

        new_row = dict(row)
        new_row["_pixel_hash"] = pixel_hash

        unique_rows.append(new_row)

    return unique_rows


# ============================================================
# ТОЧНОЕ V3 PREPROCESSING
# ============================================================

def load_image(path):

    dataset = pydicom.dcmread(path)

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
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(model, image):

    image = image.unsqueeze(0)

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    prediction = int(
        torch.argmax(
            probabilities
        ).item()
    )

    p0 = probabilities[0].item()
    p1 = probabilities[1].item()

    return prediction, p0, p1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ТОЧНАЯ ПРОВЕРКА V3")
    print("=" * 70)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    rows = read_test_rows()

    print()
    print(
        "TEST строк до удаления дублей:",
        len(rows)
    )

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
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("ПРЕДСКАЗАНИЯ V3")
    print("=" * 70)

    for row in unique_rows:

        path = PROJECT_DIR / row["dicom_path"]

        image = load_image(path)

        prediction, p0, p1 = predict(
            model,
            image
        )

        true_label = int(
            row["spine_quality"]
        )

        correct = prediction == true_label

        results.append({
            "study_id": row["study_id"],
            "dicom_path": row["dicom_path"],
            "true": true_label,
            "pred": prediction,
            "p0": p0,
            "p1": p1,
            "correct": correct
        })

        print(
            f'{row["study_id"]} | '
            f'TRUE={true_label} | '
            f'PRED={prediction} | '
            f'P0={p0:.4f} | '
            f'P1={p1:.4f} | '
            f'{"OK" if correct else "ERROR"}'
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total = len(results)

    correct = sum(
        r["correct"]
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

    tp = sum(
        r["true"] == 1 and r["pred"] == 1
        for r in results
    )

    accuracy = correct / total * 100

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
    print("=" * 70)
    print("ИТОГ V3")
    print("=" * 70)

    print(
        f"Правильных: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print()
    print("Confusion matrix:")
    print(
        f"True 0 -> Pred 0: {tn}"
    )
    print(
        f"True 0 -> Pred 1: {fp}"
    )
    print(
        f"True 1 -> Pred 0: {fn}"
    )
    print(
        f"True 1 -> Pred 1: {tp}"
    )

    print()
    print("CLASS 1")
    print(
        f"Precision: {precision:.2f}%"
    )
    print(
        f"Recall:    {recall:.2f}%"
    )
    print(
        f"F1:        {f1:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        PROJECT_DIR
        / "results"
        / "spine_quality_v3_exact_predictions.csv"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        fieldnames = [
            "study_id",
            "dicom_path",
            "true",
            "pred",
            "p0",
            "p1",
            "correct"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print("Результаты сохранены:")
    print(output_file)

    print()
    print("=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()