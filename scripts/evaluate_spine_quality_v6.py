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
MODEL_PATH = PROJECT_DIR / "models" / "spine_quality_resnet18_v6_best.pth"

IMAGE_SIZE = 224


# ============================================================
# V6 DATASET
# ============================================================

class SpineDataset:
    def __init__(self, csv_file, project_dir, image_size=224):
        self.csv_file = Path(csv_file)
        self.project_dir = Path(project_dir)
        self.image_size = image_size

        with open(
            self.csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)
            rows = list(reader)

        # Только позвоночник
        rows = [
            row
            for row in rows
            if row["anatomy"] == "spine"
            and row["spine_quality"] in {"0", "1"}
        ]

        # Убираем точные дубликаты PixelData
        unique_rows = []
        hashes = set()

        for row in rows:

            dicom_path = self.project_dir / row["dicom_path"]

            dataset = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=False
            )

            pixel_bytes = dataset.PixelData

            image_hash = hashlib.md5(pixel_bytes).hexdigest()

            if image_hash in hashes:
                continue

            hashes.add(image_hash)
            unique_rows.append(row)

        self.rows = unique_rows

        if not self.rows:
            raise ValueError(
                "После фильтрации не осталось изображений."
            )

    def __len__(self):
        return len(self.rows)

    def load_image(self, dicom_path):

        full_path = self.project_dir / dicom_path

        dataset = pydicom.dcmread(full_path)

        image = dataset.pixel_array.astype(np.float32)

        # ----------------------------------------------------
        # Percentile normalization 1–99
        # ----------------------------------------------------

        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)

        if p99 > p1:
            image = np.clip(image, p1, p99)
            image = (image - p1) / (p99 - p1)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        image = torch.from_numpy(image).unsqueeze(0)

        # ----------------------------------------------------
        # Сохраняем aspect ratio
        # ----------------------------------------------------

        height, width = image.shape[-2:]

        scale = min(
            self.image_size / height,
            self.image_size / width
        )

        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))

        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False
        )

        image = image.squeeze(0)

        # ----------------------------------------------------
        # Zero padding до 224x224
        # ----------------------------------------------------

        canvas = torch.zeros(
            1,
            self.image_size,
            self.image_size,
            dtype=torch.float32
        )

        top = (self.image_size - new_height) // 2
        left = (self.image_size - new_width) // 2

        canvas[
            :,
            top:top + new_height,
            left:left + new_width
        ] = image

        return canvas

    def __getitem__(self, index):

        row = self.rows[index]

        image = self.load_image(
            row["dicom_path"]
        )

        label = torch.tensor(
            int(row["spine_quality"]),
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label,
            "study_id": row["study_id"],
            "dicom_path": row["dicom_path"]
        }


# ============================================================
# V6 MODEL
# ============================================================

class SpineQualityResNet18V6(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=None
        )

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):

        # grayscale -> 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ImageNet normalization
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

def load_model(device):

    model = SpineQualityResNet18V6()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Возможные форматы сохранения
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            # Если checkpoint сам является state_dict
            model.load_state_dict(checkpoint)

    else:

        model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model, checkpoint


# ============================================================
# МЕТРИКИ
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    tn = np.sum(
        (y_true == 0) & (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V6 TEST — UNIQUE TEST IMAGES")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Проверяем наличие файлов
    # --------------------------------------------------------

    if not TEST_CSV.exists():

        raise FileNotFoundError(
            f"Не найден TEST CSV:\n{TEST_CSV}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Не найдена модель V6:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SpineDataset(
        TEST_CSV,
        PROJECT_DIR,
        IMAGE_SIZE
    )

    print()
    print("TEST исходных строк:", end=" ")

    with open(
        TEST_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        original_rows = list(
            csv.DictReader(file)
        )

    original_spine_rows = [
        row
        for row in original_rows
        if row["anatomy"] == "spine"
        and row["spine_quality"] in {"0", "1"}
    ]

    print(len(original_spine_rows))

    print(
        "TEST уникальных изображений:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Классы
    # --------------------------------------------------------

    class0 = sum(
        1
        for row in dataset.rows
        if row["spine_quality"] == "0"
    )

    class1 = sum(
        1
        for row in dataset.rows
        if row["spine_quality"] == "1"
    )

    print(
        "TEST class 0:",
        class0
    )

    print(
        "TEST class 1:",
        class1
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = load_model(device)

    if isinstance(checkpoint, dict):

        epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        val_accuracy = checkpoint.get(
            "val_accuracy",
            "unknown"
        )

        val_f1 = checkpoint.get(
            "val_class1_f1",
            "unknown"
        )

        print()
        print("Loaded model epoch:", epoch)

        if val_accuracy != "unknown":
            print(
                f"Saved validation accuracy: "
                f"{float(val_accuracy) * 100:.2f}%"
            )

        if val_f1 != "unknown":
            print(
                f"Saved validation class1 F1: "
                f"{float(val_f1) * 100:.2f}%"
            )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    y_true = []
    y_pred = []
    probabilities = []

    prediction_rows = []

    with torch.no_grad():

        for index in range(len(dataset)):

            item = dataset[index]

            image = item["image"].unsqueeze(0).to(device)

            label = int(
                item["label"].item()
            )

            logits = model(image)

            probs = torch.softmax(
                logits,
                dim=1
            )[0]

            prediction = int(
                torch.argmax(probs).item()
            )

            p0 = float(probs[0].item())
            p1 = float(probs[1].item())

            y_true.append(label)
            y_pred.append(prediction)

            probabilities.append(
                (p0, p1)
            )

            prediction_rows.append({
                "study_id": item["study_id"],
                "dicom_path": item["dicom_path"],
                "true_label": label,
                "pred_label": prediction,
                "prob_class0": p0,
                "prob_class1": p1
            })

    # --------------------------------------------------------
    # Метрики
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    correct = sum(
        int(true == pred)
        for true, pred
        in zip(y_true, y_pred)
    )

    total = len(y_true)

    print()
    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"TEST Accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print()
    print("Confusion matrix:")
    print(
        f"True 0: pred0 {metrics['tn']} "
        f"pred1 {metrics['fp']}"
    )

    print(
        f"True 1: pred0 {metrics['fn']} "
        f"pred1 {metrics['tp']}"
    )

    print()
    print(
        f"CLASS 1 precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"CLASS 1 recall: "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"CLASS 1 F1: "
        f"{metrics['f1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Подробные предсказания
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DETAILED PREDICTIONS")
    print("=" * 70)

    for row in prediction_rows:

        status = (
            "CORRECT"
            if row["true_label"] == row["pred_label"]
            else "ERROR"
        )

        print(
            f"{status:7s} "
            f"TRUE={row['true_label']} "
            f"PRED={row['pred_label']} "
            f"P0={row['prob_class0']:.4f} "
            f"P1={row['prob_class1']:.4f} "
            f"study={row['study_id']} "
            f"file={Path(row['dicom_path']).name}"
        )

    # --------------------------------------------------------
    # Сохранение CSV
    # --------------------------------------------------------

    result_path = (
        PROJECT_DIR
        / "results"
        / "spine_quality_v6_exact_predictions.csv"
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        result_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        fieldnames = [
            "study_id",
            "dicom_path",
            "true_label",
            "pred_label",
            "prob_class0",
            "prob_class1"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            prediction_rows
        )

    print()
    print(
        "Результаты сохранены:"
    )

    print(result_path)

    print()
    print("=" * 70)
    print("V6 TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()