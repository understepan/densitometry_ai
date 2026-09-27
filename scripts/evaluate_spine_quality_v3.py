from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.preprocessing.dataset import SpineQualityDataset
from scripts.train_spine_quality_v3 import SpineQualityResNet18V3


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TEST_CSV = PROJECT_DIR / "data" / "splits" / "test.csv"

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_v3_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, predictions):

    labels = np.array(labels)
    predictions = np.array(predictions)

    accuracy = np.mean(labels == predictions)

    tn = int(np.sum(
        (labels == 0) & (predictions == 0)
    ))

    fp = int(np.sum(
        (labels == 0) & (predictions == 1)
    ))

    fn = int(np.sum(
        (labels == 1) & (predictions == 0)
    ))

    tp = int(np.sum(
        (labels == 1) & (predictions == 1)
    ))

    precision_0 = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0
    )

    recall_0 = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1_0 = (
        2 * precision_0 * recall_0 /
        (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0
        else 0
    )

    precision_1 = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall_1 = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1_1 = (
        2 * precision_1 * recall_1 /
        (precision_1 + recall_1)
        if (precision_1 + recall_1) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EVALUATION — RESNET18 V3")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    test_dataset = SpineQualityDataset(
        csv_file=TEST_CSV,
        project_dir=PROJECT_DIR,
        image_size=IMAGE_SIZE,
        transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print()
    print(f"TEST images: {len(test_dataset)}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SpineQualityResNet18V3(
        pretrained=False,
        dropout=0.30
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded model epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation class1 F1: "
        f"{checkpoint['val_f1_1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    rows_for_csv = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            for i in range(len(labels)):

                true_label = int(
                    labels[i].cpu().item()
                )

                predicted_label = int(
                    predictions[i].cpu().item()
                )

                p0 = float(
                    probabilities[i, 0]
                    .cpu()
                    .item()
                )

                p1 = float(
                    probabilities[i, 1]
                    .cpu()
                    .item()
                )

                all_labels.append(
                    true_label
                )

                all_predictions.append(
                    predicted_label
                )

                rows_for_csv.append({
                    "study_id":
                        batch["study_id"][i],

                    "dicom_path":
                        batch["dicom_path"][i],

                    "true_label":
                        true_label,

                    "predicted_label":
                        predicted_label,

                    "p0":
                        p0,

                    "p1":
                        p1,
                })

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Correct: "
        f"{int(metrics['accuracy'] * len(all_labels))}"
        f"/{len(all_labels)}"
    )

    print(
        f"TEST Accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print()
    print("Confusion matrix:")

    print(
        f"True 0: "
        f"pred0 {metrics['tn']}, "
        f"pred1 {metrics['fp']}"
    )

    print(
        f"True 1: "
        f"pred0 {metrics['fn']}, "
        f"pred1 {metrics['tp']}"
    )

    print()
    print("CLASS 0")
    print(
        f"Precision: "
        f"{metrics['precision_0'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{metrics['recall_0'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{metrics['f1_0'] * 100:.2f}%"
    )

    print()
    print("CLASS 1")
    print(
        f"Precision: "
        f"{metrics['precision_1'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{metrics['recall_1'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{metrics['f1_1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    predicted_0 = sum(
        p == 0
        for p in all_predictions
    )

    predicted_1 = sum(
        p == 1
        for p in all_predictions
    )

    true_0 = sum(
        y == 0
        for y in all_labels
    )

    true_1 = sum(
        y == 1
        for y in all_labels
    )

    print()
    print("Распределение:")

    print(
        f"True class 0: {true_0}"
    )

    print(
        f"True class 1: {true_1}"
    )

    print(
        f"Predicted class 0: {predicted_0}"
    )

    print(
        f"Predicted class 1: {predicted_1}"
    )

    # --------------------------------------------------------
    # UNIQUE STUDIES
    # --------------------------------------------------------

    unique_studies = sorted(
        set(
            row["study_id"]
            for row in rows_for_csv
        )
    )

    print()
    print(
        f"Unique studies: "
        f"{len(unique_studies)}"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    output_path = (
        PROJECT_DIR
        / "results"
        / "spine_quality_v3_predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        fieldnames = [
            "study_id",
            "dicom_path",
            "true_label",
            "predicted_label",
            "p0",
            "p1",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows_for_csv:
            writer.writerow(row)

    print()
    print(
        f"Predictions saved:"
    )

    print(output_path)

    print()
    print("=" * 70)
    print("EVALUATION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()