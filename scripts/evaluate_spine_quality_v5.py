from pathlib import Path
import csv
import hashlib

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TEST_CSV = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_FILE = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_v5_best.pth"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "results"
    / "spine_quality_v5_unique_predictions.csv"
)

IMAGE_SIZE = 224


# ============================================================
# MODEL
# ============================================================

class SpineQualityResNet18V5(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(
            weights=None
        )

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.40),
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
# IMAGE
# ============================================================

def load_image(path):

    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(np.float32)

    # percentile normalization
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high > low:

        image = np.clip(
            image,
            low,
            high
        )

        image = (
            image - low
        ) / (
            high - low
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32
        )

    # preserve aspect ratio
    h, w = image.shape

    scale = min(
        IMAGE_SIZE / h,
        IMAGE_SIZE / w
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    result = torch.zeros(
        1,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    top = (
        IMAGE_SIZE - new_h
    ) // 2

    left = (
        IMAGE_SIZE - new_w
    ) // 2

    result[
        :,
        top:top + new_h,
        left:left + new_w
    ] = tensor

    return result


# ============================================================
# METRICS
# ============================================================

def metrics(pred, true):

    pred = np.array(pred)
    true = np.array(true)

    accuracy = np.mean(
        pred == true
    )

    tp = np.sum(
        (pred == 1) & (true == 1)
    )

    fp = np.sum(
        (pred == 1) & (true == 0)
    )

    fn = np.sum(
        (pred == 0) & (true == 1)
    )

    tn = np.sum(
        (pred == 0) & (true == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("V4 TEST — UNIQUE IMAGES")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Read test CSV
    # --------------------------------------------------------

    with open(
        TEST_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    rows = [
        row
        for row in rows
        if row["anatomy"] == "spine"
        and row["spine_quality"] in {"0", "1"}
    ]

    print(
        f"TEST spine rows: {len(rows)}"
    )

    # --------------------------------------------------------
    # Remove duplicate PixelData
    # --------------------------------------------------------

    unique_rows = []
    hashes = set()

    for row in rows:

        path = PROJECT_DIR / row["dicom_path"]

        ds = pydicom.dcmread(path)

        pixel_hash = hashlib.sha256(
            ds.pixel_array.tobytes()
        ).hexdigest()

        if pixel_hash in hashes:
            continue

        hashes.add(pixel_hash)

        unique_rows.append(row)

    print(
        f"Unique TEST images: {len(unique_rows)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = SpineQualityResNet18V5()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded V4 epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    print(
        f"Validation class1 F1: "
        f"{checkpoint['val_f1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = []
    true_labels = []

    output_rows = []

    with torch.no_grad():

        for row in unique_rows:

            path = PROJECT_DIR / row["dicom_path"]

            image = load_image(path)

            image = image.unsqueeze(0).to(device)

            logits = model(image)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            p0 = float(
                probabilities[0].cpu()
            )

            p1 = float(
                probabilities[1].cpu()
            )

            predicted = int(
                torch.argmax(
                    probabilities
                ).cpu()
            )

            true_label = int(
                row["spine_quality"]
            )

            predictions.append(
                predicted
            )

            true_labels.append(
                true_label
            )

            output_rows.append({
                "study_id":
                    row["study_id"],

                "dicom_path":
                    row["dicom_path"],

                "true_label":
                    true_label,

                "predicted_label":
                    predicted,

                "p0":
                    p0,

                "p1":
                    p1,
            })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        accuracy,
        precision,
        recall,
        f1,
        tn,
        fp,
        fn,
        tp
    ) = metrics(
        predictions,
        true_labels
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Correct: "
        f"{sum(p == t for p, t in zip(predictions, true_labels))}"
        f"/{len(true_labels)}"
    )

    print(
        f"TEST Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print("Confusion matrix:")
    print()
    print("              Pred 0    Pred 1")
    print(
        f"True 0       {tn:8d}    {fp:8d}"
    )
    print(
        f"True 1       {fn:8d}    {tp:8d}"
    )

    print()
    print(
        f"Class 1 Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Class 1 Recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Class 1 F1: "
        f"{f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "study_id",
                "dicom_path",
                "true_label",
                "predicted_label",
                "p0",
                "p1",
            ]
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    print()
    print(
        "Predictions saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()