from pathlib import Path
import csv
import random
import hashlib

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "validation.csv"

MODEL_OUTPUT = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_v4_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8

MAX_EPOCHS = 30
PATIENCE = 7

SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DATASET
# ============================================================

class SpineQualityDataset(Dataset):

    def __init__(
        self,
        csv_file,
        project_dir,
        image_size=224,
        train=False,
    ):
        self.csv_file = Path(csv_file)
        self.project_dir = Path(project_dir)
        self.image_size = image_size
        self.train = train

        with open(
            self.csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)
            all_rows = list(reader)

        # ----------------------------------------------------
        # Берём только позвоночник с известной разметкой
        # ----------------------------------------------------

        rows = []

        for row in all_rows:

            if row["anatomy"] != "spine":
                continue

            label = row["spine_quality"]

            if label not in {"0", "1"}:
                continue

            rows.append(row)

        # ----------------------------------------------------
        # Удаляем дублирующиеся PixelData
        # ----------------------------------------------------

        unique_rows = []
        hashes = set()

        for row in rows:

            full_path = self.project_dir / row["dicom_path"]

            ds = pydicom.dcmread(full_path)

            pixel_array = ds.pixel_array

            pixel_hash = hashlib.sha256(
                pixel_array.tobytes()
            ).hexdigest()

            if pixel_hash in hashes:
                continue

            hashes.add(pixel_hash)
            unique_rows.append(row)

        self.rows = unique_rows

        if not self.rows:
            raise ValueError(
                "После фильтрации не осталось изображений."
            )

        print()
        print(
            f"{self.csv_file.name}: "
            f"исходных строк = {len(rows)}, "
            f"уникальных изображений = {len(self.rows)}"
        )

    def __len__(self):
        return len(self.rows)

    # --------------------------------------------------------
    # Нормализация изображения
    # --------------------------------------------------------

    def normalize_image(self, image):

        image = image.astype(np.float32)

        # Percentile normalization.
        # Убираем влияние очень тёмных/светлых выбросов.

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

        return image

    # --------------------------------------------------------
    # Resize с сохранением пропорций
    # --------------------------------------------------------

    def resize_with_padding(self, image):

        tensor = torch.from_numpy(image).unsqueeze(0)

        height = tensor.shape[1]
        width = tensor.shape[2]

        scale = min(
            self.image_size / height,
            self.image_size / width
        )

        new_height = max(
            1,
            int(round(height * scale))
        )

        new_width = max(
            1,
            int(round(width * scale))
        )

        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        result = torch.zeros(
            1,
            self.image_size,
            self.image_size,
            dtype=torch.float32
        )

        top = (
            self.image_size - new_height
        ) // 2

        left = (
            self.image_size - new_width
        ) // 2

        result[
            :,
            top:top + new_height,
            left:left + new_width
        ] = tensor

        return result

    # --------------------------------------------------------
    # Аугментация
    # --------------------------------------------------------

    def augment(self, image):

        # Небольшой случайный сдвиг
        if random.random() < 0.5:

            shift_x = random.randint(-3, 3)
            shift_y = random.randint(-3, 3)

            image = torch.roll(
                image,
                shifts=(shift_y, shift_x),
                dims=(1, 2)
            )

        # Очень небольшой масштаб
        if random.random() < 0.5:

            scale = random.uniform(
                0.98,
                1.02
            )

            h = image.shape[1]
            w = image.shape[2]

            new_h = int(h * scale)
            new_w = int(w * scale)

            resized = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            if scale >= 1.0:

                top = (new_h - h) // 2
                left = (new_w - w) // 2

                image = resized[
                    :,
                    top:top + h,
                    left:left + w
                ]

            else:

                result = torch.zeros_like(image)

                top = (h - new_h) // 2
                left = (w - new_w) // 2

                result[
                    :,
                    top:top + new_h,
                    left:left + new_w
                ] = resized

                image = result

        return image

    # --------------------------------------------------------
    # Загрузка DICOM
    # --------------------------------------------------------

    def load_image(self, dicom_path):

        full_path = self.project_dir / dicom_path

        if not full_path.exists():

            raise FileNotFoundError(
                f"Файл не найден: {full_path}"
            )

        dataset = pydicom.dcmread(full_path)

        image = dataset.pixel_array

        image = self.normalize_image(image)

        image = self.resize_with_padding(image)

        if self.train:
            image = self.augment(image)

        return image

    # --------------------------------------------------------
    # Получение элемента
    # --------------------------------------------------------

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
            "dicom_path": row["dicom_path"],
        }


# ============================================================
# MODEL
# ============================================================

class SpineQualityResNet18V4(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(
            weights=weights
        )

        num_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Sequential(

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                num_features,
                2
            )
        )

    def forward(self, x):

        # DICOM grayscale -> RGB

        if x.shape[1] == 1:

            x = x.repeat(
                1,
                3,
                1,
                1
            )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device
        ).view(1, 3, 1, 1)

        x = (
            x - mean
        ) / std

        return self.model(x)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    labels
):

    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = (
        predictions == labels
    ).mean()

    tp = np.sum(
        (predictions == 1)
        & (labels == 1)
    )

    fp = np.sum(
        (predictions == 1)
        & (labels == 0)
    )

    fn = np.sum(
        (predictions == 0)
        & (labels == 1)
    )

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return accuracy, precision, recall, f1


# ============================================================
# TRAIN
# ============================================================

def train():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 60)
    print("BASELINE №4")
    print("RESNET18 + UNIQUE IMAGES + PERCENTILE NORMALIZATION")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = SpineQualityDataset(
        TRAIN_CSV,
        PROJECT_DIR,
        IMAGE_SIZE,
        train=True,
    )

    val_dataset = SpineQualityDataset(
        VAL_CSV,
        PROJECT_DIR,
        IMAGE_SIZE,
        train=False,
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    train_labels = [
        int(row["spine_quality"])
        for row in train_dataset.rows
    ]

    class0 = train_labels.count(0)
    class1 = train_labels.count(1)

    print()
    print(
        f"TRAIN class 0: {class0}"
    )

    print(
        f"TRAIN class 1: {class1}"
    )

    # --------------------------------------------------------
    # Weighted sampler
    # --------------------------------------------------------

    class_weights = {
        0: 1.0 / class0,
        1: 1.0 / class1,
    }

    sample_weights = [
        class_weights[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(
            sample_weights
        ),
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SpineQualityResNet18V4()

    model = model.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():

        if "model.fc" in name:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": 2e-5,
            },
            {
                "params": head_parameters,
                "lr": 2e-4,
            },
        ],
        weight_decay=2e-4,
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_val_f1 = -1.0

    best_epoch = 0
    patience_counter = 0

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_losses = []
        train_predictions = []
        train_labels_epoch = []

        for batch in train_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
                .detach()
                .cpu()
                .numpy()
            )

            train_predictions.extend(
                predictions.tolist()
            )

            train_labels_epoch.extend(
                labels.detach()
                .cpu()
                .numpy()
                .tolist()
            )

        train_accuracy, _, train_recall, train_f1 = (
            calculate_metrics(
                train_predictions,
                train_labels_epoch
            )
        )

        train_loss = np.mean(
            train_losses
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_losses = []
        val_predictions = []
        val_labels_epoch = []

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_losses.append(
                    loss.item()
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                val_predictions.extend(
                    predictions.tolist()
                )

                val_labels_epoch.extend(
                    labels.cpu()
                    .numpy()
                    .tolist()
                )

        (
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = calculate_metrics(
            val_predictions,
            val_labels_epoch
        )

        val_loss = np.mean(
            val_losses
        )

        scheduler.step(
            val_accuracy
        )

        print()
        print(
            f"Epoch {epoch}"
        )

        print(
            f"Train loss: {train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Train class1 recall: "
            f"{train_recall * 100:.2f}%"
        )

        print(
            f"Train class1 F1: "
            f"{train_f1 * 100:.2f}%"
        )

        print(
            f"Val loss: {val_loss:.4f}"
        )

        print(
            f"Val accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val class1 precision: "
            f"{val_precision * 100:.2f}%"
        )

        print(
            f"Val class1 recall: "
            f"{val_recall * 100:.2f}%"
        )

        print(
            f"Val class1 F1: "
            f"{val_f1 * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_f1 > best_val_f1
            )
        ):

            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_accuracy":
                        val_accuracy,

                    "val_f1":
                        val_f1,
                },
                MODEL_OUTPUT,
            )

            patience_counter = 0

            print(
                ">>> BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print()
            print(
                "EARLY STOPPING"
            )

            break

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("V4 TRAINING FINISHED")
    print("=" * 60)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation class1 F1: "
        f"{best_val_f1 * 100:.2f}%"
    )

    print(
        f"Model saved to:"
    )

    print(
        MODEL_OUTPUT
    )


if __name__ == "__main__":
    train()