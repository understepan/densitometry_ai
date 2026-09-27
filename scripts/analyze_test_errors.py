from pathlib import Path
import sys
import csv

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ============================================================
# Пути проекта
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


from src.preprocessing.dataset import SpineQualityDataset
from src.models.resnet18_model import SpineQualityResNet18


# ============================================================
# Настройки
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


TEST_CSV = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "spine_quality_resnet18_best.pth"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

VISUALIZATION_DIR = (
    PROJECT_DIR
    / "visualizations"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

VISUALIZATION_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Проверки
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Модель не найдена:\n{MODEL_PATH}"
    )


# ============================================================
# Dataset
# ============================================================

dataset = SpineQualityDataset(
    csv_file=TEST_CSV,
    project_dir=PROJECT_DIR,
    image_size=IMAGE_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("=" * 70)
print("АНАЛИЗ ОШИБОК TEST — SPINE QUALITY")
print("=" * 70)

print()
print(f"Устройство: {DEVICE}")
print(f"TEST изображений: {len(dataset)}")


# ============================================================
# Загружаем модель
# ============================================================

model = SpineQualityResNet18(
    pretrained=False
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print()
print(
    f"Загружена модель из эпохи: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation Accuracy: "
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# Предсказания
# ============================================================

results = []


with torch.no_grad():

    for index, batch in enumerate(loader):

        image = batch["image"].to(DEVICE)
        label = int(
            batch["label"].item()
        )

        study_id = batch["study_id"][0]
        dicom_path = batch["dicom_path"][0]

        # Предсказание
        output = model(image)

        # Вероятности
        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        probability_0 = float(
            probabilities[0].item()
        )

        probability_1 = float(
            probabilities[1].item()
        )

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        correct = (
            prediction == label
        )

        result = {
            "index": index,
            "study_id": study_id,
            "dicom_path": dicom_path,
            "true_label": label,
            "predicted_label": prediction,
            "probability_0": probability_0,
            "probability_1": probability_1,
            "correct": correct,
        }

        results.append(result)


# ============================================================
# Сохраняем CSV
# ============================================================

csv_path = (
    RESULTS_DIR
    / "spine_quality_test_predictions.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    fieldnames = [
        "index",
        "study_id",
        "dicom_path",
        "true_label",
        "predicted_label",
        "probability_0",
        "probability_1",
        "correct",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in results:
        writer.writerow(result)


# ============================================================
# Выводим таблицу
# ============================================================

print()
print("-" * 70)
print("ПРЕДСКАЗАНИЯ TEST")
print("-" * 70)

print()

for result in results:

    status = (
        "CORRECT"
        if result["correct"]
        else "ERROR"
    )

    print(
        f"[{result['index']:02d}] "
        f"TRUE={result['true_label']} "
        f"PRED={result['predicted_label']} "
        f"P0={result['probability_0']:.3f} "
        f"P1={result['probability_1']:.3f} "
        f"{status}"
    )

    print(
        f"     study: "
        f"{result['study_id']}"
    )


# ============================================================
# Считаем ошибки
# ============================================================

errors = [
    result
    for result in results
    if not result["correct"]
]

correct_results = [
    result
    for result in results
    if result["correct"]
]


print()
print("-" * 70)

print(
    f"Правильных: "
    f"{len(correct_results)}"
)

print(
    f"Ошибок: "
    f"{len(errors)}"
)


# ============================================================
# Создаём контактный лист
# ============================================================

from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------
# Настройки
# ------------------------------------------------------------

CARD_WIDTH = 500
IMAGE_HEIGHT = 400

TEXT_HEIGHT = 130

COLUMNS = 3

BACKGROUND = "white"

OUTPUT_WIDTH = COLUMNS * CARD_WIDTH

# Высота одной карточки:
# изображение + отдельная область текста
CARD_HEIGHT = IMAGE_HEIGHT + TEXT_HEIGHT

ROWS = (
    len(results) + COLUMNS - 1
) // COLUMNS

OUTPUT_HEIGHT = ROWS * CARD_HEIGHT


# ------------------------------------------------------------
# Создаём общий белый лист
# ------------------------------------------------------------

sheet = Image.new(
    "RGB",
    (
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT
    ),
    BACKGROUND
)


# ------------------------------------------------------------
# Шрифт
# ------------------------------------------------------------

try:

    font_large = ImageFont.truetype(
        "arial.ttf",
        22
    )

    font_small = ImageFont.truetype(
        "arial.ttf",
        18
    )

except:

    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()


# ------------------------------------------------------------
# Обрабатываем каждое изображение
# ------------------------------------------------------------

for item_index, result in enumerate(results):

    dataset_index = result["index"]

    sample = dataset[dataset_index]

    image_array = (
        sample["image"]
        .squeeze(0)
        .numpy()
    )


    # --------------------------------------------------------
    # Переводим изображение из [0, 1] в [0, 255]
    # --------------------------------------------------------

    image_array = (
        image_array * 255
    ).clip(0, 255).astype("uint8")


    image = Image.fromarray(
        image_array,
        mode="L"
    ).convert("RGB")


    # --------------------------------------------------------
    # Изменяем размер снимка
    # --------------------------------------------------------

    image.thumbnail(
        (
            CARD_WIDTH - 30,
            IMAGE_HEIGHT - 20
        )
    )


    # --------------------------------------------------------
    # Создаём область изображения
    # --------------------------------------------------------

    image_area = Image.new(
        "RGB",
        (
            CARD_WIDTH,
            IMAGE_HEIGHT
        ),
        "black"
    )


    x_image = (
        CARD_WIDTH - image.width
    ) // 2

    y_image = (
        IMAGE_HEIGHT - image.height
    ) // 2


    image_area.paste(
        image,
        (
            x_image,
            y_image
        )
    )


    # --------------------------------------------------------
    # Создаём карточку
    # --------------------------------------------------------

    card = Image.new(
        "RGB",
        (
            CARD_WIDTH,
            CARD_HEIGHT
        ),
        "white"
    )


    # Сначала вставляем снимок
    card.paste(
        image_area,
        (
            0,
            0
        )
    )


    # --------------------------------------------------------
    # Рисуем текст В ОТДЕЛЬНОЙ области
    # --------------------------------------------------------

    draw = ImageDraw.Draw(card)


    if result["correct"]:

        status = "CORRECT"

    else:

        status = "ERROR"


    study_id = result["study_id"]

    short_study = study_id[-12:]


    line1 = status

    line2 = (
        f"TRUE = {result['true_label']}    "
        f"PRED = {result['predicted_label']}"
    )

    line3 = (
        f"P(0) = {result['probability_0']:.2f}    "
        f"P(1) = {result['probability_1']:.2f}"
    )

    line4 = (
        f"study ...{short_study}"
    )


    # --------------------------------------------------------
    # Центрируем каждую строку
    # --------------------------------------------------------

    def draw_centered(
        text,
        y,
        font
    ):

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        x = (
            CARD_WIDTH - text_width
        ) // 2

        draw.text(
            (
                x,
                y
            ),
            text,
            fill="black",
            font=font
        )


    # --------------------------------------------------------
    # Текст находится ТОЛЬКО ниже снимка
    # --------------------------------------------------------

    draw_centered(
        line1,
        IMAGE_HEIGHT + 8,
        font_large
    )

    draw_centered(
        line2,
        IMAGE_HEIGHT + 38,
        font_small
    )

    draw_centered(
        line3,
        IMAGE_HEIGHT + 65,
        font_small
    )

    draw_centered(
        line4,
        IMAGE_HEIGHT + 92,
        font_small
    )


    # --------------------------------------------------------
    # Координаты карточки на общем листе
    # --------------------------------------------------------

    column = (
        item_index % COLUMNS
    )

    row = (
        item_index // COLUMNS
    )


    x_sheet = (
        column * CARD_WIDTH
    )

    y_sheet = (
        row * CARD_HEIGHT
    )


    # --------------------------------------------------------
    # Вставляем готовую карточку
    # --------------------------------------------------------

    sheet.paste(
        card,
        (
            x_sheet,
            y_sheet
        )
    )


# ============================================================
# Сохраняем
# ============================================================

figure_path = (
    VISUALIZATION_DIR
    / "spine_quality_test_predictions.png"
)


sheet.save(
    figure_path
)


print()
print("=" * 70)

print(
    f"Контактный лист сохранён:\n"
    f"{figure_path}"
)

print("=" * 70)


# ============================================================
# Финальный вывод
# ============================================================

print()
print("=" * 70)

print(
    f"CSV сохранён:\n{csv_path}"
)

print()

print(
    f"Контактный лист сохранён:\n"
    f"{figure_path}"
)

print("=" * 70)