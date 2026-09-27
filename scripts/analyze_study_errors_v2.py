from pathlib import Path
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont


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

MODEL_PATH = PROJECT_DIR / "models" / "spine_quality_resnet18_v2_best.pth"

VISUALIZATION_DIR = (
    PROJECT_DIR
    / "visualizations"
)

VISUALIZATION_DIR.mkdir(
    exist_ok=True
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
print("АНАЛИЗ ОШИБОК ПО ИССЛЕДОВАНИЯМ")
print("=" * 70)

print()
print(f"Устройство: {DEVICE}")
print(f"TEST изображений: {len(dataset)}")


# ============================================================
# Модель
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
    f"Модель загружена. "
    f"Эпоха: {checkpoint['epoch']}"
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

        true_label = int(
            batch["label"].item()
        )

        study_id = batch["study_id"][0]

        dicom_path = batch["dicom_path"][0]


        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


        p0 = float(
            probabilities[0].item()
        )

        p1 = float(
            probabilities[1].item()
        )


        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )


        results.append(
            {
                "index": index,
                "study_id": study_id,
                "dicom_path": dicom_path,
                "true_label": true_label,
                "predicted_label": prediction,
                "p0": p0,
                "p1": p1,
            }
        )


# ============================================================
# Группировка по исследованиям
# ============================================================

studies = defaultdict(list)


for result in results:

    studies[
        result["study_id"]
    ].append(result)


# ============================================================
# Определяем результат исследования
# ============================================================

study_results = []


for study_id, items in studies.items():

    true_label = items[0]["true_label"]


    mean_p0 = sum(
        item["p0"]
        for item in items
    ) / len(items)


    mean_p1 = sum(
        item["p1"]
        for item in items
    ) / len(items)


    study_prediction = (
        1
        if mean_p1 > mean_p0
        else 0
    )


    correct = (
        study_prediction == true_label
    )


    study_results.append(
        {
            "study_id": study_id,
            "items": items,
            "true_label": true_label,
            "prediction": study_prediction,
            "mean_p0": mean_p0,
            "mean_p1": mean_p1,
            "correct": correct,
        }
    )


# ============================================================
# Сортировка
# ============================================================

study_results.sort(
    key=lambda x: (
        x["correct"],
        x["true_label"],
        x["study_id"],
    )
)


# ============================================================
# Вывод
# ============================================================

print()
print("-" * 70)
print("ИССЛЕДОВАНИЯ")
print("-" * 70)

print()


for number, study in enumerate(
    study_results,
    start=1
):

    status = (
        "CORRECT"
        if study["correct"]
        else "ERROR"
    )


    print(
        f"{number:02d}. "
        f"TRUE={study['true_label']} "
        f"PRED={study['prediction']} "
        f"P0={study['mean_p0']:.3f} "
        f"P1={study['mean_p1']:.3f} "
        f"{status}"
    )

    print(
        f"    study: "
        f"{study['study_id']}"
    )

    print(
        f"    images: "
        f"{len(study['items'])}"
    )


# ============================================================
# Шрифты
# ============================================================

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


# ============================================================
# Создание контактного листа
# ============================================================

CARD_WIDTH = 520
IMAGE_HEIGHT = 390
TEXT_HEIGHT = 150

COLUMNS = 3

CARD_HEIGHT = (
    IMAGE_HEIGHT
    + TEXT_HEIGHT
)


ROWS = (
    len(results)
    + COLUMNS
    - 1
) // COLUMNS


sheet = Image.new(
    "RGB",
    (
        COLUMNS * CARD_WIDTH,
        ROWS * CARD_HEIGHT
    ),
    "white"
)


# ============================================================
# Функция центрирования текста
# ============================================================

def draw_centered(
    draw,
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
        CARD_WIDTH
        - text_width
    ) // 2


    draw.text(
        (x, y),
        text,
        fill="black",
        font=font
    )


# ============================================================
# Добавляем изображения
# ============================================================

for item_index, result in enumerate(results):

    dataset_index = result["index"]


    sample = dataset[
        dataset_index
    ]


    image_array = (
        sample["image"]
        .squeeze(0)
        .numpy()
    )


    image_array = (
        image_array * 255
    ).clip(
        0,
        255
    ).astype("uint8")


    image = Image.fromarray(
        image_array,
        mode="L"
    ).convert("RGB")


    # --------------------------------------------------------
    # Размер изображения
    # --------------------------------------------------------

    image.thumbnail(
        (
            CARD_WIDTH - 30,
            IMAGE_HEIGHT - 20
        )
    )


    # --------------------------------------------------------
    # Область изображения
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
        CARD_WIDTH
        - image.width
    ) // 2


    y_image = (
        IMAGE_HEIGHT
        - image.height
    ) // 2


    image_area.paste(
        image,
        (
            x_image,
            y_image
        )
    )


    # --------------------------------------------------------
    # Карточка
    # --------------------------------------------------------

    card = Image.new(
        "RGB",
        (
            CARD_WIDTH,
            CARD_HEIGHT
        ),
        "white"
    )


    card.paste(
        image_area,
        (0, 0)
    )


    draw = ImageDraw.Draw(
        card
    )


    # --------------------------------------------------------
    # Статус
    # --------------------------------------------------------

    status = (
        "CORRECT"
        if result["predicted_label"]
        == result["true_label"]
        else "ERROR"
    )


    # --------------------------------------------------------
    # Текст
    # --------------------------------------------------------

    line1 = status


    line2 = (
        f"TRUE = {result['true_label']}    "
        f"PRED = {result['predicted_label']}"
    )


    line3 = (
        f"P(0) = {result['p0']:.3f}    "
        f"P(1) = {result['p1']:.3f}"
    )


    short_study = (
        result["study_id"][-12:]
    )


    line4 = (
        f"study ...{short_study}"
    )


    line5 = (
        f"image {result['index'] + 1}    "
        f"{Path(result['dicom_path']).name}"
    )


    # --------------------------------------------------------
    # Рисуем текст
    # --------------------------------------------------------

    draw_centered(
        draw,
        line1,
        IMAGE_HEIGHT + 8,
        font_large
    )


    draw_centered(
        draw,
        line2,
        IMAGE_HEIGHT + 40,
        font_small
    )


    draw_centered(
        draw,
        line3,
        IMAGE_HEIGHT + 68,
        font_small
    )


    draw_centered(
        draw,
        line4,
        IMAGE_HEIGHT + 96,
        font_small
    )


    draw_centered(
        draw,
        line5,
        IMAGE_HEIGHT + 122,
        font_small
    )


    # --------------------------------------------------------
    # Позиция карточки
    # --------------------------------------------------------

    column = (
        item_index
        % COLUMNS
    )


    row = (
        item_index
        // COLUMNS
    )


    x_sheet = (
        column
        * CARD_WIDTH
    )


    y_sheet = (
        row
        * CARD_HEIGHT
    )


    sheet.paste(
        card,
        (
            x_sheet,
            y_sheet
        )
    )


# ============================================================
# Сохраняем полный контактный лист
# ============================================================

output_path = (
    VISUALIZATION_DIR
    / "spine_study_error_analysis_v2.png"
)


sheet.save(
    output_path
)


# ============================================================
# Отдельно сохраняем только ошибки
# ============================================================

error_results = [
    result
    for result in results
    if result["predicted_label"]
    != result["true_label"]
]


if error_results:

    error_columns = 3

    error_rows = (
        len(error_results)
        + error_columns
        - 1
    ) // error_columns


    error_sheet = Image.new(
        "RGB",
        (
            error_columns
            * CARD_WIDTH,

            error_rows
            * CARD_HEIGHT
        ),
        "white"
    )


    for item_index, result in enumerate(
        error_results
    ):

        dataset_index = result["index"]

        sample = dataset[
            dataset_index
        ]


        image_array = (
            sample["image"]
            .squeeze(0)
            .numpy()
        )


        image_array = (
            image_array * 255
        ).clip(
            0,
            255
        ).astype("uint8")


        image = Image.fromarray(
            image_array,
            mode="L"
        ).convert("RGB")


        image.thumbnail(
            (
                CARD_WIDTH - 30,
                IMAGE_HEIGHT - 20
            )
        )


        image_area = Image.new(
            "RGB",
            (
                CARD_WIDTH,
                IMAGE_HEIGHT
            ),
            "black"
        )


        x_image = (
            CARD_WIDTH
            - image.width
        ) // 2


        y_image = (
            IMAGE_HEIGHT
            - image.height
        ) // 2


        image_area.paste(
            image,
            (
                x_image,
                y_image
            )
        )


        card = Image.new(
            "RGB",
            (
                CARD_WIDTH,
                CARD_HEIGHT
            ),
            "white"
        )


        card.paste(
            image_area,
            (0, 0)
        )


        draw = ImageDraw.Draw(
            card
        )


        line1 = "ERROR"


        line2 = (
            f"TRUE = {result['true_label']}    "
            f"PRED = {result['predicted_label']}"
        )


        line3 = (
            f"P(0) = {result['p0']:.3f}    "
            f"P(1) = {result['p1']:.3f}"
        )


        short_study = (
            result["study_id"][-12:]
        )


        line4 = (
            f"study ...{short_study}"
        )


        line5 = (
            f"image {result['index'] + 1}"
        )


        draw_centered(
            draw,
            line1,
            IMAGE_HEIGHT + 8,
            font_large
        )


        draw_centered(
            draw,
            line2,
            IMAGE_HEIGHT + 40,
            font_small
        )


        draw_centered(
            draw,
            line3,
            IMAGE_HEIGHT + 68,
            font_small
        )


        draw_centered(
            draw,
            line4,
            IMAGE_HEIGHT + 96,
            font_small
        )


        draw_centered(
            draw,
            line5,
            IMAGE_HEIGHT + 122,
            font_small
        )


        column = (
            item_index
            % error_columns
        )


        row = (
            item_index
            // error_columns
        )


        error_sheet.paste(
            card,
            (
                column
                * CARD_WIDTH,

                row
                * CARD_HEIGHT
            )
        )


    error_path = (
        VISUALIZATION_DIR
        / "spine_errors_only_v2.png"
    )


    error_sheet.save(
        error_path
    )


else:

    error_path = None


# ============================================================
# Финал
# ============================================================

print()
print("=" * 70)

print(
    "АНАЛИЗ ЗАВЕРШЁН"
)

print()

print(
    f"Полный контактный лист:\n"
    f"{output_path}"
)

if error_path:

    print()

    print(
        f"Только ошибки:\n"
        f"{error_path}"
    )

print("=" * 70)