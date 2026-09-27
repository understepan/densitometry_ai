from pathlib import Path
import sys
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing.dataset import SpineQualityDataset


# =========================
# Настройки
# =========================

CSV_FILE = PROJECT_DIR / "data" / "splits" / "train.csv"

# Сколько изображений каждого класса показать
N_PER_CLASS = 3


# =========================
# Загружаем Dataset
# =========================

dataset = SpineQualityDataset(
    csv_file=CSV_FILE,
    project_dir=PROJECT_DIR,
    image_size=224,
)

print(f"Всего изображений в Dataset: {len(dataset)}")


# =========================
# Находим изображения
# каждого класса
# =========================

class_0 = []
class_1 = []

for index in range(len(dataset)):
    sample = dataset[index]

    label = int(sample["label"])

    if label == 0 and len(class_0) < N_PER_CLASS:
        class_0.append((index, sample))

    elif label == 1 and len(class_1) < N_PER_CLASS:
        class_1.append((index, sample))

    if len(class_0) == N_PER_CLASS and len(class_1) == N_PER_CLASS:
        break


samples = class_0 + class_1


# =========================
# Создаём изображение
# =========================

fig, axes = plt.subplots(
    2,
    N_PER_CLASS,
    figsize=(12, 8)
)

fig.suptitle(
    "Проверка preprocessing Dataset — Spine Quality",
    fontsize=16
)


for ax, (index, sample) in zip(axes.flatten(), samples):

    image = sample["image"].squeeze(0).numpy()

    label = sample["label"].item()
    study_id = sample["study_id"]
    dicom_path = sample["dicom_path"]

    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        f"label = {label}\n"
        f"study = {study_id[:18]}..."
    )

    ax.axis("off")


plt.tight_layout()


# =========================
# Сохраняем результат
# =========================

output_dir = PROJECT_DIR / "visualizations"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "spine_quality_dataset_preview.png"

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print()
print(f"Предпросмотр сохранён:")
print(output_file)

plt.show()