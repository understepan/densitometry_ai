from pathlib import Path
import math

import matplotlib.pyplot as plt
import pydicom


# ============================================================
# ПУТИ ПРОЕКТА
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "visualizations"
    / "contact_sheets"
)


# ============================================================
# КАКИЕ ИССЛЕДОВАНИЯ ПОКА СМОТРИМ
# ============================================================

SELECTED_STUDY_NUMBERS = [
    1,
    2,
    12,
    20,
    28,
    43,
    49,
    50,
    60,
    62,
    78,
    100,
]


# ============================================================
# ПРЕДВАРИТЕЛЬНАЯ ЭВРИСТИКА
# ============================================================

def guess_anatomy(rows, columns):
    """
    Только предварительная эвристика для визуальной проверки.

    В нашем наборе данных ранее наблюдалось:
    - ширина около 300 -> часто позвоночник
    - ширина около 280 -> часто бедро

    Это НЕ окончательная разметка.
    """

    if columns == 300:
        return "spine"

    if columns == 280:
        return "hip"

    return "unknown"


# ============================================================
# НОРМАЛИЗАЦИЯ ИЗОБРАЖЕНИЯ
# ============================================================

def normalize_image(array):
    """
    Приводим изображение к диапазону 0..1
    только для удобного отображения.
    """

    array = array.astype("float32")

    minimum = array.min()
    maximum = array.max()

    if maximum <= minimum:
        return array * 0

    normalized = (array - minimum) / (maximum - minimum)

    return normalized


# ============================================================
# ПОЛУЧЕНИЕ КОРОТКОГО UID
# ============================================================

def short_uid(uid):
    if not uid:
        return "no_uid"

    uid = str(uid)

    if len(uid) <= 12:
        return uid

    return "..." + uid[-9:]


# ============================================================
# СОЗДАНИЕ CONTACT SHEET
# ============================================================

def create_contact_sheet(study_number, study_dir):

    dicom_files = sorted(study_dir.rglob("*.dcm"))

    if not dicom_files:
        print(
            f"[ПРЕДУПРЕЖДЕНИЕ] "
            f"Нет DICOM: {study_dir.name}"
        )
        return

    images = []

    print()
    print("=" * 80)
    print(f"Исследование #{study_number}")
    print(f"study_id: {study_dir.name}")
    print(f"DICOM: {len(dicom_files)}")
    print("=" * 80)

    for dicom_path in dicom_files:

        try:
            dataset = pydicom.dcmread(dicom_path)

            array = dataset.pixel_array

            # На случай многомерного изображения
            if array.ndim > 2:
                array = array[0]

            rows = int(getattr(dataset, "Rows", array.shape[0]))
            columns = int(
                getattr(dataset, "Columns", array.shape[1])
            )

            anatomy = guess_anatomy(rows, columns)

            series_uid = str(
                getattr(dataset, "SeriesInstanceUID", "")
            )

            image_uid = str(
                getattr(dataset, "SOPInstanceUID", "")
            )

            instance_number = getattr(
                dataset,
                "InstanceNumber",
                "?"
            )

            relative_path = dicom_path.relative_to(study_dir)

            images.append(
                {
                    "path": dicom_path,
                    "relative_path": str(relative_path),
                    "array": normalize_image(array),
                    "rows": rows,
                    "columns": columns,
                    "anatomy": anatomy,
                    "series_uid": series_uid,
                    "image_uid": image_uid,
                    "instance_number": instance_number,
                }
            )

            print(
                f"{dicom_path.name:20s} "
                f"{rows:4d}x{columns:<4d} "
                f"{anatomy:8s} "
                f"Instance={instance_number}"
            )

        except Exception as error:

            print(
                f"[ОШИБКА] {dicom_path}"
            )

            print(
                f"         {error}"
            )

    if not images:
        print(
            "[ПРЕДУПРЕЖДЕНИЕ] "
            "Не удалось загрузить изображения."
        )
        return

    # --------------------------------------------------------
    # Сетка
    # --------------------------------------------------------

    number_of_images = len(images)

    columns_in_sheet = 4

    rows_in_sheet = math.ceil(
        number_of_images / columns_in_sheet
    )

    figure_width = 16
    figure_height = max(
        4,
        rows_in_sheet * 4.5
    )

    fig, axes = plt.subplots(
        rows_in_sheet,
        columns_in_sheet,
        figsize=(figure_width, figure_height)
    )

    # Если всего одно изображение,
    # axes не является массивом.
    if number_of_images == 1:
        axes = [axes]

    else:
        axes = list(
            axes.flat
            if hasattr(axes, "flat")
            else axes
        )

    # --------------------------------------------------------
    # Рисуем изображения
    # --------------------------------------------------------

    for index, image_info in enumerate(images):

        ax = axes[index]

        ax.imshow(
            image_info["array"],
            cmap="gray"
        )

        ax.axis("off")

        title = (
            f"#{index + 1} | "
            f"{image_info['anatomy']}\n"
            f"{image_info['rows']}x"
            f"{image_info['columns']} | "
            f"Inst={image_info['instance_number']}\n"
            f"{image_info['relative_path']}"
        )

        ax.set_title(
            title,
            fontsize=8
        )

    # --------------------------------------------------------
    # Пустые клетки
    # --------------------------------------------------------

    for index in range(number_of_images, len(axes)):

        axes[index].axis("off")

    # --------------------------------------------------------
    # Заголовок
    # --------------------------------------------------------

    fig.suptitle(
        (
            f"Study #{study_number}\n"
            f"{study_dir.name}\n"
            f"DICOM images: {number_of_images}"
        ),
        fontsize=14
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Сохраняем
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"study_{study_number:03d}.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print(f"Сохранено:")
    print(output_file)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 80)
    print("СОЗДАНИЕ CONTACT SHEETS")
    print("=" * 80)

    if not STUDIES_DIR.exists():

        raise FileNotFoundError(
            f"Не найдена папка исследований:\n"
            f"{STUDIES_DIR}"
        )

    study_dirs = sorted(
        [
            path
            for path in STUDIES_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print()
    print(
        f"Всего исследований: {len(study_dirs)}"
    )

    print(
        f"Выбрано для визуальной проверки: "
        f"{len(SELECTED_STUDY_NUMBERS)}"
    )

    print()

    # Нумерация исследований такая же,
    # как в нашем предыдущем отчёте:
    # 1-я папка = исследование #1,
    # 2-я папка = исследование #2 и т.д.

    for study_number in SELECTED_STUDY_NUMBERS:

        index = study_number - 1

        if index < 0 or index >= len(study_dirs):

            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] "
                f"Исследование #{study_number} "
                f"не существует."
            )

            continue

        study_dir = study_dirs[index]

        create_contact_sheet(
            study_number,
            study_dir
        )

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)

    print()
    print("Contact sheets находятся здесь:")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()