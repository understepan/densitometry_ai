from pathlib import Path
import csv


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLITS_DIR = PROJECT_DIR / "data" / "splits"


# ============================================================
# ПОЛЯ, КОТОРЫЕ ПРОВЕРЯЕМ
# ============================================================

LABEL_COLUMNS = [
    "spine_position",
    "spine_axis",
    "spine_artifacts",
    "spine_quality",
    "right_hip_position",
    "right_hip_roi",
    "left_hip_position",
    "left_hip_roi",
    "right_hip_quality",
    "left_hip_quality",
]


# ============================================================
# ЧТЕНИЕ CSV
# ============================================================

def read_csv(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# НОРМАЛИЗАЦИЯ ЗНАЧЕНИЯ
# ============================================================

def normalize(value):

    if value is None:
        return ""

    value = str(value).strip()

    return value


# ============================================================
# СТАТИСТИКА ОДНОГО LABEL
# ============================================================

def label_stats(rows, column):

    count_0 = 0
    count_1 = 0
    count_blank = 0
    count_other = 0

    for row in rows:

        value = normalize(
            row.get(column, "")
        )

        if value == "":
            count_blank += 1

        elif value == "0":
            count_0 += 1

        elif value == "1":
            count_1 += 1

        else:
            count_other += 1

    return (
        count_0,
        count_1,
        count_blank,
        count_other
    )


# ============================================================
# СТАТИСТИКА SPLIT
# ============================================================

def print_split_statistics(
    split_name,
    rows
):

    print()
    print("=" * 80)
    print(split_name)
    print("=" * 80)

    studies = set(
        row["study_id"]
        for row in rows
    )

    spine_images = sum(
        1
        for row in rows
        if row["anatomy"] == "spine"
    )

    hip_images = sum(
        1
        for row in rows
        if row["anatomy"] == "hip"
    )

    labeled_source = {
        "excel": 0,
        "pca_pair": 0,
        "unknown": 0,
    }

    for row in rows:

        source = row.get(
            "label_source",
            "unknown"
        )

        if source not in labeled_source:
            source = "unknown"

        labeled_source[source] += 1

    print()
    print(
        f"Исследований: {len(studies)}"
    )

    print(
        f"Изображений: {len(rows)}"
    )

    print(
        f"Spine: {spine_images}"
    )

    print(
        f"Hip: {hip_images}"
    )

    print()

    print("Источник разметки:")

    print(
        f"  Excel:     {labeled_source['excel']}"
    )

    print(
        f"  PCA pair:  {labeled_source['pca_pair']}"
    )

    print(
        f"  Unknown:   {labeled_source['unknown']}"
    )

    print()

    print(
        f"{'LABEL':30} {'0':>8} {'1':>8} "
        f"{'BLANK':>8} {'OTHER':>8}"
    )

    print("-" * 80)

    for column in LABEL_COLUMNS:

        (
            count_0,
            count_1,
            count_blank,
            count_other
        ) = label_stats(
            rows,
            column
        )

        print(
            f"{column:30} "
            f"{count_0:8} "
            f"{count_1:8} "
            f"{count_blank:8} "
            f"{count_other:8}"
        )


# ============================================================
# ПРОВЕРКА ИССЛЕДОВАНИЙ МЕЖДУ SPLIT
# ============================================================

def check_leakage(
    train_rows,
    val_rows,
    test_rows
):

    train_ids = {
        row["study_id"]
        for row in train_rows
    }

    val_ids = {
        row["study_id"]
        for row in val_rows
    }

    test_ids = {
        row["study_id"]
        for row in test_rows
    }

    print()
    print("=" * 80)
    print("ПРОВЕРКА STUDY_ID")
    print("=" * 80)
    print()

    print(
        "TRAIN ∩ VALIDATION:",
        len(train_ids & val_ids)
    )

    print(
        "TRAIN ∩ TEST:",
        len(train_ids & test_ids)
    )

    print(
        "VALIDATION ∩ TEST:",
        len(val_ids & test_ids)
    )

    if (
        len(train_ids & val_ids) == 0
        and len(train_ids & test_ids) == 0
        and len(val_ids & test_ids) == 0
    ):

        print()
        print(
            "УТЕЧКИ МЕЖДУ SPLIT НЕТ."
        )

    else:

        print()
        print(
            "ОБНАРУЖЕНА УТЕЧКА!"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ LABELS")
    print("=" * 80)

    train_file = (
        SPLITS_DIR / "train.csv"
    )

    val_file = (
        SPLITS_DIR / "validation.csv"
    )

    test_file = (
        SPLITS_DIR / "test.csv"
    )

    # --------------------------------------------------------
    # Проверяем наличие файлов
    # --------------------------------------------------------

    for file_path in [
        train_file,
        val_file,
        test_file
    ]:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Файл не найден: {file_path}"
            )

    # --------------------------------------------------------
    # Читаем
    # --------------------------------------------------------

    train_rows = read_csv(
        train_file
    )

    val_rows = read_csv(
        val_file
    )

    test_rows = read_csv(
        test_file
    )

    # --------------------------------------------------------
    # Проверяем каждый split
    # --------------------------------------------------------

    print_split_statistics(
        "TRAIN",
        train_rows
    )

    print_split_statistics(
        "VALIDATION",
        val_rows
    )

    print_split_statistics(
        "TEST",
        test_rows
    )

    # --------------------------------------------------------
    # Проверяем отсутствие leakage
    # --------------------------------------------------------

    check_leakage(
        train_rows,
        val_rows,
        test_rows
    )

    # --------------------------------------------------------
    # Общая проверка
    # --------------------------------------------------------

    total = (
        len(train_rows)
        + len(val_rows)
        + len(test_rows)
    )

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)
    print()

    print(
        f"TRAIN:      {len(train_rows)}"
    )

    print(
        f"VALIDATION: {len(val_rows)}"
    )

    print(
        f"TEST:       {len(test_rows)}"
    )

    print(
        f"TOTAL:      {total}"
    )

    if total == 499:

        print()
        print(
            "ВСЕ 499 ИЗОБРАЖЕНИЙ УЧТЕНЫ."
        )

    else:

        print()
        print(
            "ВНИМАНИЕ: количество изображений "
            "не равно 499."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()