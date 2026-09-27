from pathlib import Path
import csv
import random


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "splits"
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# ЧТЕНИЕ
# ============================================================

def read_csv(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(csv.DictReader(f))


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("РАЗБИЕНИЕ DATASET ПО ИССЛЕДОВАНИЯМ")
    print("=" * 80)
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Не найден файл: {INPUT_FILE}"
        )

    rows = read_csv(INPUT_FILE)

    print(
        f"Всего изображений: {len(rows)}"
    )

    # --------------------------------------------------------
    # Собираем уникальные исследования
    # --------------------------------------------------------

    studies = sorted(
        set(
            row["study_id"]
            for row in rows
        )
    )

    print(
        f"Уникальных исследований: {len(studies)}"
    )

    print()

    # --------------------------------------------------------
    # Проверяем количество
    # --------------------------------------------------------

    if len(studies) != 100:

        raise ValueError(
            "Ожидалось 100 исследований, "
            f"получено {len(studies)}"
        )

    # --------------------------------------------------------
    # Перемешиваем исследования
    # --------------------------------------------------------

    random.seed(SEED)

    random.shuffle(studies)

    # --------------------------------------------------------
    # Размеры
    # --------------------------------------------------------

    total = len(studies)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    train_studies = set(
        studies[:train_count]
    )

    val_studies = set(
        studies[
            train_count:
            train_count + val_count
        ]
    )

    test_studies = set(
        studies[
            train_count + val_count:
        ]
    )

    # --------------------------------------------------------
    # Проверки
    # --------------------------------------------------------

    assert len(train_studies) == train_count
    assert len(val_studies) == val_count
    assert len(test_studies) == test_count

    assert (
        train_studies.isdisjoint(
            val_studies
        )
    )

    assert (
        train_studies.isdisjoint(
            test_studies
        )
    )

    assert (
        val_studies.isdisjoint(
            test_studies
        )
    )

    # --------------------------------------------------------
    # Разделяем изображения
    # --------------------------------------------------------

    train_rows = []
    val_rows = []
    test_rows = []

    for row in rows:

        study_id = row["study_id"]

        if study_id in train_studies:

            train_rows.append(row)

        elif study_id in val_studies:

            val_rows.append(row)

        elif study_id in test_studies:

            test_rows.append(row)

        else:

            raise ValueError(
                f"Исследование не найдено "
                f"в split: {study_id}"
            )

    # --------------------------------------------------------
    # Сохраняем
    # --------------------------------------------------------

    train_file = (
        OUTPUT_DIR
        / "train.csv"
    )

    val_file = (
        OUTPUT_DIR
        / "validation.csv"
    )

    test_file = (
        OUTPUT_DIR
        / "test.csv"
    )

    write_csv(
        train_file,
        train_rows
    )

    write_csv(
        val_file,
        val_rows
    )

    write_csv(
        test_file,
        test_rows
    )

    # --------------------------------------------------------
    # Сохраняем списки study_id
    # --------------------------------------------------------

    def write_study_ids(
        path,
        study_ids
    ):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            for study_id in sorted(
                study_ids
            ):

                f.write(
                    study_id + "\n"
                )

    write_study_ids(
        OUTPUT_DIR / "train_studies.txt",
        train_studies
    )

    write_study_ids(
        OUTPUT_DIR / "validation_studies.txt",
        val_studies
    )

    write_study_ids(
        OUTPUT_DIR / "test_studies.txt",
        test_studies
    )

    # --------------------------------------------------------
    # Статистика
    # --------------------------------------------------------

    print("=" * 80)
    print("SPLIT")
    print("=" * 80)
    print()

    print(
        f"TRAIN:      {len(train_studies)} исследований, "
        f"{len(train_rows)} изображений"
    )

    print(
        f"VALIDATION: {len(val_studies)} исследований, "
        f"{len(val_rows)} изображений"
    )

    print(
        f"TEST:       {len(test_studies)} исследований, "
        f"{len(test_rows)} изображений"
    )

    print()

    print("=" * 80)
    print("ПРОВЕРКА УТЕЧКИ")
    print("=" * 80)
    print()

    train_ids = set(
        row["study_id"]
        for row in train_rows
    )

    val_ids = set(
        row["study_id"]
        for row in val_rows
    )

    test_ids = set(
        row["study_id"]
        for row in test_rows
    )

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

    print()

    if (
        len(train_ids & val_ids) == 0
        and len(train_ids & test_ids) == 0
        and len(val_ids & test_ids) == 0
        and (
            len(train_rows)
            + len(val_rows)
            + len(test_rows)
        ) == 499
    ):

        print(
            "ВСЕ ПРОВЕРКИ SPLIT ПРОЙДЕНЫ."
        )

    else:

        print(
            "ОШИБКА В SPLIT."
        )

    print()

    print(
        "Файлы сохранены в:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 80)


if __name__ == "__main__":
    main()