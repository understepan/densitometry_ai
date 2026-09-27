from pathlib import Path
import sys
from collections import defaultdict, Counter
import csv


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# Файлы split
# ============================================================

SPLITS = {
    "TRAIN": PROJECT_DIR / "data" / "splits" / "train.csv",
    "VALIDATION": PROJECT_DIR / "data" / "splits" / "validation.csv",
    "TEST": PROJECT_DIR / "data" / "splits" / "test.csv",
}


# ============================================================
# Проверка одного split
# ============================================================

def check_split(name, csv_path):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)


    # --------------------------------------------------------
    # Оставляем только spine
    # --------------------------------------------------------

    spine_rows = [
        row
        for row in rows
        if row["anatomy"] == "spine"
        and row["spine_quality"] in {"0", "1"}
    ]


    # --------------------------------------------------------
    # Группируем по study_id
    # --------------------------------------------------------

    studies = defaultdict(list)

    for row in spine_rows:

        studies[row["study_id"]].append(
            row
        )


    print(
        f"Всего строк в split: {len(rows)}"
    )

    print(
        f"Spine изображений с label: "
        f"{len(spine_rows)}"
    )

    print(
        f"Уникальных spine-исследований: "
        f"{len(studies)}"
    )


    # --------------------------------------------------------
    # Проверяем labels внутри каждого study
    # --------------------------------------------------------

    inconsistent = []

    for study_id, study_rows in studies.items():

        labels = {
            row["spine_quality"]
            for row in study_rows
        }

        if len(labels) > 1:

            inconsistent.append(
                (
                    study_id,
                    study_rows,
                    labels
                )
            )


    print()
    print(
        f"Исследований с разными label "
        f"внутри одного study: "
        f"{len(inconsistent)}"
    )


    # --------------------------------------------------------
    # Показываем проблемные исследования
    # --------------------------------------------------------

    if inconsistent:

        print()
        print("ПРОБЛЕМНЫЕ ИССЛЕДОВАНИЯ:")
        print()

        for (
            study_id,
            study_rows,
            labels
        ) in inconsistent:

            print(
                f"study_id: {study_id}"
            )

            print(
                f"labels: {sorted(labels)}"
            )

            for row in study_rows:

                print(
                    f"  label={row['spine_quality']} "
                    f"path={row['dicom_path']}"
                )

            print()


    # --------------------------------------------------------
    # Количество изображений на исследование
    # --------------------------------------------------------

    image_counts = Counter(
        len(rows_for_study)
        for rows_for_study
        in studies.values()
    )


    print()
    print(
        "Количество spine-изображений "
        "на одно исследование:"
    )

    for count in sorted(image_counts):

        number_of_studies = image_counts[count]

        print(
            f"  {count} изображений: "
            f"{number_of_studies} исследований"
        )


    # --------------------------------------------------------
    # Распределение label по исследованиям
    # --------------------------------------------------------

    study_labels = Counter()

    for study_id, study_rows in studies.items():

        label = study_rows[0]["spine_quality"]

        study_labels[label] += 1


    print()
    print(
        "Распределение label "
        "по исследованиям:"
    )

    print(
        f"  label 0: "
        f"{study_labels['0']}"
    )

    print(
        f"  label 1: "
        f"{study_labels['1']}"
    )


    return inconsistent


# ============================================================
# Запускаем проверку
# ============================================================

all_inconsistent = []

for split_name, split_path in SPLITS.items():

    inconsistent = check_split(
        split_name,
        split_path
    )

    all_inconsistent.extend(
        [
            (
                split_name,
                item
            )
            for item in inconsistent
        ]
    )


# ============================================================
# Итог
# ============================================================

print()
print("=" * 70)
print("ИТОГОВАЯ ПРОВЕРКА")
print("=" * 70)

if all_inconsistent:

    print()
    print(
        "ОБНАРУЖЕНЫ НЕСОГЛАСОВАННЫЕ LABEL."
    )

    print(
        f"Количество проблемных случаев: "
        f"{len(all_inconsistent)}"
    )

else:

    print()
    print(
        "Во всех split каждый study имеет "
        "одинаковый spine_quality label "
        "для всех своих spine-изображений."
    )

print()
print("ПРОВЕРКА ЗАВЕРШЕНА.")