from pathlib import Path
import csv


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR
    / "laterality_predictions.csv"
)


# ============================================================
# ЧТЕНИЕ
# ============================================================

def read_csv_file(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        return list(reader)


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():

    print("=" * 80)
    print("КОНТРОЛЬ PCA LATERALITY")
    print("=" * 80)
    print()

    rows = read_csv_file(INPUT_FILE)

    # --------------------------------------------------------
    # Группируем по исследованию
    # --------------------------------------------------------

    groups = {}

    for row in rows:

        study_id = row["study_id"]

        groups.setdefault(
            study_id,
            []
        ).append(row)

    # --------------------------------------------------------
    # Выводим каждую пару
    # --------------------------------------------------------

    separated = 0
    not_separated = 0

    print(
        f"Исследований с предсказаниями: "
        f"{len(groups)}"
    )

    print()

    for number, study_id in enumerate(
        sorted(groups),
        start=1
    ):

        pair = groups[study_id]

        if len(pair) != 2:
            continue

        first = pair[0]
        second = pair[1]

        first_angle = float(
            first["pca_angle"]
        )

        second_angle = float(
            second["pca_angle"]
        )

        first_side = first[
            "predicted_laterality"
        ]

        second_side = second[
            "predicted_laterality"
        ]

        if first_side != second_side:
            separated += 1
            status = "РАЗДЕЛЕНА"
        else:
            not_separated += 1
            status = "НЕ РАЗДЕЛЕНА"

        print("-" * 80)

        print(
            f"Пара #{number}"
        )

        print(
            f"study_id: {study_id}"
        )

        print()

        print(
            f"1) {first['dicom_path']}"
        )

        print(
            f"   угол PCA: "
            f"{first_angle:.2f}°"
        )

        print(
            f"   сторона: {first_side}"
        )

        print()

        print(
            f"2) {second['dicom_path']}"
        )

        print(
            f"   угол PCA: "
            f"{second_angle:.2f}°"
        )

        print(
            f"   сторона: {second_side}"
        )

        print()

        print(
            f"Результат: {status}"
        )

    # --------------------------------------------------------
    # Итог
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)
    print()

    print(
        f"Разделённых пар: "
        f"{separated}"
    )

    print(
        f"Неразделённых пар: "
        f"{not_separated}"
    )

    if separated + not_separated > 0:

        percentage = (
            separated
            / (separated + not_separated)
            * 100
        )

        print(
            f"Доля разделённых пар: "
            f"{percentage:.1f}%"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()