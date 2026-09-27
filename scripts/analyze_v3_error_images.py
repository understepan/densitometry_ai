from pathlib import Path
import csv


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(
    r"C:\hakaton\densitometry_ai"
)

PREDICTIONS_FILE = (
    PROJECT_DIR
    / "results"
    / "spine_quality_v3_predictions.csv"
)

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)


# 4 ошибки V3 + 2 правильных случая
TARGET_STUDIES = {
    "2.25.102755089973625799055786462646268820450": "ERROR",
    "2.25.114887542067602662452692698814728642117": "ERROR",
    "2.25.269648774413746842548908998641183395622": "ERROR",
    "2.25.338439598607961312040815816824093323492": "ERROR",
    "2.25.29638026465755240167985424461715024381": "CORRECT",
    "2.25.48906424954135283790354958243772558962": "CORRECT",
}


# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# ============================================================

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_column(columns, candidates):

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


# ============================================================
# ЧТЕНИЕ PREDICTIONS
# ============================================================

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Не найден файл:\n{PREDICTIONS_FILE}"
    )


with open(
    PREDICTIONS_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    prediction_rows = list(reader)

    prediction_columns = reader.fieldnames or []


print("=" * 80)
print("АНАЛИЗ V3 — КОНКРЕТНЫЕ ОШИБКИ И ПРАВИЛЬНЫЕ СЛУЧАИ")
print("=" * 80)

print()
print(
    f"Файл predictions: {PREDICTIONS_FILE}"
)

print(
    f"Строк predictions: {len(prediction_rows)}"
)

print()
print("Колонки predictions:")

for column in prediction_columns:
    print(f"  - {column}")


# ============================================================
# ОПРЕДЕЛЯЕМ НАЗВАНИЯ КОЛОНОК
# ============================================================

study_column = find_column(
    prediction_columns,
    [
        "study_id",
        "study",
    ]
)

dicom_column = find_column(
    prediction_columns,
    [
        "dicom_path",
        "path",
        "file",
        "dicom",
    ]
)

true_column = find_column(
    prediction_columns,
    [
        "true_label",
        "true",
        "label",
        "target",
        "spine_quality",
    ]
)

pred_column = find_column(
    prediction_columns,
    [
        "predicted_label",
        "pred",
        "prediction",
    ]
)

p0_column = find_column(
    prediction_columns,
    [
        "prob_0",
        "p0",
        "P0",
        "probability_0",
    ]
)

p1_column = find_column(
    prediction_columns,
    [
        "prob_1",
        "p1",
        "P1",
        "probability_1",
    ]
)


required = {
    "study": study_column,
    "dicom": dicom_column,
    "true": true_column,
    "pred": pred_column,
    "p0": p0_column,
    "p1": p1_column,
}


print()
print("Используемые колонки:")

for name, column in required.items():
    print(
        f"  {name:>5}: {column}"
    )


missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:

    print()
    print(
        "ОШИБКА: не найдены колонки:"
    )

    for name in missing:
        print(
            f"  - {name}"
        )

    raise ValueError(
        "Не удалось определить структуру predictions CSV."
    )


# ============================================================
# ЧТЕНИЕ MANIFEST
# ============================================================

manifest_by_path = {}


if MANIFEST_FILE.exists():

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        manifest_rows = list(reader)

    print()
    print(
        f"Строк labeled_manifest: "
        f"{len(manifest_rows)}"
    )

    for row in manifest_rows:

        path = row.get(
            "dicom_path",
            ""
        )

        if path:
            manifest_by_path[path] = row

else:

    print()
    print(
        "ВНИМАНИЕ: labeled_manifest.csv "
        "не найден."
    )

    print(
        "Размеры/anatomy будут недоступны."
    )


# ============================================================
# СОБИРАЕМ ЦЕЛЕВЫЕ СТРОКИ
# ============================================================

selected = []

for row in prediction_rows:

    study_id = row[study_column]

    if study_id not in TARGET_STUDIES:
        continue

    p0 = to_float(
        row[p0_column]
    )

    p1 = to_float(
        row[p1_column]
    )

    true_label = row[true_column]
    pred_label = row[pred_column]

    dicom_path = row[dicom_column]

    manifest = manifest_by_path.get(
        dicom_path,
        {}
    )

    selected.append(
        {
            "study_id": study_id,
            "status": TARGET_STUDIES[study_id],
            "dicom_path": dicom_path,
            "true": true_label,
            "pred": pred_label,
            "p0": p0,
            "p1": p1,
            "anatomy": manifest.get(
                "anatomy",
                "unknown"
            ),
            "rows": manifest.get(
                "rows",
                "?"
            ),
            "columns": manifest.get(
                "columns",
                "?"
            ),
        }
    )


# ============================================================
# СОРТИРОВКА
# ============================================================

selected.sort(
    key=lambda x: (
        x["status"],
        x["study_id"],
        -(x["p1"] or 0)
    )
)


# ============================================================
# ВЫВОД
# ============================================================

print()
print("=" * 80)
print(
    f"НАЙДЕНО ЦЕЛЕВЫХ ИЗОБРАЖЕНИЙ: "
    f"{len(selected)}"
)
print("=" * 80)


current_study = None

for item in selected:

    study_id = item["study_id"]

    if study_id != current_study:

        current_study = study_id

        print()
        print("-" * 80)

        print(
            f"{item['status']}"
        )

        print(
            f"Study: {study_id}"
        )

        print(
            f"TRUE={item['true']} "
            f"PRED={item['pred']}"
        )

        print("-" * 80)

    print(
        f"DICOM: {Path(item['dicom_path']).name}"
    )

    print(
        f"  path     : {item['dicom_path']}"
    )

    print(
        f"  anatomy  : {item['anatomy']}"
    )

    print(
        f"  size     : "
        f"{item['rows']} x {item['columns']}"
    )

    print(
        f"  TRUE     : {item['true']}"
    )

    print(
        f"  PRED     : {item['pred']}"
    )

    if item["p0"] is not None:
        print(
            f"  P(class0): "
            f"{item['p0']:.4f}"
        )

    if item["p1"] is not None:
        print(
            f"  P(class1): "
            f"{item['p1']:.4f}"
        )

    if (
        item["p0"] is not None
        and item["p1"] is not None
    ):

        margin = abs(
            item["p1"] - item["p0"]
        )

        print(
            f"  margin   : "
            f"{margin:.4f}"
        )

    print()


# ============================================================
# СВОДКА
# ============================================================

print()
print("=" * 80)
print("СВОДКА")
print("=" * 80)

for study_id, status in TARGET_STUDIES.items():

    rows = [
        item
        for item in selected
        if item["study_id"] == study_id
    ]

    print()
    print(
        f"{status}: "
        f"{study_id}"
    )

    print(
        f"Изображений в predictions: "
        f"{len(rows)}"
    )

    spine_rows = [
        item
        for item in rows
        if item["anatomy"] == "spine"
    ]

    print(
        f"Spine: "
        f"{len(spine_rows)}"
    )

    if spine_rows:

        p1_values = [
            item["p1"]
            for item in spine_rows
            if item["p1"] is not None
        ]

        if p1_values:

            average_p1 = (
                sum(p1_values)
                / len(p1_values)
            )

            print(
                f"Средний P(class1): "
                f"{average_p1:.4f}"
            )

            print(
                f"Min P(class1): "
                f"{min(p1_values):.4f}"
            )

            print(
                f"Max P(class1): "
                f"{max(p1_values):.4f}"
            )


print()
print("=" * 80)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)