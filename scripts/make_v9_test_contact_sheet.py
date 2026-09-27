from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "spine_quality_v9_exact_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "visualizations"
    / "v9_test_contact_sheet.png"
)


def normalize_image(arr):
    """
    Нормализация изображения примерно в том же стиле,
    который использовался при подготовке V9.
    """

    arr = arr.astype(np.float32)

    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 > p1:
        arr = np.clip(
            (arr - p1) / (p99 - p1),
            0,
            1
        )
    else:
        mn = arr.min()
        mx = arr.max()

        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)

    return arr


def read_dicom(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array


def resolve_dicom_path(path_string):
    """
    В prediction CSV путь записан относительно PROJECT_ROOT.
    Например:
    data\raw\training\...
    """

    path = Path(str(path_string))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main():

    print("=" * 80)
    print("V9 TEST CONTACT SHEET")
    print("=" * 80)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    # study_id приводим к строке
    manifest["study_id"] = (
        manifest["study_id"].astype(str)
    )

    predictions["study_id"] = (
        predictions["study_id"].astype(str)
    )

    # Приводим prediction-колонки к числам
    predictions["spine_quality"] = pd.to_numeric(
        predictions["spine_quality"],
        errors="coerce"
    )

    predictions["prediction"] = pd.to_numeric(
        predictions["prediction"],
        errors="coerce"
    )

    predictions["probability_class1"] = pd.to_numeric(
        predictions["probability_class1"],
        errors="coerce"
    )

    # Нужные экспертные признаки
    manifest_columns = [
        "study_id",
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "comment",
    ]

    labels = manifest[
        manifest_columns
    ].copy()

    labels["spine_position"] = pd.to_numeric(
        labels["spine_position"],
        errors="coerce"
    )

    labels["spine_axis"] = pd.to_numeric(
        labels["spine_axis"],
        errors="coerce"
    )

    labels["spine_artifacts"] = pd.to_numeric(
        labels["spine_artifacts"],
        errors="coerce"
    )

    # Один study = одна строка
    labels = labels.drop_duplicates(
        subset=["study_id"]
    )

    # Объединяем prediction V9
    df = predictions.merge(
        labels,
        on="study_id",
        how="left"
    )

    # Удаляем возможные дубликаты
    df = df.drop_duplicates(
        subset=["study_id"]
    ).copy()

    # TRUE
    df["true_label"] = (
        df["spine_quality"]
    )

    # PRED
    df["pred_label"] = (
        df["prediction"]
    )

    # Ошибка / правильный ответ
    df["error"] = (
        df["true_label"]
        != df["pred_label"]
    )

    # Сначала ошибки, затем правильные
    df = df.sort_values(
        [
            "error",
            "true_label",
            "study_id"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    print()
    print(
        "Исследований:",
        len(df)
    )

    print(
        "Ошибок:",
        int(df["error"].sum())
    )

    print()
    print("=" * 80)
    print("ПОРЯДОК ИЗОБРАЖЕНИЙ")
    print("=" * 80)

    for i, row in df.iterrows():

        result = (
            "ERROR"
            if row["error"]
            else "CORRECT"
        )

        print(
            f"{i + 1:02d}. "
            f"{result} | "
            f"{row['study_id']} | "
            f"TRUE={int(row['true_label'])} | "
            f"PRED={int(row['pred_label'])} | "
            f"P1={row['probability_class1']:.3f} | "
            f"POS={int(row['spine_position'])} | "
            f"AXIS={int(row['spine_axis'])} | "
            f"ART={int(row['spine_artifacts'])}"
        )

    # Размер contact sheet
    n = len(df)

    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, rows * 5)
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    # Сначала отключаем все оси
    for ax in axes:
        ax.axis("off")

    # Рисуем изображения
    for i, row in df.iterrows():

        ax = axes[i]

        dicom_path = resolve_dicom_path(
            row["dicom_path"]
        )

        try:

            arr = read_dicom(
                dicom_path
            )

            arr = normalize_image(
                arr
            )

            ax.imshow(
                arr,
                cmap="gray"
            )

        except Exception as e:

            ax.text(
                0.5,
                0.5,
                "Ошибка чтения DICOM\n\n"
                + str(e),
                ha="center",
                va="center",
                fontsize=10
            )

        true_label = int(
            row["true_label"]
        )

        pred_label = int(
            row["pred_label"]
        )

        probability = float(
            row["probability_class1"]
        )

        position = int(
            row["spine_position"]
        )

        axis_label = int(
            row["spine_axis"]
        )

        artifacts = int(
            row["spine_artifacts"]
        )

        if true_label == pred_label:
            result = "CORRECT"
        else:
            result = "ERROR"

        title = (
            f"{i + 1}. {result}\n"
            f"TRUE={true_label}   "
            f"PRED={pred_label}   "
            f"P1={probability:.3f}\n"
            f"POS={position}   "
            f"AXIS={axis_label}   "
            f"ART={artifacts}"
        )

        if pd.notna(row["comment"]):
            title += (
                f"\nКомментарий: {row['comment']}"
            )

        ax.set_title(
            title,
            fontsize=9
        )

    # Если ячеек больше, чем изображений
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "V9 — TEST: 15 уникальных исследований",
        fontsize=16
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("=" * 80)
    print("CONTACT SHEET СОЗДАН")
    print("=" * 80)

    print()
    print(
        "Файл:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("ГОТОВО")


if __name__ == "__main__":
    main()