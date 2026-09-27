from pathlib import Path
import csv
from collections import defaultdict


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_DIR = Path(r"C:\hakaton\densitometry_ai")

PREDICTIONS_CSV = (
    PROJECT_DIR
    / "results"
    / "spine_quality_v3_predictions.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR
    / "results"
    / "spine_quality_v3_study_level.csv"
)


# ============================================================
# ЗАГРУЗКА PREDICTIONS
# ============================================================

def load_predictions():

    rows = []

    with open(
        PREDICTIONS_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# STUDY-LEVEL ANALYSIS
# ============================================================

def analyze_studies(rows):

    studies = defaultdict(list)

    for row in rows:
        studies[row["study_id"]].append(row)

    results = []

    for study_id, study_rows in studies.items():

        true_labels = [
            int(row["true_label"])
            for row in study_rows
        ]

        p0_values = [
            float(row["p0"])
            for row in study_rows
        ]

        p1_values = [
            float(row["p1"])
            for row in study_rows
        ]

        # Все изображения одного исследования должны
        # иметь одинаковую истинную метку.
        true_label = true_labels[0]

        # Study-level prediction:
        # средняя вероятность по изображениям.
        mean_p0 = sum(p0_values) / len(p0_values)
        mean_p1 = sum(p1_values) / len(p1_values)

        predicted_label = (
            1
            if mean_p1 >= mean_p0
            else 0
        )

        correct = (
            predicted_label == true_label
        )

        results.append({
            "study_id": study_id,
            "num_images": len(study_rows),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "mean_p0": mean_p0,
            "mean_p1": mean_p1,
            "correct": correct,
        })

    results.sort(
        key=lambda x: (
            x["correct"],
            x["true_label"],
            x["study_id"]
        )
    )

    return results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results):

    total = len(results)

    correct = sum(
        result["correct"]
        for result in results
    )

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for result in results:

        true_label = result["true_label"]
        predicted_label = result["predicted_label"]

        if true_label == 0 and predicted_label == 0:
            tn += 1

        elif true_label == 0 and predicted_label == 1:
            fp += 1

        elif true_label == 1 and predicted_label == 0:
            fn += 1

        elif true_label == 1 and predicted_label == 1:
            tp += 1

    precision_0 = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0
    )

    recall_0 = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1_0 = (
        2 * precision_0 * recall_0 /
        (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0
        else 0
    )

    precision_1 = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall_1 = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1_1 = (
        2 * precision_1 * recall_1 /
        (precision_1 + recall_1)
        if (precision_1 + recall_1) > 0
        else 0
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,

        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        fieldnames = [
            "study_id",
            "num_images",
            "true_label",
            "predicted_label",
            "mean_p0",
            "mean_p1",
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STUDY-LEVEL ANALYSIS — RESNET18 V3")
    print("=" * 70)

    rows = load_predictions()

    print()
    print(
        f"Строк predictions: {len(rows)}"
    )

    results = analyze_studies(rows)

    metrics = calculate_metrics(results)

    # --------------------------------------------------------
    # ОБЩИЕ РЕЗУЛЬТАТЫ
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STUDY-LEVEL RESULTS")
    print("=" * 70)

    print()
    print(
        f"Исследований: {metrics['total']}"
    )

    print(
        f"Правильно: "
        f"{metrics['correct']}/"
        f"{metrics['total']}"
    )

    print(
        f"Study-level Accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print()
    print("Confusion matrix:")

    print(
        f"True 0: "
        f"pred0 {metrics['tn']}, "
        f"pred1 {metrics['fp']}"
    )

    print(
        f"True 1: "
        f"pred0 {metrics['fn']}, "
        f"pred1 {metrics['tp']}"
    )

    # --------------------------------------------------------
    # CLASS 0
    # --------------------------------------------------------

    print()
    print("CLASS 0")

    print(
        f"Precision: "
        f"{metrics['precision_0'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{metrics['recall_0'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{metrics['f1_0'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CLASS 1
    # --------------------------------------------------------

    print()
    print("CLASS 1")

    print(
        f"Precision: "
        f"{metrics['precision_1'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{metrics['recall_1'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{metrics['f1_1'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # ПОДРОБНЫЙ СПИСОК
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL TEST STUDIES")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1
    ):

        status = (
            "CORRECT"
            if result["correct"]
            else "ERROR"
        )

        print()
        print(
            f"{index:02d}. "
            f"TRUE={result['true_label']} "
            f"PRED={result['predicted_label']} "
            f"P0={result['mean_p0']:.3f} "
            f"P1={result['mean_p1']:.3f} "
            f"{status}"
        )

        print(
            f"    study: "
            f"{result['study_id']}"
        )

        print(
            f"    images: "
            f"{result['num_images']}"
        )

    # --------------------------------------------------------
    # ТОЛЬКО ОШИБКИ
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERRORS ONLY")
    print("=" * 70)

    error_count = 0

    for result in results:

        if result["correct"]:
            continue

        error_count += 1

        print()
        print(
            f"{error_count:02d}. "
            f"TRUE={result['true_label']} "
            f"PRED={result['predicted_label']} "
            f"P0={result['mean_p0']:.3f} "
            f"P1={result['mean_p1']:.3f}"
        )

        print(
            f"    study: "
            f"{result['study_id']}"
        )

        print(
            f"    images: "
            f"{result['num_images']}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(results)

    print()
    print("=" * 70)
    print("ANALYSIS FINISHED")
    print("=" * 70)

    print()
    print(
        f"Результат сохранён:"
    )

    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()