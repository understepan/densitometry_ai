from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "label_analysis"
    / "spine_labeled_only.csv"
)


def main():

    print("=" * 70)
    print("RULE-BASED SPINE BASELINE")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH
    )

    # --------------------------------------------------------
    # RULE
    # --------------------------------------------------------

    df["rule_prediction"] = (
        (df["position"] == 1)
        |
        (df["axis"] == 1)
        |
        (df["artifacts"] == 1)
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    true = df["final"].astype(int)
    pred = df["rule_prediction"]

    tp = (
        (true == 1)
        & (pred == 1)
    ).sum()

    tn = (
        (true == 0)
        & (pred == 0)
    ).sum()

    fp = (
        (true == 0)
        & (pred == 1)
    ).sum()

    fn = (
        (true == 1)
        & (pred == 0)
    ).sum()

    accuracy = (
        (tp + tn)
        / len(df)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print()
    print("RULE:")
    print(
        "position=1 OR axis=1 OR artifacts=1"
    )

    print()
    print(
        f"Всего: {len(df)}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall: {recall * 100:.2f}%"
    )

    print(
        f"F1: {f1 * 100:.2f}%"
    )

    print()
    print("Confusion matrix:")
    print()
    print("                 Pred 0    Pred 1")
    print(
        f"True 0          "
        f"{tn:8d}  "
        f"{fp:8d}"
    )
    print(
        f"True 1          "
        f"{fn:8d}  "
        f"{tp:8d}"
    )

    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    errors = df[
        df["final"]
        !=
        df["rule_prediction"]
    ].copy()

    print()
    print("-" * 70)
    print("ОШИБКИ ПРАВИЛА")
    print("-" * 70)

    print(
        f"Количество: {len(errors)}"
    )

    if len(errors) > 0:

        print()

        print(
            errors[
                [
                    "study_id",
                    "position",
                    "axis",
                    "artifacts",
                    "final",
                    "rule_prediction",
                    "comment"
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # BY SPLIT
    # --------------------------------------------------------

    split_dir = (
        PROJECT_ROOT
        / "data"
        / "splits"
    )

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        split_path = (
            split_dir
            / f"{split_name}.csv"
        )

        split_df = pd.read_csv(
            split_path
        )

        study_ids = set(
            split_df["study_id"]
            .astype(str)
        )

        part = df[
            df["study_id"]
            .astype(str)
            .isin(study_ids)
        ].copy()

        if len(part) == 0:
            continue

        y_true = part["final"].astype(int)
        y_pred = part["rule_prediction"]

        acc = (
            y_true == y_pred
        ).mean()

        print()
        print(
            f"{split_name.upper()}:"
        )

        print(
            f"  исследований: {len(part)}"
        )

        print(
            f"  accuracy: {acc * 100:.2f}%"
        )

    print()
    print("=" * 70)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    main()