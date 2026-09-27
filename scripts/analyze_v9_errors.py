from pathlib import Path

import pandas as pd


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
    / "results"
    / "v9_test_error_analysis.csv"
)


def main():

    print("=" * 70)
    print("АНАЛИЗ ОШИБОК V9")
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    manifest["study_id"] = (
        manifest["study_id"]
        .astype(str)
    )

    predictions["study_id"] = (
        predictions["study_id"]
        .astype(str)
    )

    # Берём только нужные экспертные признаки
    expert_columns = [
        "study_id",
        "spine_quality",
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "comment"
    ]

    expert = (
        manifest[
            expert_columns
        ]
        .drop_duplicates(
            subset=["study_id"]
        )
    )

    result = predictions.merge(
        expert,
        on="study_id",
        how="left",
        suffixes=(
            "_prediction",
            "_manifest"
        )
    )

    # Оставляем только необходимые столбцы
    result = result[
        [
            "study_id",
            "spine_quality_prediction",
            "prediction",
            "probability_class1",
            "spine_position",
            "spine_axis",
            "spine_artifacts",
            "comment"
        ]
    ]

    result["correct"] = (
        result["spine_quality_prediction"]
        == result["prediction"]
    )

    result["error_type"] = "correct"

    result.loc[
        (~result["correct"])
        &
        (result["spine_quality_prediction"] == 0)
        &
        (result["prediction"] == 1),
        "error_type"
    ] = "false_positive"

    result.loc[
        (~result["correct"])
        &
        (result["spine_quality_prediction"] == 1)
        &
        (result["prediction"] == 0),
        "error_type"
    ] = "false_negative"

    result = result.sort_values(
        [
            "correct",
            "spine_quality_prediction",
            "probability_class1"
        ]
    )

    print()
    print(
        "ВСЕ TEST-ИЗОБРАЖЕНИЯ"
    )

    print()

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ОШИБКИ V9"
    )

    print()

    errors = result[
        ~result["correct"]
    ]

    if len(errors) == 0:

        print(
            "Ошибок нет."
        )

    else:

        print(
            errors.to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 70
    )

    print(
        "СВОДКА"
    )

    print()

    print(
        "Всего:",
        len(result)
    )

    print(
        "Правильно:",
        result["correct"].sum()
    )

    print(
        "Ошибок:",
        (~result["correct"]).sum()
    )

    print()

    print(
        "False Positive:",
        (
            result["error_type"]
            == "false_positive"
        ).sum()
    )

    print(
        "False Negative:",
        (
            result["error_type"]
            == "false_negative"
        ).sum()
    )

    print()

    print(
        "Сочетания экспертных признаков:"
    )

    summary = (
        result
        .groupby(
            [
                "spine_position",
                "spine_axis",
                "spine_artifacts"
            ]
        )
        .agg(
            count=(
                "study_id",
                "count"
            ),
            errors=(
                "correct",
                lambda x: (~x).sum()
            )
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        "Файл сохранён:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()