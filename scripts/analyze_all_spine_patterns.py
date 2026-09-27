from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)


def main():

    print("=" * 70)
    print("ПОЛНЫЙ АНАЛИЗ PATTERN ДЛЯ SPINE")
    print("=" * 70)

    df = pd.read_csv(
        MANIFEST_PATH
    )

    numeric = [
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "spine_quality"
    ]

    for col in numeric:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Только spine
    df = df[
        df["anatomy"] == "spine"
    ].copy()

    # Только строки с полной разметкой
    df = df[
        df[numeric]
        .notna()
        .all(axis=1)
    ].copy()

    # Один уникальный PixelData на исследование
    # Для этого используем путь + study.
    # Дубликаты одного изображения нам здесь не нужны.
    df = df.drop_duplicates(
        subset=["study_id"]
    ).copy()

    print()
    print(
        "Уникальных исследований:",
        len(df)
    )

    print()
    print(
        "FINAL:"
    )

    print(
        df["spine_quality"]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # Все комбинации
    # ========================================================

    print()
    print("=" * 70)
    print("КОМБИНАЦИИ POSITION + AXIS + ARTIFACTS")
    print("=" * 70)

    combinations = (
        df
        .groupby(
            [
                "spine_position",
                "spine_axis",
                "spine_artifacts"
            ]
        )[
            "spine_quality"
        ]
        .value_counts()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    print(
        combinations.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL=1
    # ========================================================

    print()
    print("=" * 70)
    print("ТОЛЬКО FINAL = 1")
    print("=" * 70)

    final1 = df[
        df["spine_quality"] == 1
    ]

    print(
        "Количество:",
        len(final1)
    )

    print()

    print(
        "Position:"
    )

    print(
        final1[
            "spine_position"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Axis:"
    )

    print(
        final1[
            "spine_axis"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Artifacts:"
    )

    print(
        final1[
            "spine_artifacts"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # FINAL=0
    # ========================================================

    print()
    print("=" * 70)
    print("ТОЛЬКО FINAL = 0")
    print("=" * 70)

    final0 = df[
        df["spine_quality"] == 0
    ]

    print(
        "Количество:",
        len(final0)
    )

    print()

    print(
        "Position:"
    )

    print(
        final0[
            "spine_position"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Axis:"
    )

    print(
        final0[
            "spine_axis"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Artifacts:"
    )

    print(
        final0[
            "spine_artifacts"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # OR RULE
    # ========================================================

    rule_prediction = (
        (df["spine_position"] == 1)
        |
        (df["spine_axis"] == 1)
        |
        (df["spine_artifacts"] == 1)
    ).astype(int)

    accuracy = (
        rule_prediction
        == df["spine_quality"]
    ).mean()

    print()
    print("=" * 70)
    print("RULE: POSITION OR AXIS OR ARTIFACTS")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    errors = df[
        rule_prediction
        != df["spine_quality"]
    ].copy()

    print(
        "Ошибок:",
        len(errors)
    )

    if len(errors) > 0:

        print()

        print(
            errors[
                [
                    "study_id",
                    "spine_position",
                    "spine_axis",
                    "spine_artifacts",
                    "spine_quality",
                    "comment"
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # СТРАННЫЕ CASES
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL=1 ПРИ ВСЕХ ТРЁХ ПРИЗНАКАХ = 0")
    print("=" * 70)

    strange = df[
        (df["spine_quality"] == 1)
        &
        (df["spine_position"] == 0)
        &
        (df["spine_axis"] == 0)
        &
        (df["spine_artifacts"] == 0)
    ]

    print(
        "Количество:",
        len(strange)
    )

    if len(strange) > 0:

        print(
            strange[
                [
                    "study_id",
                    "spine_quality",
                    "comment"
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()