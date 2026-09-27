from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = (
    PROJECT_ROOT
    / "results"
    / "label_analysis"
    / "spine_label_analysis.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


def main():

    print("=" * 70)
    print("ПРОВЕРКА LABELS ДЛЯ MULTI-TASK")
    print("=" * 70)

    df = pd.read_csv(LABELS_PATH)

    df["study_id"] = (
        df["study_id"]
        .astype(str)
        .str.strip()
    )

    print()
    print(
        f"Всего исследований: {len(df)}"
    )

    columns = [
        "position",
        "axis",
        "artifacts",
        "final"
    ]

    # ========================================================
    # ALL DATA
    # ========================================================

    print()
    print("-" * 70)
    print("ВСЕ ДАННЫЕ")
    print("-" * 70)

    for col in columns:

        valid = df[col].isin([0, 1])

        print()
        print(col)

        print(
            f"  размечено: {valid.sum()}"
        )

        print(
            f"  пропущено: {(~valid).sum()}"
        )

        print(
            f"  class 0: {(df.loc[valid, col] == 0).sum()}"
        )

        print(
            f"  class 1: {(df.loc[valid, col] == 1).sum()}"
        )

    # ========================================================
    # SPLITS
    # ========================================================

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        path = (
            SPLIT_DIR
            / f"{split_name}.csv"
        )

        split = pd.read_csv(path)

        split["study_id"] = (
            split["study_id"]
            .astype(str)
            .str.strip()
        )

        part = df[
            df["study_id"]
            .isin(
                set(split["study_id"])
            )
        ].copy()

        print()
        print("=" * 70)
        print(split_name.upper())
        print("=" * 70)

        print(
            f"Исследований: {len(part)}"
        )

        for col in columns:

            valid = part[col].isin([0, 1])

            c0 = (
                (part.loc[valid, col] == 0)
                .sum()
            )

            c1 = (
                (part.loc[valid, col] == 1)
                .sum()
            )

            missing = (
                ~valid
            ).sum()

            print(
                f"{col:10s}: "
                f"0={c0}, "
                f"1={c1}, "
                f"missing={missing}"
            )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()