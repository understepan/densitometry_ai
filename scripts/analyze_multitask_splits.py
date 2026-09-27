from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)

SPLITS = {
    "TRAIN": PROJECT_ROOT / "data" / "splits" / "train.csv",
    "VALIDATION": PROJECT_ROOT / "data" / "splits" / "validation.csv",
    "TEST": PROJECT_ROOT / "data" / "splits" / "test.csv",
}


def main():

    print("=" * 70)
    print("АНАЛИЗ MULTI-TASK РАЗМЕТКИ ПО SPLIT")
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest["study_id"] = (
        manifest["study_id"].astype(str)
    )

    columns = [
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "spine_quality"
    ]

    for column in columns:

        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce"
        )

    manifest = manifest[
        manifest["anatomy"] == "spine"
    ].copy()

    manifest = manifest[
        manifest[columns]
        .notna()
        .all(axis=1)
    ].copy()

    # Один study = одна уникальная spine-разметка
    manifest = manifest.drop_duplicates(
        subset=["study_id"]
    ).copy()

    print()
    print(
        "Всего исследований:",
        len(manifest)
    )

    for split_name, split_path in SPLITS.items():

        split = pd.read_csv(
            split_path
        )

        split["study_id"] = (
            split["study_id"].astype(str)
        )

        ids = set(
            split["study_id"]
        )

        df = manifest[
            manifest["study_id"].isin(ids)
        ].copy()

        print()
        print("=" * 70)
        print(split_name)
        print("=" * 70)

        print(
            "Исследований:",
            len(df)
        )

        for column in columns:

            counts = (
                df[column]
                .value_counts()
                .sort_index()
            )

            print()
            print(column)

            print(
                counts.to_string()
            )

        print()

        print(
            "Комбинации:"
        )

        combo = (
            df
            .groupby(
                [
                    "spine_position",
                    "spine_axis",
                    "spine_artifacts"
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        print(
            combo.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()