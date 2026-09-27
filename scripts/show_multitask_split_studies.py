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

    manifest = pd.read_csv(MANIFEST_PATH)

    manifest["study_id"] = manifest["study_id"].astype(str)

    for column in [
        "spine_position",
        "spine_axis",
        "spine_artifacts",
        "spine_quality",
    ]:
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce"
        )

    manifest = manifest[
        manifest["anatomy"] == "spine"
    ].copy()

    manifest = manifest[
        manifest[
            [
                "spine_position",
                "spine_axis",
                "spine_artifacts",
                "spine_quality",
            ]
        ].notna().all(axis=1)
    ].copy()

    # Один study = одна уникальная spine-разметка
    manifest = manifest.drop_duplicates(
        subset=["study_id"]
    )

    print("=" * 100)
    print("ИССЛЕДОВАНИЯ ПО SPLIT")
    print("=" * 100)

    for split_name, split_path in SPLITS.items():

        split = pd.read_csv(split_path)

        split["study_id"] = split["study_id"].astype(str)

        ids = set(split["study_id"])

        df = manifest[
            manifest["study_id"].isin(ids)
        ].copy()

        df = df.sort_values(
            [
                "spine_quality",
                "spine_artifacts",
                "spine_axis",
                "spine_position",
            ]
        )

        print()
        print("=" * 100)
        print(split_name)
        print("=" * 100)

        print(
            df[
                [
                    "study_id",
                    "spine_position",
                    "spine_axis",
                    "spine_artifacts",
                    "spine_quality",
                    "comment",
                ]
            ].to_string(index=False)
        )

    print()
    print("=" * 100)
    print("ГОТОВО")
    print("=" * 100)


if __name__ == "__main__":
    main()