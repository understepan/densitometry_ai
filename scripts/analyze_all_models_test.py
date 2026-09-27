from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(r"C:\hakaton\densitometry_ai")

MANIFEST = ROOT / "data" / "processed" / "labeled_manifest.csv"
TEST_SPLIT = ROOT / "data" / "splits" / "test.csv"

RESULTS = ROOT / "results"


# ============================================================
# LOAD TEST
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

test_split = pd.read_csv(
    TEST_SPLIT
)

manifest["study_id"] = (
    manifest["study_id"].astype(str)
)

test_ids = (
    test_split["study_id"].astype(str)
)


test_df = manifest[
    (manifest["anatomy"] == "spine")
    &
    (manifest["spine_quality"].notna())
    &
    (manifest["study_id"].isin(test_ids))
].copy()


# ============================================================
# EXACT PIXEL DEDUP
# ============================================================

import hashlib
import pydicom


def resolve_path(path):

    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    return path


hashes = []

for path in test_df["dicom_path"]:

    path = resolve_path(path)

    ds = pydicom.dcmread(path)

    pixel_bytes = (
        ds.pixel_array.tobytes()
    )

    digest = hashlib.md5(
        pixel_bytes
    ).hexdigest()

    hashes.append(digest)


test_df["pixel_hash"] = hashes

test_df = (
    test_df
    .drop_duplicates(
        subset=["pixel_hash"],
        keep="first"
    )
    .drop(columns=["pixel_hash"])
    .reset_index(drop=True)
)


print("=" * 80)
print("ANALYSIS OF V9 / V11 / V13 / V15")
print("=" * 80)

print()
print(
    f"TEST samples: {len(test_df)}"
)

print(
    f"Unique studies: "
    f"{test_df['study_id'].nunique()}"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

v13v15_path = (
    RESULTS /
    "v13_v15_test_comparison.csv"
)

v13v15 = pd.read_csv(
    v13v15_path
)


# ------------------------------------------------------------
# V9
# ------------------------------------------------------------

v9_path = (
    RESULTS /
    "spine_quality_v9_exact_predictions.csv"
)

v9 = pd.read_csv(
    v9_path
)


# ------------------------------------------------------------
# V11
# ------------------------------------------------------------

v11_path = (
    RESULTS /
    "spine_quality_v11_test_threshold045.csv"
)

v11 = pd.read_csv(
    v11_path
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def prepare_predictions(
    df,
    prediction_column,
    probability_column,
    prefix
):

    result = df[
        [
            "study_id",
            prediction_column,
            probability_column
        ]
    ].copy()

    result = result.rename(
        columns={
            prediction_column:
                f"{prefix}_prediction",

            probability_column:
                f"{prefix}_probability"
        }
    )

    return result


v9_clean = prepare_predictions(
    v9,
    "prediction",
    "probability_class1",
    "V9"
)


v11_clean = prepare_predictions(
    v11,
    "prediction",
    "probability_class1",
    "V11"
)


v13_clean = prepare_predictions(
    v13v15,
    "V13_prediction",
    "V13_probability_class1",
    "V13"
)


v15_clean = prepare_predictions(
    v13v15,
    "V15_prediction",
    "V15_probability_class1",
    "V15"
)


# ============================================================
# MERGE
# ============================================================

columns_to_keep = [
    "study_id",
    "dicom_path",
    "spine_quality",
    "spine_position",
    "spine_axis",
    "spine_artifacts",
    "comment"
]

result = test_df[
    columns_to_keep
].copy()


result = result.merge(
    v9_clean,
    on="study_id",
    how="left"
)

result = result.merge(
    v11_clean,
    on="study_id",
    how="left"
)

result = result.merge(
    v13_clean,
    on="study_id",
    how="left"
)

result = result.merge(
    v15_clean,
    on="study_id",
    how="left"
)


# ============================================================
# CORRECT / ERROR
# ============================================================

for model in [
    "V9",
    "V11",
    "V13",
    "V15"
]:

    result[
        f"{model}_correct"
    ] = (
        result[
            f"{model}_prediction"
        ]
        ==
        result["spine_quality"]
    )


# ============================================================
# NUMBER OF MODELS CORRECT
# ============================================================

result["correct_count"] = (
    result[
        [
            "V9_correct",
            "V11_correct",
            "V13_correct",
            "V15_correct"
        ]
    ]
    .sum(axis=1)
)


result["error_count"] = (
    4 - result["correct_count"]
)


# ============================================================
# CONSENSUS
# ============================================================

def consensus(row):

    predictions = [
        row["V9_prediction"],
        row["V11_prediction"],
        row["V13_prediction"],
        row["V15_prediction"]
    ]

    counts = {
        0: predictions.count(0),
        1: predictions.count(1)
    }

    if counts[0] > counts[1]:
        return 0

    if counts[1] > counts[0]:
        return 1

    return -1


result["consensus"] = result.apply(
    consensus,
    axis=1
)


# ============================================================
# PRINT ALL CASES
# ============================================================

print()
print("=" * 80)
print("ВСЕ 15 TEST-ИССЛЕДОВАНИЙ")
print("=" * 80)

for _, row in result.iterrows():

    print()
    print(
        f"Study: {row['study_id']}"
    )

    print(
        f"TRUE: {int(row['spine_quality'])}"
    )

    print(
        "Labels: "
        f"position={row['spine_position']} "
        f"axis={row['spine_axis']} "
        f"artifacts={row['spine_artifacts']}"
    )

    print(
        "V9 : "
        f"{int(row['V9_prediction'])} "
        f"(p1={row['V9_probability']:.3f}) "
        f"{'OK' if row['V9_correct'] else 'ERROR'}"
    )

    print(
        "V11: "
        f"{int(row['V11_prediction'])} "
        f"(p1={row['V11_probability']:.3f}) "
        f"{'OK' if row['V11_correct'] else 'ERROR'}"
    )

    print(
        "V13: "
        f"{int(row['V13_prediction'])} "
        f"(p1={row['V13_probability']:.3f}) "
        f"{'OK' if row['V13_correct'] else 'ERROR'}"
    )

    print(
        "V15: "
        f"{int(row['V15_prediction'])} "
        f"(p1={row['V15_probability']:.3f}) "
        f"{'OK' if row['V15_correct'] else 'ERROR'}"
    )

    print(
        f"Correct models: "
        f"{int(row['correct_count'])}/4"
    )

    if pd.notna(row["comment"]):

        comment = str(
            row["comment"]
        ).strip()

        if comment:
            print(
                f"Comment: {comment}"
            )


# ============================================================
# ERROR-ONLY TABLE
# ============================================================

errors = result[
    result["error_count"] > 0
].copy()


print()
print("=" * 80)
print("ИССЛЕДОВАНИЯ, ГДЕ ХОТЯ БЫ ОДНА МОДЕЛЬ ОШИБЛАСЬ")
print("=" * 80)

print()

print(
    errors[
        [
            "study_id",
            "spine_quality",
            "spine_position",
            "spine_axis",
            "spine_artifacts",
            "V9_prediction",
            "V11_prediction",
            "V13_prediction",
            "V15_prediction",
            "correct_count",
            "comment"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# EACH MODEL ERRORS
# ============================================================

for model in [
    "V9",
    "V11",
    "V13",
    "V15"
]:

    model_errors = result[
        ~result[
            f"{model}_correct"
        ]
    ]

    print()
    print("=" * 80)
    print(
        f"{model} ERRORS: "
        f"{len(model_errors)}"
    )
    print("=" * 80)

    if len(model_errors) == 0:

        print("Ошибок нет.")

    else:

        print(
            model_errors[
                [
                    "study_id",
                    "spine_quality",
                    "spine_position",
                    "spine_axis",
                    "spine_artifacts",
                    f"{model}_prediction",
                    f"{model}_probability",
                    "comment"
                ]
            ].to_string(
                index=False
            )
        )


# ============================================================
# PATTERN ANALYSIS
# ============================================================

print()
print("=" * 80)
print("АНАЛИЗ КОМБИНАЦИЙ LABELS")
print("=" * 80)


patterns = (
    result
    .groupby(
        [
            "spine_position",
            "spine_axis",
            "spine_artifacts"
        ],
        dropna=False
    )
    .agg(
        samples=(
            "study_id",
            "count"
        ),

        true_positive=(
            "spine_quality",
            lambda x:
            int((x == 1).sum())
        ),

        true_normal=(
            "spine_quality",
            lambda x:
            int((x == 0).sum())
        ),

        V9_correct=(
            "V9_correct",
            "sum"
        ),

        V11_correct=(
            "V11_correct",
            "sum"
        ),

        V13_correct=(
            "V13_correct",
            "sum"
        ),

        V15_correct=(
            "V15_correct",
            "sum"
        )
    )
    .reset_index()
)


print(
    patterns.to_string(
        index=False
    )
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for model in [
    "V9",
    "V11",
    "V13",
    "V15"
]:

    correct = int(
        result[
            f"{model}_correct"
        ].sum()
    )

    total = len(result)

    accuracy = (
        correct / total * 100
    )

    print(
        f"{model}: "
        f"{correct}/{total} = "
        f"{accuracy:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    RESULTS /
    "all_models_test_error_analysis.csv"
)

result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


patterns_path = (
    RESULTS /
    "all_models_test_label_patterns.csv"
)

patterns.to_csv(
    patterns_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 80)
print("ФАЙЛЫ СОХРАНЕНЫ")
print("=" * 80)

print(output_path)
print(patterns_path)

print()
print("ГОТОВО")