from pathlib import Path
from collections import Counter
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_DIR)
)

from src.preprocessing.dataset import SpineQualityDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "train.csv"
)

VAL_FILE = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "validation.csv"
)

TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "splits"
    / "test.csv"
)


def test_dataset(name, csv_file):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    dataset = SpineQualityDataset(
        csv_file=csv_file,
        project_dir=PROJECT_DIR,
        image_size=224,
    )

    print(
        f"Количество изображений: "
        f"{len(dataset)}"
    )

    labels = []

    for i in range(
        len(dataset)
    ):

        item = dataset[i]

        labels.append(
            int(item["label"])
        )

        # Проверяем первые изображения
        if i < 3:

            print()
            print(
                f"Изображение {i}:"
            )

            print(
                "  shape:",
                tuple(
                    item["image"].shape
                )
            )

            print(
                "  dtype:",
                item["image"].dtype
            )

            print(
                "  min:",
                float(
                    item["image"].min()
                )
            )

            print(
                "  max:",
                float(
                    item["image"].max()
                )
            )

            print(
                "  label:",
                int(item["label"])
            )

            print(
                "  study_id:",
                item["study_id"]
            )

    counts = Counter(labels)

    print()
    print(
        "Распределение:"
    )

    print(
        "  class 0:",
        counts[0]
    )

    print(
        "  class 1:",
        counts[1]
    )

    # --------------------------------------------------------
    # Проверки
    # --------------------------------------------------------

    assert len(dataset) > 0

    item = dataset[0]

    assert (
        tuple(item["image"].shape)
        == (1, 224, 224)
    )

    assert (
        item["image"].dtype.is_floating_point
    )

    assert (
        float(item["image"].min()) >= 0.0
    )

    assert (
        float(item["image"].max()) <= 1.0
    )

    assert (
        int(item["label"]) in {0, 1}
    )

    print()
    print(
        "ПРОВЕРКИ DATASET ПРОЙДЕНЫ."
    )


def main():

    print("=" * 80)
    print("ТЕСТ PYTORCH DATASET")
    print("=" * 80)

    test_dataset(
        "TRAIN",
        TRAIN_FILE
    )

    test_dataset(
        "VALIDATION",
        VAL_FILE
    )

    test_dataset(
        "TEST",
        TEST_FILE
    )

    print()
    print("=" * 80)
    print(
        "ВСЕ DATASET ПРОВЕРКИ ПРОЙДЕНЫ."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()