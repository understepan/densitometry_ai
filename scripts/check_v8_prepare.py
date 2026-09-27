from pathlib import Path
import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)


def main():

    print("=" * 70)
    print("ПРОВЕРКА ПОДГОТОВКИ V8")
    print("=" * 70)

    df = pd.read_csv(MANIFEST_PATH)

    print("Всего строк:", len(df))

    # Только позвоночник
    df = df[df["anatomy"] == "spine"].copy()

    print("Spine:", len(df))

    # Только строки с валидным final label
    df["label"] = pd.to_numeric(
        df["spine_quality"],
        errors="coerce"
    )

    df = df[df["label"].isin([0, 1])].copy()

    print("С валидной меткой:", len(df))

    print("\nКлассы:")
    print(df["label"].value_counts().sort_index())

    print("\nИсследований:")
    print(df["study_id"].nunique())

    # Проверяем существование DICOM
    exists = 0
    missing = 0

    for path in df["dicom_path"]:

        full_path = PROJECT_ROOT / path

        if full_path.exists():
            exists += 1
        else:
            missing += 1

    print("\nDICOM:")
    print("существуют:", exists)
    print("отсутствуют:", missing)

    # Проверяем PixelData
    print("\nПроверка точных дубликатов PixelData...")

    hashes = {}

    for _, row in df.iterrows():

        path = PROJECT_ROOT / row["dicom_path"]

        try:
            ds = pydicom.dcmread(
                str(path),
                force=True
            )

            pixel = ds.PixelData

            if pixel in hashes:
                hashes[pixel].append(
                    row["dicom_path"]
                )
            else:
                hashes[pixel] = [
                    row["dicom_path"]
                ]

        except Exception as e:

            print(
                "Ошибка:",
                row["dicom_path"],
                e
            )

    duplicate_groups = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    print(
        "Уникальных PixelData:",
        len(hashes)
    )

    print(
        "Групп дубликатов:",
        len(duplicate_groups)
    )

    duplicated_rows = sum(
        len(group) - 1
        for group in duplicate_groups
    )

    print(
        "Лишних дубликатов:",
        duplicated_rows
    )

    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    main()