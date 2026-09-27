from pathlib import Path
import hashlib

import numpy as np
import pydicom


PROJECT_DIR = Path(
    r"C:\hakaton\densitometry_ai"
)

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)


TARGET_STUDIES = [
    "2.25.102755089973625799055786462646268820450",
    "2.25.114887542067602662452692698814728642117",
    "2.25.269648774413746842548908998641183395622",
    "2.25.338439598607961312040815816824093323492",
    "2.25.29638026465755240167985424461715024381",
    "2.25.48906424954135283790354958243772558962",
]


def image_hash(array):
    return hashlib.md5(
        array.tobytes()
    ).hexdigest()


print("=" * 80)
print("ПРОВЕРКА DICOM-ИЗОБРАЖЕНИЙ")
print("=" * 80)


for study_id in TARGET_STUDIES:

    print()
    print("-" * 80)
    print(f"STUDY: {study_id}")
    print("-" * 80)

    study_dir = (
        STUDIES_DIR
        / study_id
    )

    dicom_files = []

    for path in study_dir.rglob("*.dcm"):

        try:

            ds = pydicom.dcmread(
                path
            )

            if hasattr(
                ds,
                "PixelData"
            ):
                dicom_files.append(
                    path
                )

        except Exception:
            pass

    print(
        f"DICOM с PixelData: "
        f"{len(dicom_files)}"
    )

    hashes = {}

    for path in sorted(dicom_files):

        try:

            ds = pydicom.dcmread(
                path
            )

            image = ds.pixel_array

            h = image_hash(
                image
            )

            hashes.setdefault(
                h,
                []
            ).append(path)

            print()
            print(
                f"{path.name}"
            )

            print(
                f"  size: "
                f"{image.shape}"
            )

            print(
                f"  dtype: "
                f"{image.dtype}"
            )

            print(
                f"  min: "
                f"{image.min()}"
            )

            print(
                f"  max: "
                f"{image.max()}"
            )

            print(
                f"  mean: "
                f"{image.mean():.4f}"
            )

            print(
                f"  hash: "
                f"{h}"
            )

        except Exception as error:

            print(
                f"ОШИБКА: {path}"
            )

            print(error)

    print()
    print("УНИКАЛЬНЫЕ HASH:")

    for h, paths in hashes.items():

        print()
        print(
            f"Hash: {h}"
        )

        for path in paths:

            print(
                f"  - {path.name}"
            )

    duplicate_groups = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    print()

    if duplicate_groups:

        print(
            "НАЙДЕНЫ ПОЛНОСТЬЮ ОДИНАКОВЫЕ "
            "PIXEL DATA:"
        )

        for group in duplicate_groups:

            print(
                "  Группа:"
            )

            for path in group:

                print(
                    f"    {path.name}"
                )

    else:

        print(
            "Полностью одинаковых "
            "PixelData не найдено."
        )


print()
print("=" * 80)
print("ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 80)