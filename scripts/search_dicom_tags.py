from pathlib import Path

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDIES_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "training"
    / "Исследования"
)


STUDY_NAMES = [
    "1.2.840.113619.2.110.512719.20250403090444",
    "1.2.840.113619.2.110.512719.20250403092052",
    "2.25.114887542067602662452692698814787122937",
]


KEYWORDS = [
    "later",
    "right",
    "left",
    "side",
    "body",
    "part",
    "view",
    "position",
    "description",
    "protocol",
    "region",
    "anatom",
    "series",
    "image",
]


def main():

    print("=" * 100)
    print("ПОИСК ПОЛЕЗНЫХ DICOM-ТЕГОВ")
    print("=" * 100)

    for study_name in STUDY_NAMES:

        study_dir = STUDIES_DIR / study_name

        if not study_dir.exists():
            print(f"Не найдена папка: {study_dir}")
            continue

        dicom_files = sorted(
            study_dir.rglob("*.dcm")
        )

        if not dicom_files:
            continue

        # Берём только первый DICOM исследования,
        # чтобы вывод не был огромным.
        dicom_path = dicom_files[0]

        print()
        print("=" * 100)
        print(f"ИССЛЕДОВАНИЕ: {study_name}")
        print(f"ФАЙЛ: {dicom_path.name}")
        print("=" * 100)

        ds = pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True
        )

        found = []

        for element in ds.iterall():

            keyword = element.keyword or ""
            name = str(element.name)

            text = (
                f"{keyword} {name}"
            ).lower()

            if any(
                word in text
                for word in KEYWORDS
            ):
                found.append(
                    (
                        element.tag,
                        keyword,
                        name,
                        str(element.value)
                    )
                )

        for tag, keyword, name, value in found:

            print(
                f"{tag} | "
                f"{keyword} | "
                f"{name} | "
                f"{value}"
            )

        print()
        print(f"Найдено подходящих тегов: {len(found)}")

        # Проверяем приватные теги отдельно.
        private_tags = [
            element
            for element in ds.iterall()
            if element.tag.is_private
        ]

        print(
            f"Приватных тегов: "
            f"{len(private_tags)}"
        )

        for element in private_tags:

            print(
                f"PRIVATE | "
                f"{element.tag} | "
                f"{element.name} | "
                f"{element.value}"
            )

    print()
    print("=" * 100)
    print("ПОИСК ЗАВЕРШЁН")
    print("=" * 100)


if __name__ == "__main__":
    main()