from pathlib import Path
import csv
from collections import Counter, defaultdict

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "image_manifest.csv"
)


def normalize(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


def main():
    print("=" * 90)
    print("ПРОВЕРКА DICOM-ТЕГОВ ДЛЯ ОПРЕДЕЛЕНИЯ СТОРОНЫ БЕДРА")
    print("=" * 90)

    hip_images = []

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:
            if row["anatomy"] == "hip":
                hip_images.append(row)

    print()
    print(f"HIP DICOM-файлов: {len(hip_images)}")
    print()

    # Кандидаты на информацию о стороне
    tag_names = [
        "Laterality",
        "ImageLaterality",
        "PatientOrientation",
        "ViewPosition",
        "BodyPartExamined",
        "AnatomicRegionSequence",
    ]

    counters = {
        tag: Counter()
        for tag in tag_names
    }

    examples = defaultdict(list)

    errors = []

    for index, row in enumerate(hip_images, start=1):

        path = Path(row["dicom_path"])

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )
        except Exception as error:
            errors.append(
                (
                    row["study_id"],
                    str(path),
                    str(error)
                )
            )
            continue

        for tag_name in tag_names:

            if not hasattr(ds, tag_name):
                continue

            value = getattr(ds, tag_name)

            # Для последовательности просто фиксируем наличие
            if tag_name == "AnatomicRegionSequence":
                if value is not None and len(value) > 0:
                    counters[tag_name]["PRESENT"] += 1

                    if len(examples[tag_name]) < 5:
                        examples[tag_name].append(
                            (
                                row["study_id"],
                                str(path),
                                str(value)
                            )
                        )

                continue

            value = normalize(value)

            if value is None:
                counters[tag_name]["EMPTY"] += 1
            else:
                counters[tag_name][value] += 1

                if len(examples[tag_name]) < 10:
                    examples[tag_name].append(
                        (
                            row["study_id"],
                            str(path),
                            value
                        )
                    )

    print("=" * 90)
    print("РЕЗУЛЬТАТ ПО ТЕГАМ")
    print("=" * 90)

    for tag_name in tag_names:

        print()
        print(f"--- {tag_name} ---")

        counter = counters[tag_name]

        if not counter:
            print("Тег не найден ни в одном HIP DICOM.")
            continue

        for value, count in counter.most_common():
            print(
                f"{value}: {count}"
            )

        if examples[tag_name]:
            print()
            print("Примеры:")

            for study_id, path, value in examples[tag_name]:
                print(
                    f"  study={study_id}"
                )
                print(
                    f"  value={value}"
                )
                print(
                    f"  file={path}"
                )

    print()
    print("=" * 90)
    print("ОШИБКИ")
    print("=" * 90)

    print(f"Ошибок чтения: {len(errors)}")

    for study_id, path, error in errors[:10]:
        print()
        print(f"study={study_id}")
        print(f"file={path}")
        print(f"error={error}")

    print()
    print("=" * 90)
    print("ГОТОВО")
    print("=" * 90)


if __name__ == "__main__":
    main()