from pathlib import Path
from collections import defaultdict, Counter

import pydicom


PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = PROJECT_DIR / "data" / "raw" / "training"
STUDIES_DIR = TRAINING_DIR / "Исследования"


def main():
    print("=" * 80)
    print("АНАЛИЗ ВСЕХ ОБУЧАЮЩИХ DICOM")
    print("=" * 80)
    print()

    # ---------------------------------------------------------
    # Находим все DICOM
    # ---------------------------------------------------------

    dicom_files = sorted(
        STUDIES_DIR.rglob("*.dcm")
    )

    print(
        f"Всего найдено DICOM-файлов: "
        f"{len(dicom_files)}"
    )

    print()

    # ---------------------------------------------------------
    # Статистика
    # ---------------------------------------------------------

    studies = defaultdict(list)

    total_read_errors = 0
    read_errors = []

    modality_counter = Counter()
    shape_counter = Counter()
    series_counter = Counter()

    # ---------------------------------------------------------
    # Читаем DICOM
    # ---------------------------------------------------------

    for index, file_path in enumerate(
        dicom_files,
        start=1
    ):

        try:
            dataset = pydicom.dcmread(
                file_path,
                stop_before_pixels=True
            )

            # Имя папки исследования
            study_folder = file_path.relative_to(
                STUDIES_DIR
            ).parts[0]

            study_uid = str(
                getattr(
                    dataset,
                    "StudyInstanceUID",
                    "<ОТСУТСТВУЕТ>"
                )
            )

            series_uid = str(
                getattr(
                    dataset,
                    "SeriesInstanceUID",
                    "<ОТСУТСТВУЕТ>"
                )
            )

            modality = str(
                getattr(
                    dataset,
                    "Modality",
                    "<ОТСУТСТВУЕТ>"
                )
            )

            rows = getattr(
                dataset,
                "Rows",
                None
            )

            columns = getattr(
                dataset,
                "Columns",
                None
            )

            shape = (
                rows,
                columns
            )

            studies[study_folder].append({
                "file": file_path,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "modality": modality,
                "shape": shape,
            })

            modality_counter[modality] += 1
            shape_counter[shape] += 1
            series_counter[
                (study_folder, series_uid)
            ] += 1

        except Exception as error:

            total_read_errors += 1

            read_errors.append({
                "file": file_path,
                "error": str(error),
            })

    # ---------------------------------------------------------
    # Общая статистика
    # ---------------------------------------------------------

    print("=" * 80)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 80)
    print()

    print(
        f"Исследований: "
        f"{len(studies)}"
    )

    print(
        f"DICOM-файлов: "
        f"{len(dicom_files)}"
    )

    print(
        f"Ошибок чтения: "
        f"{total_read_errors}"
    )

    print()

    # ---------------------------------------------------------
    # Modality
    # ---------------------------------------------------------

    print("=" * 80)
    print("MODALITY")
    print("=" * 80)
    print()

    for modality, count in modality_counter.most_common():
        print(
            f"{modality}: "
            f"{count}"
        )

    print()

    # ---------------------------------------------------------
    # Размеры изображений
    # ---------------------------------------------------------

    print("=" * 80)
    print("РАЗМЕРЫ ИЗОБРАЖЕНИЙ")
    print("=" * 80)
    print()

    for shape, count in shape_counter.most_common():
        print(
            f"{shape[0]} x {shape[1]}: "
            f"{count}"
        )

    print()

    # ---------------------------------------------------------
    # Количество серий
    # ---------------------------------------------------------

    series_per_study = {}

    for study_folder, files in studies.items():

        unique_series = {
            item["series_uid"]
            for item in files
        }

        series_per_study[study_folder] = (
            len(unique_series)
        )

    series_distribution = Counter(
        series_per_study.values()
    )

    print("=" * 80)
    print("КОЛИЧЕСТВО СЕРИЙ НА ИССЛЕДОВАНИЕ")
    print("=" * 80)
    print()

    for series_count, study_count in sorted(
        series_distribution.items()
    ):
        print(
            f"{series_count} серий: "
            f"{study_count} исследований"
        )

    print()

    # ---------------------------------------------------------
    # Количество DICOM на исследование
    # ---------------------------------------------------------

    dicom_distribution = Counter(
        len(files)
        for files in studies.values()
    )

    print("=" * 80)
    print("КОЛИЧЕСТВО DICOM НА ИССЛЕДОВАНИЕ")
    print("=" * 80)
    print()

    for dicom_count, study_count in sorted(
        dicom_distribution.items()
    ):
        print(
            f"{dicom_count} DICOM: "
            f"{study_count} исследований"
        )

    print()

    # ---------------------------------------------------------
    # Первые 10 исследований подробно
    # ---------------------------------------------------------

    print("=" * 80)
    print("ПЕРВЫЕ 10 ИССЛЕДОВАНИЙ")
    print("=" * 80)
    print()

    for study_folder in sorted(studies)[:10]:

        files = studies[study_folder]

        unique_series = defaultdict(list)

        for item in files:
            unique_series[
                item["series_uid"]
            ].append(item)

        print("-" * 80)

        print(
            f"Исследование: "
            f"{study_folder}"
        )

        print(
            f"DICOM: "
            f"{len(files)}"
        )

        print(
            f"Серий: "
            f"{len(unique_series)}"
        )

        for series_uid, series_files in (
            unique_series.items()
        ):

            first = series_files[0]

            print()
            print(
                f"  SeriesInstanceUID: "
                f"{series_uid}"
            )

            print(
                f"  Количество файлов: "
                f"{len(series_files)}"
            )

            print(
                f"  Modality: "
                f"{first['modality']}"
            )

            print(
                f"  Размер: "
                f"{first['shape'][0]} x "
                f"{first['shape'][1]}"
            )

        print()

    # ---------------------------------------------------------
    # Ошибки
    # ---------------------------------------------------------

    if read_errors:

        print("=" * 80)
        print("ОШИБКИ ЧТЕНИЯ")
        print("=" * 80)
        print()

        for item in read_errors:

            print(
                f"Файл: "
                f"{item['file']}"
            )

            print(
                f"Ошибка: "
                f"{item['error']}"
            )

            print()

    # ---------------------------------------------------------
    # Проверяем, совпадает ли StudyInstanceUID
    # внутри папки исследования
    # ---------------------------------------------------------

    print("=" * 80)
    print("ПРОВЕРКА StudyInstanceUID ВНУТРИ ПАПОК")
    print("=" * 80)
    print()

    inconsistent_studies = []

    for study_folder, files in studies.items():

        unique_study_uids = {
            item["study_uid"]
            for item in files
        }

        if len(unique_study_uids) > 1:

            inconsistent_studies.append(
                (
                    study_folder,
                    unique_study_uids
                )
            )

    print(
        f"Исследований с несколькими "
        f"StudyInstanceUID: "
        f"{len(inconsistent_studies)}"
    )

    if inconsistent_studies:

        print()

        for study_folder, uids in (
            inconsistent_studies
        ):

            print(
                f"Исследование: "
                f"{study_folder}"
            )

            for uid in uids:
                print(
                    f"  {uid}"
                )

    print()

    # ---------------------------------------------------------
    # Финал
    # ---------------------------------------------------------

    print("=" * 80)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 80)


if __name__ == "__main__":
    main()