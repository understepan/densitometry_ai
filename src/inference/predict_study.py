
from pathlib import Path
import sys
import time

import pydicom

from predict_v13 import predict


def find_dicoms(study_dir: Path):
    """Находит DICOM-файлы внутри исследования."""
    return sorted(study_dir.rglob("*.dcm"))


def guess_anatomy(rows, columns):
    """Эвристика, проверенная на нашем датасете."""
    if columns == 300:
        return "spine"
    if columns == 280:
        return "hip"
    if (rows, columns) in {(401, 248), (405, 248)}:
        return "hip"
    return "unknown"


def process_study(study_dir: str):
    study_path = Path(study_dir)

    if not study_path.exists():
        raise FileNotFoundError(
            f"Папка исследования не найдена:\n{study_path}"
        )

    if not study_path.is_dir():
        raise NotADirectoryError(
            f"Указанный путь не является папкой:\n{study_path}"
        )

    dicom_files = find_dicoms(study_path)

    if not dicom_files:
        raise FileNotFoundError(
            f"В исследовании не найдено DICOM-файлов:\n{study_path}"
        )

    print("=" * 70)
    print("DENSITOMETRY AI — ОБРАБОТКА ИССЛЕДОВАНИЯ")
    print("=" * 70)
    print(f"Исследование: {study_path.name}")
    print(f"Найдено DICOM: {len(dicom_files)}")
    print()

    results = []
    report_rows = []

    for index, dicom_path in enumerate(dicom_files, start=1):
        print(f"[{index}/{len(dicom_files)}] {dicom_path.name}")
        start_time = time.perf_counter()

        row = {
            "path_to_study": str(study_path),
            "study_uid": "",
            "image_uid": "",
            "anatomical_region": "unknown",
            "quality_class": "",
            "violation_type": "не определён",
            "processing_status": "Failure",
            "time_of_processing": "",
            "pathology_probability": "",
            "position_probability": "",
            "axis_probability": "",
            "artifacts_probability": "",
            "error": "",
        }

        try:
            ds = pydicom.dcmread(dicom_path)

            row["study_uid"] = str(
                getattr(ds, "StudyInstanceUID", "")
            )
            row["image_uid"] = str(
                getattr(ds, "SOPInstanceUID", "")
            )

            rows = int(getattr(ds, "Rows", 0))
            columns = int(getattr(ds, "Columns", 0))
            row["anatomical_region"] = guess_anatomy(
                rows, columns
            )

            result = predict(str(dicom_path))

            prediction = result["prediction"]
            probability = float(result["probability"])

            row["quality_class"] = (
                1 if prediction == "Патология" else 0
            )
            row["pathology_probability"] = probability
            row["position_probability"] = float(
                result["position_probability"]
            )
            row["axis_probability"] = float(
                result["axis_probability"]
            )
            row["artifacts_probability"] = float(
                result["artifacts_probability"]
            )
            row["processing_status"] = "Success"

            result["file"] = dicom_path.name
            result["path"] = str(dicom_path)
            result["rows"] = rows
            result["columns"] = columns
            result["modality"] = str(
                getattr(ds, "Modality", "")
            )
            result["processing_status"] = "Success"
            results.append(result)

            print(f"  Результат: {prediction}")
            print(f"  Вероятность патологии: {probability:.3f}")

        except Exception as error:
            row["error"] = str(error)
            print(f"  ОШИБКА: {error}")

        finally:
            row["time_of_processing"] = round(
                time.perf_counter() - start_time, 4
            )
            report_rows.append(row)
            print()

    print("=" * 70)
    print("ИТОГ")
    print("=" * 70)
    print(f"Всего файлов: {len(dicom_files)}")
    print(f"Успешно обработано: {len(results)}")
    print(f"Ошибок: {len(dicom_files) - len(results)}")

    if results:
        pathology_count = sum(
            r["prediction"] == "Патология" for r in results
        )
        normal_count = len(results) - pathology_count
        average_probability = sum(
            r["probability"] for r in results
        ) / len(results)

        print(f"Норма: {normal_count}")
        print(f"Патология: {pathology_count}")
        print(
            f"Средняя вероятность патологии: "
            f"{average_probability:.3f}"
        )

    print("=" * 70)

    return {
        "results": results,
        "report_rows": report_rows,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование:")
        print("python predict_study.py путь_к_исследованию")
        raise SystemExit(1)

    process_study(sys.argv[1])