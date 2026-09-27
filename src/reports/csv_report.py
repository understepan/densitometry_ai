
import csv
import io


CSV_COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
    "pathology_probability",
    "position_probability",
    "axis_probability",
    "artifacts_probability",
    "error",
]


def create_csv_report(rows):
    """Создаёт CSV-отчёт в кодировке UTF-8 с BOM."""
    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_COLUMNS,
        extrasaction="ignore",
        delimiter=";",
    )
    writer.writeheader()

    for row in rows:
        writer.writerow({
            column: row.get(column, "")
            for column in CSV_COLUMNS
        })

    return buffer.getvalue().encode("utf-8-sig")