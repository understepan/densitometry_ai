
from pathlib import Path
import sys

import numpy as np
import pydicom
import streamlit as st
from PIL import Image


# ---------- ПУТИ К ПРОЕКТУ ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INFERENCE_DIR = PROJECT_ROOT / "src" / "inference"
REPORTS_DIR = PROJECT_ROOT / "src" / "reports"

sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(REPORTS_DIR))

from predict_study import process_study
from csv_report import create_csv_report


# ---------- НАСТРОЙКИ STREAMLIT ----------

st.set_page_config(
    page_title="Densitometry AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- ДИЗАЙН ----------

st.markdown("""
<style>
    .stApp {
        background: #0b1220;
        color: #e5eaf3;
    }

    h1, h2, h3 {
        color: #f3f6fc !important;
    }

    .hero {
        background: linear-gradient(135deg, #14243b, #1b3656);
        border: 1px solid #29496b;
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 750;
        color: #f4f8ff;
        letter-spacing: 0.3px;
    }

    .hero-subtitle {
        color: #a9bdd6;
        font-size: 15px;
        margin-top: 5px;
    }

    .metric-card {
        background: #131e30;
        border: 1px solid #293a53;
        border-radius: 14px;
        padding: 20px;
        min-height: 112px;
    }

    .metric-label {
        font-size: 13px;
        color: #9eafc6;
        margin-bottom: 9px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 750;
        color: #f4f7fc;
    }

    .result-card {
        background: #131e30;
        border: 1px solid #293a53;
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0 20px 0;
    }

    .result-normal {
        color: #62d6a1;
        font-size: 27px;
        font-weight: 750;
    }

    .result-pathology {
        color: #ff8585;
        font-size: 27px;
        font-weight: 750;
    }

    .muted {
        color: #9eafc6;
        font-size: 13px;
    }

    .section-label {
        color: #a9bdd6;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.3px;
        margin-bottom: 8px;
    }

    div[data-testid="stExpander"] {
        background: #131e30;
        border: 1px solid #293a53;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    .stButton > button {
        background: #3b82f6;
        color: white;
        border: 0;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #2563eb;
        color: white;
    }

    div[data-testid="stProgress"] > div > div {
        background: #3b82f6;
    }

    .notice {
        background: #172438;
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 12px 16px;
        color: #c1d0e4;
        font-size: 13px;
        margin: 12px 0 20px 0;
    }

    .stDownloadButton > button {
        background: #16834b;
        color: white;
        border: 0;
        border-radius: 10px;
        min-height: 46px;
        font-weight: 700;
    }

    .stDownloadButton > button:hover {
        background: #126b3e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

def format_percent(value):
    return f"{value * 100:.1f}%"


def make_preview(dicom_path):
    """Создаёт изображение для просмотра в интерфейсе."""
    dataset = pydicom.dcmread(dicom_path)
    image = dataset.pixel_array

    # Для многокадрового DICOM показываем первый кадр.
    if image.ndim == 3:
        image = image[0]

    image = image.astype(np.float32)

    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 <= p1:
        image = np.zeros_like(image, dtype=np.uint8)
    else:
        image = np.clip(image, p1, p99)
        image = (
            (image - p1) / (p99 - p1) * 255
        ).astype(np.uint8)

    # В MONOCHROME1 чёрное и белое отображаются наоборот.
    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = 255 - image

    return Image.fromarray(image)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_previous_report():
    """Удаляет предыдущий отчёт перед новым анализом."""
    st.session_state["csv_report"] = None
    st.session_state["csv_filename"] = None
    st.session_state["study_data"] = None


# ---------- СОСТОЯНИЕ ПРИЛОЖЕНИЯ ----------

if "csv_report" not in st.session_state:
    st.session_state["csv_report"] = None

if "csv_filename" not in st.session_state:
    st.session_state["csv_filename"] = None

if "study_data" not in st.session_state:
    st.session_state["study_data"] = None


# ---------- ШАПКА ----------

st.markdown("""
<div class="hero">
    <div class="hero-title">🦴 Densitometry AI</div>
    <div class="hero-subtitle">
        Интеллектуальная обработка DICOM-исследований · Модель V13
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
    Демонстрационный исследовательский прототип.
    Результаты модели не являются медицинским диагнозом
    и требуют независимой проверки специалистом.
</div>
""", unsafe_allow_html=True)


# ---------- ВЫБОР ИССЛЕДОВАНИЯ ----------

st.markdown(
    '<div class="section-label">АНАЛИЗ ИССЛЕДОВАНИЯ</div>',
    unsafe_allow_html=True,
)

study_path = st.text_input(
    "Путь к папке исследования",
    placeholder=r"C:\data\study",
)

analyze = st.button(
    "🔍  Анализировать исследование",
    type="primary",
    use_container_width=True,
)


# ---------- АНАЛИЗ ----------

if analyze:
    clear_previous_report()

    if not study_path.strip():
        st.warning("Сначала укажи путь к папке исследования.")
        st.stop()

    folder = Path(study_path.strip())

    if not folder.exists():
        st.error("Папка исследования не найдена.")
        st.stop()

    if not folder.is_dir():
        st.error("Указанный путь не является папкой.")
        st.stop()

    try:
        with st.spinner("Обрабатываем DICOM-файлы..."):
            study_data = process_study(str(folder))

        results = study_data["results"]
        report_rows = study_data["report_rows"]

        st.session_state["csv_report"] = create_csv_report(
            report_rows
        )

        safe_folder_name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in folder.name
        )

        st.session_state["csv_filename"] = (
            f"densitometry_report_{safe_folder_name}.csv"
        )

        st.session_state["study_data"] = study_data

        if not results:
            st.warning(
                "Не удалось обработать ни одного изображения. "
                "Ошибки сохранены в CSV-отчёте."
            )
        else:
            st.success(
                f"Обработка завершена. "
                f"Успешно: {len(results)} из {len(report_rows)}."
            )

    except Exception as error:
        st.error(f"Ошибка при анализе исследования: {error}")

        # Формируем отчёт об ошибке на уровне исследования.
        report_rows = [{
            "path_to_study": str(folder),
            "study_uid": "",
            "image_uid": "",
            "anatomical_region": "unknown",
            "quality_class": "",
            "violation_type": "",
            "processing_status": "Failure",
            "time_of_processing": "",
            "pathology_probability": "",
            "position_probability": "",
            "axis_probability": "",
            "artifacts_probability": "",
            "error": str(error)
        }]

        st.session_state["csv_report"] = create_csv_report(
            report_rows
        )

        safe_folder_name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in folder.name
        )

        st.session_state["csv_filename"] = (
            f"densitometry_report_{safe_folder_name}.csv"
        )


# ---------- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ----------

study_data = st.session_state["study_data"]

if study_data is not None:
    results = study_data["results"]
    report_rows = study_data["report_rows"]

    successful_count = len(results)
    failed_count = len(report_rows) - successful_count

    st.divider()
    st.markdown(
        '<div class="section-label">СВОДКА</div>',
        unsafe_allow_html=True,
    )

    if results:
        pathology_count = sum(
            r["prediction"] == "Патология"
            for r in results
        )
        normal_count = len(results) - pathology_count

        average_probability = sum(
            r["probability"] for r in results
        ) / len(results)

        # Демонстрационное правило большинства.
        study_prediction = (
            "Патология"
            if pathology_count > len(results) / 2
            else "Норма"
        )

        result_class = (
            "result-pathology"
            if study_prediction == "Патология"
            else "result-normal"
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="muted">
                    Результат по демонстрационному правилу большинства
                </div>
                <div class="{result_class}">
                    {study_prediction.upper()}
                </div>
                <div class="muted">
                    Среднее значение выходной вероятности модели:
                    {format_percent(average_probability)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- МЕТРИКИ ----------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("ИЗОБРАЖЕНИЙ", str(successful_count))

        with c2:
            metric_card("НОРМА", str(normal_count))

        with c3:
            metric_card("ПАТОЛОГИЯ", str(pathology_count))

        with c4:
            metric_card(
                "СРЕДНЕЕ P",
                format_percent(average_probability),
            )

    # ---------- СТАТУС ОБРАБОТКИ ----------

    st.markdown("")

    c1, c2 = st.columns(2)

    with c1:
        metric_card("УСПЕШНО", str(successful_count))

    with c2:
        metric_card("ОШИБКИ", str(failed_count))

    # ---------- РЕЗУЛЬТАТЫ ПО DICOM ----------

    st.divider()
    st.markdown(
        '<div class="section-label">ИЗОБРАЖЕНИЯ</div>',
        unsafe_allow_html=True,
    )

    if results:
        for index, result in enumerate(results, start=1):
            prediction = result["prediction"]
            probability = result["probability"]

            with st.expander(
                f"{index}. {result['file']}  ·  "
                f"{prediction}  ·  "
                f"{format_percent(probability)}",
                expanded=(index == 1),
            ):
                image_col, info_col = st.columns([1.15, 1])

                with image_col:
                    try:
                        preview = make_preview(result["path"])

                        st.image(
                            preview,
                            caption=result["file"],
                            use_container_width=True,
                        )
                    except Exception as image_error:
                        st.warning(
                            "Не удалось показать изображение: "
                            f"{image_error}"
                        )

                with info_col:
                    st.markdown("**Выходы модели**")

                    st.write(f"Класс: **{prediction}**")

                    st.write(
                        "Вероятность патологии: "
                        f"**{format_percent(probability)}**"
                    )

                    st.progress(
                        max(0, min(100, round(probability * 100)))
                    )

                    st.write(
                        "Позиционирование: "
                        f"**{format_percent(result['position_probability'])}**"
                    )

                    st.write(
                        "Отклонение оси: "
                        f"**{format_percent(result['axis_probability'])}**"
                    )

                    st.write(
                        "Артефакты: "
                        f"**{format_percent(result['artifacts_probability'])}**"
                    )

                    st.caption(
                        f"Размер: {result['rows']} × "
                        f"{result['columns']} · "
                        f"Modality: {result['modality']}"
                    )

    else:
        st.info("Успешно обработанных изображений нет.")

    # ---------- ОШИБКИ ----------

    failed_rows = [
        row for row in report_rows
        if row["processing_status"] == "Failure"
    ]

    if failed_rows:
        st.divider()
        st.markdown(
            '<div class="section-label">ОШИБКИ ОБРАБОТКИ</div>',
            unsafe_allow_html=True,
        )

        for row in failed_rows:
            with st.expander(
                f"Ошибка: {Path(row['path_to_study']).name} "
                f"· {row['image_uid'] or 'UID отсутствует'}",
                expanded=False,
            ):
                st.write(f"Папка: {row['path_to_study']}")
                st.write(f"Study UID: {row['study_uid'] or 'Нет'}")
                st.write(f"Image UID: {row['image_uid'] or 'Нет'}")
                st.error(row["error"] or "Неизвестная ошибка")

    # ---------- CSV-ОТЧЁТ ----------

    st.divider()
    st.markdown(
        '<div class="section-label">ОТЧЁТ</div>',
        unsafe_allow_html=True,
    )

    st.write("DEBUG: CSV =", st.session_state["csv_report"] is not None)
    st.write("DEBUG: Имя файла =", st.session_state["csv_filename"])
    if st.session_state["csv_report"] is not None:
        st.write(
            "Отчёт содержит по одной строке на каждый найденный "
            "DICOM-файл, включая файлы с ошибками."
        )

        st.download_button(
            label="📥 Скачать CSV-отчёт",
            data=st.session_state["csv_report"],
            file_name=st.session_state["csv_filename"],
            mime="text/csv",
            use_container_width=True,
            key="download_csv_report",
        )

    st.divider()
    st.caption(
        "Вероятности — выходы модели, а не клинически "
        "калиброванные оценки риска. Итог исследования "
        "рассчитывается по демонстрационному правилу большинства. "
        "Прототип не предназначен для постановки диагноза."
    )