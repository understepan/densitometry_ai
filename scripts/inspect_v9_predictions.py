from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATH = (
    PROJECT_ROOT
    / "results"
    / "spine_quality_v9_exact_predictions.csv"
)


def main():

    print("=" * 80)
    print("СТРУКТУРА V9 PREDICTIONS")
    print("=" * 80)

    df = pd.read_csv(PATH)

    print()
    print("Файл:")
    print(PATH)

    print()
    print("Количество строк:", len(df))

    print()
    print("СТОЛБЦЫ:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:02d}. {column}")

    print()
    print("=" * 80)
    print("ПЕРВЫЕ 5 СТРОК")
    print("=" * 80)

    print(
        df.head().to_string(index=False)
    )

    print()
    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)


if __name__ == "__main__":
    main()