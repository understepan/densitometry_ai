from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_manifest.csv"
)


df = pd.read_csv(PATH)

print("=" * 70)
print("ПРОВЕРКА LABELED MANIFEST")
print("=" * 70)

print()

print("Файл:")
print(PATH)

print()

print("Количество строк:")
print(len(df))

print()

print("Колонки:")
for i, col in enumerate(df.columns):
    print(
        f"{i + 1:2d}. {col}"
    )

print()

print("Первые 3 строки:")

print(
    df.head(3).to_string(
        index=False
    )
)

print()

print("=" * 70)