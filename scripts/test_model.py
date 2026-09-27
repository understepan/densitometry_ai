from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.resnet18_model import SpineQualityResNet18


print("Создание модели...")

model = SpineQualityResNet18(pretrained=True)

print("Модель создана.")
print(model)

print()
print("Проверка входного изображения...")

x = torch.randn(2, 1, 224, 224)

with torch.no_grad():
    output = model(x)

print(f"Размер входа:  {tuple(x.shape)}")
print(f"Размер выхода: {tuple(output.shape)}")

assert output.shape == (2, 2)

print()
print("ВСЕ ПРОВЕРКИ МОДЕЛИ ПРОЙДЕНЫ.")