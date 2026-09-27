import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SpineQualityResNet18(nn.Module):
    """
    ResNet18 для бинарной классификации spine_quality.

    На вход:
        [batch, 1, 224, 224]

    На выход:
        [batch, 2]

    Классы:
        0
        1
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = resnet18(weights=weights)

        # Исходный ResNet18 ожидает 3 канала RGB.
        # Наши DICOM после preprocessing имеют 1 канал.
        # Поэтому первый слой оставляем стандартным,
        # а в forward повторяем grayscale-канал 3 раза.

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            2
        )

    def forward(self, x):
        # x: [B, 1, H, W]

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Нормализация под ImageNet,
        # на котором предварительно обучалась ResNet18.
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=x.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=x.device
        ).view(1, 3, 1, 1)

        x = (x - mean) / std

        return self.model(x)