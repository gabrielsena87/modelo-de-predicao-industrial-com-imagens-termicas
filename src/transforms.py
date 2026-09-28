"""
Fábrica de transformações para o pipeline DINOv2.

Responsabilidade única: construir o pipeline torchvision.transforms.Compose.
"""
from __future__ import annotations

from torchvision import transforms as T

from .config import DINOv2Config


def build_transforms(config: DINOv2Config) -> T.Compose:
    """Constrói o pipeline canônico de transformação DINOv2.

    Fluxo:
        1. Resize para (image_size x image_size) com interpolação bicúbica.
        2. ToTensor: PIL (H×W×C, 0-255) → FloatTensor (C×H×W, 0.0-1.0).
        3. Normalize com estatísticas ImageNet.
    """
    return T.Compose([
        T.Resize(
            size=(config.image_size, config.image_size),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.ToTensor(),
        T.Normalize(mean=config.mean, std=config.std),
    ])
