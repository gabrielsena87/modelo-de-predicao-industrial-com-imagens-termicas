"""
Módulo principal do pipeline de predição industrial com imagens térmicas.

Extrai features visuais via DINOv2 ViT-B/14 (Meta AI) e treina
classificadores para identificar equipamentos elétricos em imagens
térmicas infravermelhas.
"""

from .config import DINOv2Config, IMAGENET_MEAN, IMAGENET_STD
from .transforms import build_transforms
from .dataset import ThermalImageDataset, safe_collate_fn
from .feature_extractor import DINOv2FeatureExtractor

__all__ = [
    "DINOv2Config",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transforms",
    "ThermalImageDataset",
    "safe_collate_fn",
    "DINOv2FeatureExtractor",
]
