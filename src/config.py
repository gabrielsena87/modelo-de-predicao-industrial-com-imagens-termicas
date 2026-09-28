"""
Configuração centralizada do pipeline DINOv2.

Todos os hiperparâmetros, caminhos e constantes de normalização ficam
nesta dataclass imutável (frozen=True), eliminando números mágicos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DINOV2_PATCH_SIZE: int = 14


@dataclass(frozen=True)
class DINOv2Config:
    """Configuração imutável para o pipeline de pré-processamento DINOv2.

    Attributes:
        data_dir: Diretório raiz contendo subpastas por classe.
        image_size: Resolução espacial alvo (height = width). Deve ser
            múltiplo positivo de ``patch_size``.
        patch_size: Tamanho do patch do DINOv2 em pixels.
        mean: Média por canal para normalização (R, G, B).
        std: Desvio padrão por canal para normalização (R, G, B).
        batch_size: Número de imagens por mini-batch.
        num_workers: Workers paralelos de carregamento. ``0`` desabilita
            multiprocessing (útil para debugging).
        pin_memory: Fixar memória do host para transferências GPU mais rápidas.
        device: Dispositivo de computação ('cuda' ou 'cpu').
    """

    data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    image_size: int = 224
    patch_size: int = DINOV2_PATCH_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"
    supported_extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    )

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) deve ser múltiplo "
                f"positivo de patch_size ({self.patch_size})."
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size deve ser positivo, recebeu {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers deve ser >= 0, recebeu {self.num_workers}.")
