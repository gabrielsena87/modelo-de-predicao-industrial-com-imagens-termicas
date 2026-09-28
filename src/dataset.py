"""
Dataset customizado para imagens térmicas com carregamento resiliente.

Responsabilidades (SRP):
    - Escanear subdiretórios por classe (ImageFolder-style).
    - Carregar imagem, converter para RGB, aplicar transform.
    - Ignorar imagens corrompidas sem quebrar o pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .config import DINOv2Config

logger = logging.getLogger(__name__)


class ThermalImageDataset(Dataset):
    """Dataset de imagens térmicas organizado por subpastas (uma por classe).

    Varre ``config.data_dir`` recursivamente, indexando arquivos de imagem
    válidos e mapeando nomes de subpasta para índices de classe.
    """

    def __init__(self, config: DINOv2Config, transform: Optional[Compose] = None) -> None:
        if not config.data_dir.is_dir():
            raise FileNotFoundError(f"Diretório não encontrado: {config.data_dir}")

        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: dict[str, int] = {}

        # Descobrir classes a partir das subpastas
        class_dirs = sorted(
            [d for d in config.data_dir.iterdir() if d.is_dir()]
        )
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Indexar imagens
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in config.supported_extensions:
                    self.samples.append((img_path, class_idx))

        logger.info(
            "Indexadas %d imagens em %d classes de '%s'.",
            len(self.samples), len(self.classes), config.data_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Optional[Tuple[Tensor, int, str]]:
        """Retorna (tensor, label, path) ou None se corrompida."""
        img_path, label = self.samples[index]
        try:
            with Image.open(img_path) as img:
                img.load()
                image = img.convert("RGB")
        except (OSError, SyntaxError, ValueError) as exc:
            logger.warning("Ignorando imagem corrompida '%s': %s", img_path, exc)
            return None

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(img_path)


def safe_collate_fn(batch):
    """Filtra amostras None (imagens corrompidas) do batch."""
    valid = [s for s in batch if s is not None]
    if not valid:
        return None
    tensors, labels, paths = zip(*valid)
    return torch.stack(tensors), torch.tensor(labels), paths
