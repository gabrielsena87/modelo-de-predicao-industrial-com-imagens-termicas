"""
Extrator de features DINOv2 com suporte a CUDA.

Encapsula o backbone DINOv2 ViT-B/14 (Meta AI) para extrair
embeddings visuais de 768 dimensões a partir de imagens térmicas.
"""
from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from .config import DINOv2Config
from .dataset import ThermalImageDataset, safe_collate_fn
from .transforms import build_transforms

logger = logging.getLogger(__name__)


class DINOv2FeatureExtractor:
    """Extrai embeddings do CLS token do DINOv2 ViT-B/14.

    O modelo é carregado via torch.hub e movido para o device
    configurado (CUDA se disponível).

    Attributes:
        model: Backbone DINOv2 em modo de avaliação.
        device: Dispositivo de computação ativo.
        config: Configuração do pipeline.
    """

    EMBEDDING_DIM: int = 768  # ViT-B/14

    def __init__(self, config: DINOv2Config) -> None:
        self.config = config

        # Resolver device: usar CUDA se disponível e configurado
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA ativado: %s", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            if config.device == "cuda":
                logger.warning("CUDA solicitado mas indisponível. Usando CPU.")

        # Carregar backbone DINOv2 ViT-B/14 do torch hub
        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("DINOv2 ViT-B/14 carregado no device '%s'.", self.device)

    @torch.no_grad()
    def extract_single(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extrai embedding de um único tensor de imagem.

        Args:
            image_tensor: Tensor [C, H, W] ou [1, C, H, W] normalizado.

        Returns:
            Array numpy de shape (768,).
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        features = self.model(image_tensor)
        return features.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_from_dataset(
        self, dataset: ThermalImageDataset
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """Extrai features de todo o dataset.

        Args:
            dataset: Dataset de imagens térmicas já transformadas.

        Returns:
            Tupla (features, labels, paths) onde:
                - features: ndarray shape (N, 768)
                - labels: ndarray shape (N,)
                - paths: lista de caminhos das imagens
        """
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=safe_collate_fn,
            drop_last=False,
        )

        all_features = []
        all_labels = []
        all_paths = []

        for batch in loader:
            if batch is None:
                continue

            images, labels, paths = batch
            images = images.to(self.device)

            features = self.model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)

        logger.info(
            "Extraídas %d features (dim=%d) do dataset.",
            features_array.shape[0], features_array.shape[1],
        )

        return features_array, labels_array, all_paths
