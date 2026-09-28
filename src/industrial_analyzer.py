"""
Orquestrador industrial de inspeção termográfica automatizada.

Responsabilidade Única (SRP):
Integrar a classificação semântica DINOv2 (identificação do ativo),
a detecção geométrica do ponto de calor (HotspotDetector) e
a avaliação normativa de risco (NormativeThermalEngine) em um fluxo único.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from .config import DINOv2Config
from .feature_extractor import DINOv2FeatureExtractor
from .hotspot_detector import HotspotDetector, HotspotLocation
from .normative_rules import NormativeThermalEngine, ThermalDiagnosis
from .transforms import build_transforms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectionReport:
    """Relatório consolidado de inspeção térmica de um equipamento.

    Attributes:
        equipment_class: Classe do equipamento identificada pelo DINOv2.
        classification_confidence: Probabilidade associada à classificação (0.0 a 1.0).
        hotspot: Dados geométricos e fotométricos do ponto quente identificado.
        diagnosis: Diagnóstico de severidade e ação recomendada segundo a NBR 15572.
        annotated_image: Imagem com overlays de mira, bounding box e cartão de status.
    """

    equipment_class: str
    classification_confidence: float
    hotspot: HotspotLocation
    diagnosis: ThermalDiagnosis
    annotated_image: np.ndarray


class IndustrialThermalAnalyzer:
    """Sistema integrado de diagnóstico preditivo industrial com DINOv2 e termometria."""

    def __init__(
        self,
        model_path: Path,
        config: DINOv2Config | None = None,
        hotspot_detector: HotspotDetector | None = None,
    ) -> None:
        """Inicializa o analisador carregando o extrator DINOv2 e o classificador treinado.

        Args:
            model_path: Caminho do arquivo .pkl contendo o pipeline treinado.
            config: Configuração do pipeline (instanciada por padrão se omitida).
            hotspot_detector: Instância do detector de pontos quentes (ou padrão).
        """
        self.config = config or DINOv2Config()
        self.transform = build_transforms(self.config)
        self.extractor = DINOv2FeatureExtractor(self.config)
        self.hotspot_detector = hotspot_detector or HotspotDetector()

        self.classifier_pipeline, self.classes = self._load_classifier(model_path)
        logger.info(f"IndustrialThermalAnalyzer inicializado com sucesso para {len(self.classes)} classes.")

    def inspect_image(
        self, image_input: Union[str, Path, np.ndarray], estimated_span_celsius: float = 50.0
    ) -> InspectionReport:
        """Executa a inspeção completa de uma imagem térmica.

        Etapas:
        1. Carrega e valida a imagem em memória (BGR).
        2. Extrai embeddings DINOv2 e infere a classe do ativo com confiança.
        3. Localiza geometricamente o ponto mais quente (Hotspot) via segmentação espectral.
        4. Avalia a severidade conforme a ABNT NBR 15572 / NFPA 70B.
        5. Gera a visualização gráfica anotada da inspeção.
        """
        image_bgr = self._resolve_bgr_image(image_input)

        # 1. Identificação do Equipamento via DINOv2
        equipment, confidence = self._classify_equipment(image_bgr)

        # 2. Localização Geométrica do Ponto Quente
        hotspot = self.hotspot_detector.locate_hotspot(image_bgr)

        # 3. Diagnóstico Normativo (NBR 15572 / NFPA 70B)
        diagnosis = NormativeThermalEngine.estimate_from_rgb_contrast(
            equipment_name=equipment,
            relative_delta=hotspot.relative_delta,
            estimated_span_celsius=estimated_span_celsius,
        )

        # 4. Renderização do Painel Visual de Inspeção
        annotated = self._render_visual_report(image_bgr, equipment, confidence, hotspot, diagnosis)

        return InspectionReport(
            equipment_class=equipment,
            classification_confidence=confidence,
            hotspot=hotspot,
            diagnosis=diagnosis,
            annotated_image=annotated,
        )

    def _classify_equipment(self, image_bgr: np.ndarray) -> tuple[str, float]:
        """Extrai características com o DINOv2 e calcula a classe mais provável."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        tensor = self.transform(pil_img)

        embedding = self.extractor.extract_single(tensor).reshape(1, -1)
        probabilities = self.classifier_pipeline.predict_proba(embedding)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        return self.classes[best_index], confidence

    @staticmethod
    def _resolve_bgr_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Garante que a entrada seja convertida em uma matriz numpy BGR válida."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_input}")
            return img
        if isinstance(image_input, np.ndarray):
            return image_input.copy()
        raise TypeError(f"Tipo de entrada não suportado: {type(image_input)}")

    @staticmethod
    def _load_classifier(path: Path) -> tuple[object, list[str]]:
        """Carrega o artefato de modelo previamente treinado."""
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de modelo não encontrado em: {path}")
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        return artifact["pipeline"], artifact["classes"]

    @classmethod
    def _render_visual_report(
        cls,
        canvas_bgr: np.ndarray,
        equipment: str,
        confidence: float,
        hotspot: HotspotLocation,
        diagnosis: ThermalDiagnosis,
    ) -> np.ndarray:
        """Desenha a mira de precisão, contorno de anomalia e cartão normativo superior."""
        annotated = HotspotDetector.draw_hotspot(canvas_bgr, hotspot, color_bgr=diagnosis.color_bgr)
        h, w = annotated.shape[:2]

        # Painel superior semitransparente para status
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
        annotated = cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0)

        # Linha 1: Ativo identificado e confiança
        title_text = f"Ativo: {equipment.upper()} ({confidence*100:.1f}%)"
        cv2.putText(annotated, title_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Linha 2: Severidade normativa e elevação térmica estimada
        status_text = f"NBR 15572: {diagnosis.severity.value} | Delta T est: +{diagnosis.delta_t_celsius:.1f}C"
        cv2.putText(annotated, status_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, diagnosis.color_bgr, 2)

        return annotated
