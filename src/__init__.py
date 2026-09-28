"""
Módulo principal do pipeline de predição industrial com imagens térmicas.

Funcionalidades:
- Extração de features visuais via DINOv2 ViT-B/14 (Meta AI).
- Identificação de ativos industriais em alta tensão.
- Detecção geométrica e fotométrica de pontos quentes (Hotspots).
- Diagnóstico normativo de severidade térmica conforme ABNT NBR 15572 e NFPA 70B.
"""

from .config import DINOv2Config, IMAGENET_MEAN, IMAGENET_STD
from .dataset import ThermalImageDataset, safe_collate_fn
from .feature_extractor import DINOv2FeatureExtractor
from .hotspot_detector import HotspotDetector, HotspotLocation
from .industrial_analyzer import IndustrialThermalAnalyzer, InspectionReport
from .normative_rules import NormativeThermalEngine, SeverityLevel, ThermalDiagnosis
from .transforms import build_transforms

__all__ = [
    "DINOv2Config",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transforms",
    "ThermalImageDataset",
    "safe_collate_fn",
    "DINOv2FeatureExtractor",
    "HotspotDetector",
    "HotspotLocation",
    "SeverityLevel",
    "ThermalDiagnosis",
    "NormativeThermalEngine",
    "IndustrialThermalAnalyzer",
    "InspectionReport",
]
