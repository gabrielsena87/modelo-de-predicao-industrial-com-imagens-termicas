"""
Testes unitários automatizados para o subsistema de termometria e diagnóstico normativo.

Princípios F.I.R.S.T. (Clean Code):
- Fast: Execução em menos de 1 segundo para testes puramente analíticos.
- Independent: Cada caso de teste é autossuficiente e não compartilha estado mutável.
- Repeatable: Resultados determinísticos independentes de ambiente externo.
- Self-Validating: Asserções booleanas estritas (pass/fail).
- Timely: Validação paralela ao desenvolvimento dos novos módulos.
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from src.hotspot_detector import HotspotDetector, HotspotLocation
from src.normative_rules import NormativeThermalEngine, SeverityLevel, ThermalDiagnosis


class TestHotspotDetector(unittest.TestCase):
    """Bateria de testes unitários para detecção geométrica de anomalias térmicas."""

    def setUp(self) -> None:
        self.detector = HotspotDetector(blur_kernel_size=5)

    def test_locates_synthetic_peak_correctly(self) -> None:
        """Valida se o detector localiza a coordenada exata do pixel mais quente."""
        image_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
        image_bgr[:, :] = [40, 10, 10]  # Fundo frio

        target_x, target_y = 120, 80
        cv2.circle(image_bgr, (target_x, target_y), radius=8, color=(255, 255, 255), thickness=-1)

        hotspot = self.detector.locate_hotspot(image_bgr)

        self.assertLessEqual(abs(hotspot.peak_coordinate[0] - target_x), 2)
        self.assertLessEqual(abs(hotspot.peak_coordinate[1] - target_y), 2)
        self.assertGreater(hotspot.peak_intensity, 0.90)
        self.assertGreater(hotspot.relative_delta, 0.50)

    def test_rejects_even_kernel_size_with_value_error(self) -> None:
        """Garante que tamanhos pares de kernel disparam ValueError."""
        with self.assertRaises(ValueError):
            HotspotDetector(blur_kernel_size=4)

    def test_drawing_preserves_canvas_properties(self) -> None:
        """Garante que a renderização visual preserva shape e dtype do canvas."""
        canvas = np.zeros((100, 150, 3), dtype=np.uint8)
        dummy_hotspot = HotspotLocation(
            peak_coordinate=(50, 50),
            bounding_box=(40, 40, 20, 20),
            area_pixels=400,
            peak_intensity=1.0,
            mean_intensity=0.8,
            background_intensity=0.2,
            relative_delta=0.8,
        )

        rendered = self.detector.draw_hotspot(canvas, dummy_hotspot)
        self.assertEqual(rendered.shape, canvas.shape)
        self.assertEqual(rendered.dtype, canvas.dtype)


class TestNormativeThermalEngine(unittest.TestCase):
    """Bateria de testes unitários para as regras normativas NBR 15572 e NFPA 70B."""

    def test_classifies_all_nbr_15572_thresholds(self) -> None:
        """Valida os intervalos de severidade previstos na norma."""
        diag_normal = NormativeThermalEngine.evaluate_by_delta_t("Power Transformers", 6.5)
        self.assertEqual(diag_normal.severity, SeverityLevel.NORMAL)

        diag_attention = NormativeThermalEngine.evaluate_by_delta_t("Circuit Breakers", 18.0)
        self.assertEqual(diag_attention.severity, SeverityLevel.ATTENTION)

        diag_severe = NormativeThermalEngine.evaluate_by_delta_t("Disconnectors", 35.0)
        self.assertEqual(diag_severe.severity, SeverityLevel.SEVERE)

        diag_critical = NormativeThermalEngine.evaluate_by_delta_t("Wave Traps", 55.0)
        self.assertEqual(diag_critical.severity, SeverityLevel.CRITICAL)

    def test_action_text_contains_asset_and_severity(self) -> None:
        """Garante a formatação adequada da ordem de ação gerada."""
        diag = NormativeThermalEngine.evaluate_by_delta_t("Surge Arresters", 60.0)
        self.assertIn("Surge Arresters", diag.recommended_action)
        self.assertIn("PERIGO IMINENTE", diag.recommended_action)
        self.assertEqual(diag.color_bgr, (0, 0, 230))  # Vermelho para crítico


class TestIndustrialThermalAnalyzer(unittest.TestCase):
    """Teste de integração ponta-a-ponta do analisador industrial."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.industrial_analyzer import IndustrialThermalAnalyzer
        model_path = Path("outputs/models/classifier_dinov2.pkl")
        if not model_path.exists():
            raise unittest.SkipTest("Modelo pré-treinado não localizado em outputs/models/")
        cls.analyzer = IndustrialThermalAnalyzer(model_path=model_path)

    def test_end_to_end_inspection_returns_valid_report(self) -> None:
        """Garante que uma imagem real gera um relatório normativo completo e consistente."""
        test_image = Path("data/raw/Power Transformers/FLIR0150.jpg")
        if not test_image.exists():
            raise unittest.SkipTest("Imagem de teste não encontrada.")

        report = self.analyzer.inspect_image(test_image)

        self.assertEqual(report.equipment_class, "Power Transformers")
        self.assertGreaterEqual(report.classification_confidence, 0.90)
        self.assertIsInstance(report.diagnosis.severity, SeverityLevel)
        self.assertGreater(report.hotspot.area_pixels, 0)
        self.assertEqual(len(report.annotated_image.shape), 3)


if __name__ == "__main__":
    unittest.main()
