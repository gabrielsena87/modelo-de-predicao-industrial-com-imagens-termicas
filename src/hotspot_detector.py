"""
Detector geométrico de pontos quentes (Hotspots) para imagens térmicas RGB.

Responsabilidade Única (SRP):
Analisar a distribuição espectral de luminância e matiz de imagens térmicas
RGB (paletas industriais como Ironbow, Rainbow ou Lava) e extrair com
precisão a coordenada (x, y), contorno e intensidade do ponto mais quente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HotspotLocation:
    """Dados geométricos e fotométricos imutáveis do ponto quente detectado.

    Attributes:
        peak_coordinate: Ponto cartesiano (x, y) de temperatura máxima relativa.
        bounding_box: Retângulo delimitador (x, y, largura, altura) da região crítica.
        area_pixels: Quantidade de pixels na área do ponto quente.
        peak_intensity: Intensidade normalizada do pico (0.0 a 1.0).
        mean_intensity: Intensidade média da região do ponto quente (0.0 a 1.0).
        background_intensity: Intensidade média de referência do fundo (0.0 a 1.0).
        relative_delta: Elevação normalizada (peak - background).
    """

    peak_coordinate: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    area_pixels: int
    peak_intensity: float
    mean_intensity: float
    background_intensity: float
    relative_delta: float


class HotspotDetector:
    """Localizador de anomalias térmicas e pontos quentes em imagens termográficas RGB.

    Nas paletas termográficas industriais, as temperaturas elevadas são mapeadas
    para combinações de alta luminância e canais quentes (amarelo/branco).
    Este detector calcula o mapa de calor relativo e segmenta a região crítica.
    """

    def __init__(self, blur_kernel_size: int = 5) -> None:
        """Inicializa o detector com suavização para rejeição de ruído de sensor.

        Args:
            blur_kernel_size: Tamanho ímpar da janela do filtro Gaussiano.
        """
        if blur_kernel_size % 2 == 0 or blur_kernel_size < 1:
            raise ValueError("blur_kernel_size deve ser um inteiro ímpar positivo.")
        self._blur_kernel_size = blur_kernel_size

    def compute_thermal_intensity_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """Calcula o mapa escalar de calor normalizado [0.0, 1.0] a partir da imagem BGR.

        Combina a luminância perceptual com a saturação de tons quentes (R e G)
        típicos das paletas FLIR Ironbow e Rainbow.
        """
        blurred = cv2.GaussianBlur(image_bgr, (self._blur_kernel_size, self._blur_kernel_size), 0)
        blue = blurred[:, :, 0].astype(np.float32)
        green = blurred[:, :, 1].astype(np.float32)
        red = blurred[:, :, 2].astype(np.float32)

        # Nas paletas FLIR:
        # Frio = preto/azul/roxo (Blue dominante, Red/Green baixos)
        # Quente = vermelho/laranja (Red alto, Green médio)
        # Pico = amarelo/branco (Red e Green saturados, Blue sobe no branco puro)
        perceptual_luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        warmth_bias = np.maximum(0.0, (red + green) - 1.2 * blue)

        intensity = 0.6 * perceptual_luminance + 0.4 * warmth_bias
        min_val, max_val = intensity.min(), intensity.max()

        if max_val - min_val > 1e-5:
            normalized_intensity = (intensity - min_val) / (max_val - min_val)
        else:
            normalized_intensity = np.zeros_like(intensity)

        return normalized_intensity

    def locate_hotspot(
        self, image_bgr: np.ndarray, elevation_ratio: float = 0.60
    ) -> HotspotLocation:
        """Localiza geometricamente o ponto de calor máximo e sua região de influência.

        Args:
            image_bgr: Imagem térmica em formato BGR (padrão OpenCV).
            elevation_ratio: Proporção [0.1, 0.9] de elevação acima do fundo para corte do hotspot.

        Returns:
            Instância de HotspotLocation com coordenadas e contrastes calculados.
        """
        intensity_map = self.compute_thermal_intensity_map(image_bgr)
        peak_y, peak_x = np.unravel_index(np.argmax(intensity_map), intensity_map.shape)
        peak_value = float(intensity_map[peak_y, peak_x])
        background_intensity = float(np.median(intensity_map))

        delta_elevation = max(0.0, peak_value - background_intensity)
        threshold_value = background_intensity + elevation_ratio * delta_elevation

        hotspot_mask = (intensity_map >= threshold_value).astype(np.uint8) * 255

        # Refinamento morfológico para unir componentes do mesmo ponto quente
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Seleciona o contorno que engloba a maior concentração térmica
            chosen_contour = self._select_contour_enclosing_peak(contours, (peak_x, peak_y))
            x, y, w, h = cv2.boundingRect(chosen_contour)
            area = int(cv2.contourArea(chosen_contour))

            # Centróide geométrico do ponto de calor
            moments = cv2.moments(chosen_contour)
            if moments["m00"] > 0:
                center_x = int(round(moments["m10"] / moments["m00"]))
                center_y = int(round(moments["m01"] / moments["m00"]))
            else:
                center_x, center_y = int(x + w / 2), int(y + h / 2)
        else:
            x, y, w, h = max(0, peak_x - 5), max(0, peak_y - 5), 10, 10
            area = 100
            center_x, center_y = peak_x, peak_y

        roi_intensity = intensity_map[y : y + h, x : x + w]
        mean_hotspot = float(np.mean(roi_intensity))
        background_intensity = float(np.median(intensity_map))
        relative_delta = max(0.0, peak_value - background_intensity)

        return HotspotLocation(
            peak_coordinate=(center_x, center_y),
            bounding_box=(int(x), int(y), int(w), int(h)),
            area_pixels=area,
            peak_intensity=peak_value,
            mean_intensity=mean_hotspot,
            background_intensity=background_intensity,
            relative_delta=relative_delta,
        )

    @staticmethod
    def _select_contour_enclosing_peak(
        contours: list[np.ndarray], peak_point: Tuple[int, int]
    ) -> np.ndarray:
        """Retorna o contorno que contém o ponto de pico ou o maior contorno."""
        px, py = peak_point
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (float(px), float(py)), measureDist=False) >= 0:
                return cnt
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def draw_hotspot(
        canvas_bgr: np.ndarray,
        hotspot: HotspotLocation,
        color_bgr: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """Desenha a mira gráfica, o contorno e os dados do ponto quente na imagem."""
        annotated = canvas_bgr.copy()
        px, py = hotspot.peak_coordinate
        x, y, w, h = hotspot.bounding_box

        # Caixa delimitadora da anomalia térmica
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2)

        # Mira de precisão no pixel de maior temperatura relativa
        cross_size = 8
        cv2.line(annotated, (px - cross_size, py), (px + cross_size, py), color_bgr, 2)
        cv2.line(annotated, (px, py - cross_size), (px, py + cross_size), color_bgr, 2)
        cv2.circle(annotated, (px, py), 4, color_bgr, 1)

        return annotated
