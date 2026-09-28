"""
Script de demonstração de inspeção industrial automatizada.

Executa o pipeline completo em amostras reais de cada classe:
1. Identificação do ativo via DINOv2
2. Localização geométrica do ponto quente (Hotspot)
3. Diagnóstico de severidade pela ABNT NBR 15572 / NFPA 70B
4. Exportação do mosaico de inspeção anotado para docs/figures/hotspot_inspections.png
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.industrial_analyzer import IndustrialThermalAnalyzer

SAMPLE_IMAGES = [
    ("Circuit Breakers", Path("data/raw/Circuit Breakers/FLIR2296.jpg")),
    ("Disconnectors", Path("data/raw/Disconnectors/FLIR2284.jpg")),
    ("Power Transformers", Path("data/raw/Power Transformers/FLIR0150.jpg")),
    ("Surge Arresters", Path("data/raw/Surge Arresters/FLIR2572.jpg")),
    ("Wave Traps", Path("data/raw/Wave Traps/FLIR2556.jpg")),
]


def run_demonstration():
    model_path = Path("outputs/models/classifier_dinov2.pkl")
    analyzer = IndustrialThermalAnalyzer(model_path=model_path)

    fig, axes = plt.subplots(1, len(SAMPLE_IMAGES), figsize=(20, 5), dpi=300)

    print("\n" + "=" * 80)
    print("      RELATÓRIO DE INSPEÇÃO TERMOGRÁFICA INDUSTRIAL (NBR 15572 / NFPA 70B)")
    print("=" * 80)

    for idx, (expected_class, img_path) in enumerate(SAMPLE_IMAGES):
        report = analyzer.inspect_image(img_path)

        print(f"\n[ATIVO {idx+1}]: {expected_class}")
        print(f"  • Identificado pelo DINOv2: {report.equipment_class} ({report.classification_confidence*100:.1f}%)")
        print(f"  • Coordenada do Ponto Quente: X={report.hotspot.peak_coordinate[0]}, Y={report.hotspot.peak_coordinate[1]}")
        print(f"  • Bounding Box do Hotspot: {report.hotspot.bounding_box} (Área: {report.hotspot.area_pixels} px)")
        print(f"  • Elevação Térmica Estimada (Delta T): +{report.diagnosis.delta_t_celsius:.1f}°C")
        print(f"  • Severidade NBR 15572: {report.diagnosis.severity.value}")
        print(f"  • Ação Recomendada: {report.diagnosis.recommended_action}")

        # Converter BGR para RGB para plotar com matplotlib
        annotated_rgb = cv2.cvtColor(report.annotated_image, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(annotated_rgb)
        axes[idx].set_title(f"{report.equipment_class}\n{report.diagnosis.severity.value}", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("Inspeção Termográfica Industrial Automatizada (DINOv2 + Detecção de Hotspots + NBR 15572)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    output_path = Path("docs/figures/hotspot_inspections.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print(f"Mosaico visual de inspeção exportado para: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demonstration()
