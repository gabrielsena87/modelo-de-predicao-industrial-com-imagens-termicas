"""
Motor de regras normativas para diagnóstico de severidade térmica.

Responsabilidade Única (SRP):
Aplicar os critérios e tabelas de decisão das normas ABNT NBR 15572,
NBR 15718 e NFPA 70B para converter elevações térmicas (Delta T)
em diagnósticos de severidade e recomendações operacionais de manutenção.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class SeverityLevel(str, Enum):
    """Níveis de severidade térmica padronizados pela NBR 15572 e NFPA 70B."""

    NORMAL = "Normal (Aceitável)"
    ATTENTION = "Atenção (Alerta Inicial)"
    SEVERE = "Grave (Ação Programada)"
    CRITICAL = "Crítico (Intervenção Imediata)"


@dataclass(frozen=True)
class ThermalDiagnosis:
    """Diagnóstico normativo completo de um ativo inspecionado.

    Attributes:
        equipment: Nome do equipamento identificado pelo classificador.
        severity: Nível de severidade normativo atribuído.
        delta_t_celsius: Gradiente de temperatura medido ou estimado (°C).
        color_bgr: Código de cor BGR para sinalização visual na interface gráfica.
        recommended_action: Ação de manutenção recomendada pela norma.
        standard_reference: Norma técnica de referência utilizada.
    """

    equipment: str
    severity: SeverityLevel
    delta_t_celsius: float
    color_bgr: Tuple[int, int, int]
    recommended_action: str
    standard_reference: str


class NormativeThermalEngine:
    """Motor de avaliação de severidade segundo NBR 15572 e NFPA 70B."""

    # Paleta de cores industrial para UX (BGR para OpenCV)
    SEVERITY_COLORS = {
        SeverityLevel.NORMAL: (0, 180, 0),        # Verde
        SeverityLevel.ATTENTION: (0, 215, 255),   # Amarelo
        SeverityLevel.SEVERE: (0, 120, 255),      # Laranja
        SeverityLevel.CRITICAL: (0, 0, 230),      # Vermelho
    }

    # Ações normativas recomendadas pela NBR 15572 / NFPA 70B
    RECOMMENDED_ACTIONS = {
        SeverityLevel.NORMAL: (
            "Operação em condições térmicas adequadas. Manter rotina de inspeção periódica."
        ),
        SeverityLevel.ATTENTION: (
            "Indício de início de sobreaquecimento por resistência de contato ou carga. "
            "Reinspecionar na próxima ronda e programar manutenção na próxima parada."
        ),
        SeverityLevel.SEVERE: (
            "Aquecimento significativo. Risco de degradação acelerada de isoladores e contatos. "
            "Programar intervenção corretiva em até 48 horas e monitorar corrente de carga."
        ),
        SeverityLevel.CRITICAL: (
            "PERIGO IMINENTE! Severidade crítica com risco de fusão, arco elétrico ou queima do ativo. "
            "Intervenção de campo imediata e redução/desligamento de carga preventiva."
        ),
    }

    @classmethod
    def evaluate_by_delta_t(
        cls, equipment_name: str, delta_t_celsius: float
    ) -> ThermalDiagnosis:
        """Avalia a severidade com base no Delta T exato (°C) segundo a NBR 15572."""
        severity = cls._classify_delta_t(delta_t_celsius)
        action = cls._build_action_text(equipment_name, severity)
        color = cls.SEVERITY_COLORS[severity]

        return ThermalDiagnosis(
            equipment=equipment_name,
            severity=severity,
            delta_t_celsius=round(delta_t_celsius, 2),
            color_bgr=color,
            recommended_action=action,
            standard_reference="ABNT NBR 15572 / NFPA 70B",
        )

    @classmethod
    def estimate_from_rgb_contrast(
        cls, equipment_name: str, relative_delta: float, estimated_span_celsius: float = 50.0
    ) -> ThermalDiagnosis:
        """Estima o Delta T e a severidade a partir do contraste térmico relativo da imagem RGB.

        Em imagens puramente RGB de paleta termográfica (como FLIR Ironbow),
        o Delta relativo [0.0, 1.0] entre o ponto de calor máximo e o fundo
        é extrapolado proporcionalmente para a faixa dinâmica típica de inspeções elétricas.
        """
        estimated_delta_t = relative_delta * estimated_span_celsius
        diagnosis = cls.evaluate_by_delta_t(equipment_name, estimated_delta_t)

        return ThermalDiagnosis(
            equipment=diagnosis.equipment,
            severity=diagnosis.severity,
            delta_t_celsius=diagnosis.delta_t_celsius,
            color_bgr=diagnosis.color_bgr,
            recommended_action=diagnosis.recommended_action,
            standard_reference="ABNT NBR 15572 (Estimativa Fotométrica RGB)",
        )

    @staticmethod
    def _classify_delta_t(delta_t: float) -> SeverityLevel:
        """Mapeia os limites clássicos de temperatura da NBR 15572 / NFPA 70B."""
        if delta_t < 10.0:
            return SeverityLevel.NORMAL
        if delta_t < 25.0:
            return SeverityLevel.ATTENTION
        if delta_t < 50.0:
            return SeverityLevel.SEVERE
        return SeverityLevel.CRITICAL

    @classmethod
    def _build_action_text(cls, equipment: str, severity: SeverityLevel) -> str:
        base_action = cls.RECOMMENDED_ACTIONS[severity]
        return f"[{equipment}] {base_action}"
