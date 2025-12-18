"""
QBitaLabs Digital Twin Engine (QBita Twin™)

Biological Digital Twin platform for:
- Patient-specific disease modeling
- Drug response prediction
- Treatment optimization
- Aging trajectory simulation

Features:
- Multi-scale physiological models
- Quantum-accurate molecular dynamics
- Machine learning integration
- Real-time health monitoring
"""

from __future__ import annotations

from qbitalabs.digital_twin.engine import (
    DigitalTwinEngine,
    PatientTwin,
    TwinState,
)
from qbitalabs.digital_twin.models import (
    PhysiologicalModel,
    MetabolismModel,
    ImmuneSystemModel,
    CardiovascularModel,
    GeneRegulatoryModel,
)

__all__ = [
    "DigitalTwinEngine",
    "PatientTwin",
    "TwinState",
    "PhysiologicalModel",
    "MetabolismModel",
    "ImmuneSystemModel",
    "CardiovascularModel",
    "GeneRegulatoryModel",
]
