"""
QBitaLabs Quantum Computing Module

Provides quantum computing capabilities for molecular simulation,
optimization, and machine learning.

Supported backends:
- Qiskit (IBM Quantum)
- Cirq (Google Quantum)
- PennyLane (Xanadu)
- IonQ
"""

from __future__ import annotations

from qbitalabs.quantum.backends import (
    BaseQuantumBackend,
    QiskitBackend,
    CirqBackend,
    PennyLaneBackend,
    IonQBackend,
    SimulatorBackend,
)

__all__ = [
    "BaseQuantumBackend",
    "QiskitBackend",
    "CirqBackend",
    "PennyLaneBackend",
    "IonQBackend",
    "SimulatorBackend",
]
