"""
QBitaLabs Neuromorphic Computing Module

Provides neuromorphic computing capabilities using:
- BrainChip Akida
- Intel Loihi
- SynSense
- Software simulators

Features:
- Spiking Neural Networks (SNNs)
- On-chip learning
- Event-driven processing
- Ultra-low power inference
"""

from __future__ import annotations

from qbitalabs.neuromorphic.backends import (
    BaseNeuromorphicBackend,
    AkidaBackend,
    LoihiBackend,
    SynSenseBackend,
    SimulatorBackend,
)
from qbitalabs.neuromorphic.snn import (
    SpikingNeuralNetwork,
    LIFNeuron,
    SynapticConnection,
)

__all__ = [
    "BaseNeuromorphicBackend",
    "AkidaBackend",
    "LoihiBackend",
    "SynSenseBackend",
    "SimulatorBackend",
    "SpikingNeuralNetwork",
    "LIFNeuron",
    "SynapticConnection",
]
