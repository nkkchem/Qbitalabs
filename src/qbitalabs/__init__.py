"""
QBitaLabs: Quantum-Bio Swarm Intelligence Platform

Swarm intelligence for quantum biology and human health.

This platform provides:
- SWARM Agent Architecture: 100s of coordinating AI "protein agents"
- Quantum Computing Layer: Qiskit, Cirq, PennyLane, IonQ integration
- Neuromorphic Computing: Intel Loihi, BrainChip Akida, SynSense
- Biological Digital Twins: Quantum-accurate simulation for disease prediction

Architecture:
    Swarm = Bio-inspired SWARM agent fabric (100s of agents)
    Q = Quantum layer using Qiskit + PennyLane + Cirq
    Bio = Deep biology + health stack with neuromorphic edge processing

Example:
    >>> from qbitalabs.swarm import SwarmOrchestrator, SwarmConfig
    >>> orchestrator = SwarmOrchestrator(SwarmConfig(max_agents=100))
    >>> await orchestrator.run(max_cycles=1000)

Copyright (c) 2024-2025 QBitaLabs, Inc. All rights reserved.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Neeraj Kumar"
__email__ = "neeraj@qbitalabs.com"
__company__ = "QBitaLabs, Inc."

# Lazy imports to avoid circular dependencies and improve startup time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbitalabs.swarm import SwarmFabric, SwarmOrchestrator
    from qbitalabs.quantum import QuantumBackend
    from qbitalabs.neuromorphic import NeuromorphicProcessor, SpikingNetwork
    from qbitalabs.digital_twin import PatientTwin, CohortTwin
    from qbitalabs.workflows import DrugDiscoveryPipeline, DigitalTwinPipeline, WorkflowEngine


def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "SwarmFabric":
        from qbitalabs.swarm import SwarmFabric
        return SwarmFabric
    elif name == "SwarmOrchestrator":
        from qbitalabs.swarm import SwarmOrchestrator
        return SwarmOrchestrator
    elif name == "QuantumBackend":
        from qbitalabs.quantum import QuantumBackend
        return QuantumBackend
    elif name == "NeuromorphicProcessor":
        from qbitalabs.neuromorphic import NeuromorphicProcessor
        return NeuromorphicProcessor
    elif name == "SpikingNetwork":
        from qbitalabs.neuromorphic import SpikingNetwork
        return SpikingNetwork
    elif name == "PatientTwin":
        from qbitalabs.digital_twin import PatientTwin
        return PatientTwin
    elif name == "CohortTwin":
        from qbitalabs.digital_twin import CohortTwin
        return CohortTwin
    elif name == "DrugDiscoveryPipeline":
        from qbitalabs.workflows import DrugDiscoveryPipeline
        return DrugDiscoveryPipeline
    elif name == "DigitalTwinPipeline":
        from qbitalabs.workflows import DigitalTwinPipeline
        return DigitalTwinPipeline
    elif name == "WorkflowEngine":
        from qbitalabs.workflows import WorkflowEngine
        return WorkflowEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__company__",
    "SwarmFabric",
    "SwarmOrchestrator",
    "QuantumBackend",
    "NeuromorphicProcessor",
    "SpikingNetwork",
    "PatientTwin",
    "CohortTwin",
    "DrugDiscoveryPipeline",
    "DigitalTwinPipeline",
    "WorkflowEngine",
]
