"""
SWARM Patterns for QBitaLabs

Bio-inspired coordination patterns:
- Hierarchical: Multi-level organization
- Stigmergy: Environment-mediated coordination
- ProteinSwarm: Protein-like binding and complexes
- AntColony: Pheromone-based optimization
- ParticleSwarm: Velocity-based optimization
"""

from __future__ import annotations

from qbitalabs.swarm.patterns.hierarchical import HierarchicalPattern
from qbitalabs.swarm.patterns.stigmergy import StigmergyPattern, PheromoneTrail
from qbitalabs.swarm.patterns.protein_swarm import ProteinSwarmPattern, ProteinAgent, ProteinComplex
from qbitalabs.swarm.patterns.ant_colony import AntColonyPattern
from qbitalabs.swarm.patterns.particle_swarm import ParticleSwarmPattern

__all__ = [
    "HierarchicalPattern",
    "StigmergyPattern",
    "PheromoneTrail",
    "ProteinSwarmPattern",
    "ProteinAgent",
    "ProteinComplex",
    "AntColonyPattern",
    "ParticleSwarmPattern",
]
