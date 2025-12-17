"""
QBitaLabs SWARM Agent System

Bio-inspired multi-agent architecture where 100s of agents coordinate like protein swarms.

Components:
- BaseAgent: Foundation for all SWARM agents
- SwarmOrchestrator: Manages 100s of coordinating agents
- SwarmFabric: High-level interface for swarm operations
- MessageBus: Asynchronous message passing
- Coordinator: Task coordination and consensus

Example:
    >>> from qbitalabs.swarm import SwarmOrchestrator, SwarmConfig
    >>> orchestrator = SwarmOrchestrator(SwarmConfig(max_agents=100))
    >>> await orchestrator.spawn_agent_pool(MolecularAgent, count=30)
    >>> await orchestrator.run(max_cycles=1000)
"""

from __future__ import annotations

from qbitalabs.swarm.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentContext,
)
from qbitalabs.swarm.orchestrator import (
    SwarmOrchestrator,
)
from qbitalabs.swarm.swarm_fabric import (
    SwarmFabric,
)
from qbitalabs.swarm.message_bus import (
    MessageBus,
)
from qbitalabs.swarm.coordinator import (
    Coordinator,
    ConsensusResult,
)
from qbitalabs.core.config import SwarmConfig
from qbitalabs.core.types import (
    AgentState,
    AgentRole,
    MessageType,
    MessagePriority,
)

__all__ = [
    # Core classes
    "BaseAgent",
    "AgentMessage",
    "AgentContext",
    "SwarmOrchestrator",
    "SwarmFabric",
    "MessageBus",
    "Coordinator",
    "ConsensusResult",
    # Config
    "SwarmConfig",
    # Types
    "AgentState",
    "AgentRole",
    "MessageType",
    "MessagePriority",
]
