"""
SWARM Orchestrator for QBitaLabs

Manages 100s of agents coordinating like proteins in a cell:
- Hierarchical organization (strategic -> planning -> execution)
- Event-driven asynchronous messaging
- Stigmergy-based coordination
- Automatic load balancing and scaling
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Type

import structlog

from qbitalabs.core.config import SwarmConfig
from qbitalabs.core.types import (
    AgentID,
    AgentRole,
    AgentState,
    MessagePriority,
    MessageType,
)
from qbitalabs.core.exceptions import SwarmCapacityError, SwarmError
from qbitalabs.swarm.base_agent import AgentContext, AgentMessage, BaseAgent
from qbitalabs.swarm.message_bus import MessageBus
from qbitalabs.swarm.coordinator import Coordinator, ConsensusResult, Task

logger = structlog.get_logger(__name__)


class SwarmOrchestrator:
    """
    Orchestrates 100s of SWARM agents for quantum-bio discovery.

    Architecture inspired by:
    - Ant Colony Optimization (pheromone trails)
    - Protein signaling cascades
    - Cellular self-organization

    Supports heterogeneous compute:
    - Classical GPU agents
    - Quantum circuit executor agents
    - Neuromorphic processor agents

    Example:
        >>> config = SwarmConfig(max_agents=200)
        >>> orchestrator = SwarmOrchestrator(config)
        >>> await orchestrator.spawn_agent_pool(MolecularAgent, count=30)
        >>> await orchestrator.run(max_cycles=1000)
    """

    def __init__(self, config: SwarmConfig | None = None):
        """
        Initialize the orchestrator.

        Args:
            config: Swarm configuration. Uses defaults if not provided.
        """
        self.config = config or SwarmConfig()

        # Agent management
        self.agents: dict[AgentID, BaseAgent] = {}
        self.agent_pools: dict[AgentRole, list[AgentID]] = defaultdict(list)

        # Hierarchical layers
        self.strategic_agents: list[AgentID] = []
        self.planning_agents: list[AgentID] = []
        self.execution_agents: list[AgentID] = []

        # Shared context (stigmergy environment)
        self.context = AgentContext()

        # Communication
        self.message_bus = MessageBus()
        self.coordinator = Coordinator(
            consensus_threshold=self.config.consensus_threshold,
        )

        # State
        self.running = False
        self._tasks: list[asyncio.Task] = []

        # Metrics
        self.cycle_count = 0
        self.messages_processed = 0
        self.discoveries_made = 0
        self.quantum_jobs_completed = 0

        self._logger = structlog.get_logger("orchestrator")

    async def initialize(self) -> None:
        """Initialize the orchestrator and message bus."""
        await self.message_bus.start()
        self._logger.info("Orchestrator initialized")

    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.running = False

        # Stop all agents
        for agent in self.agents.values():
            agent.state = AgentState.TERMINATED

        # Stop message bus
        await self.message_bus.stop()

        # Cancel running tasks
        for task in self._tasks:
            task.cancel()

        self._logger.info("Orchestrator shutdown complete")

    async def register_agent(
        self,
        agent: BaseAgent,
        layer: str = "execution",
    ) -> AgentID:
        """
        Register an agent with the swarm.

        Args:
            agent: Agent instance to register.
            layer: Hierarchy layer ("strategic", "planning", "execution").

        Returns:
            Agent ID.

        Raises:
            SwarmCapacityError: If max agents reached.
        """
        if len(self.agents) >= self.config.max_agents:
            raise SwarmCapacityError(
                f"Maximum agent limit ({self.config.max_agents}) reached",
                current_agents=len(self.agents),
                max_agents=self.config.max_agents,
            )

        # Setup agent
        agent.context = self.context
        agent._broadcast_callback = self._handle_agent_broadcast

        # Register with swarm
        self.agents[agent.agent_id] = agent
        self.agent_pools[agent.role].append(agent.agent_id)

        # Register with message bus
        self.message_bus.register_agent(
            agent.agent_id,
            agent.role,
            agent.receive_message,
        )

        # Assign to hierarchy layer
        if layer == "strategic":
            self.strategic_agents.append(agent.agent_id)
        elif layer == "planning":
            self.planning_agents.append(agent.agent_id)
        else:
            self.execution_agents.append(agent.agent_id)

        self._logger.info(
            "Agent registered",
            agent_id=str(agent.agent_id)[:8],
            role=agent.role.value,
            layer=layer,
        )

        return agent.agent_id

    async def unregister_agent(self, agent_id: AgentID) -> None:
        """
        Unregister an agent from the swarm.

        Args:
            agent_id: Agent ID to unregister.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return

        # Remove from pools
        if agent_id in self.agent_pools.get(agent.role, []):
            self.agent_pools[agent.role].remove(agent_id)

        # Remove from hierarchy
        for layer_list in [
            self.strategic_agents,
            self.planning_agents,
            self.execution_agents,
        ]:
            if agent_id in layer_list:
                layer_list.remove(agent_id)

        # Unregister from message bus
        self.message_bus.unregister_agent(agent_id, agent.role)

        # Remove agent
        del self.agents[agent_id]

        self._logger.info("Agent unregistered", agent_id=str(agent_id)[:8])

    async def spawn_agent_pool(
        self,
        agent_class: Type[BaseAgent],
        count: int,
        role: AgentRole | None = None,
        layer: str = "execution",
        **kwargs: Any,
    ) -> list[AgentID]:
        """
        Spawn a pool of identical agents.

        Args:
            agent_class: Class of agent to spawn.
            count: Number of agents to spawn.
            role: Role for agents (uses class default if not specified).
            layer: Hierarchy layer for agents.
            **kwargs: Additional arguments for agent constructor.

        Returns:
            List of spawned agent IDs.
        """
        agent_ids = []

        for i in range(count):
            agent_kwargs = kwargs.copy()
            if role is not None:
                agent_kwargs["role"] = role

            agent = agent_class(**agent_kwargs)
            agent_id = await self.register_agent(agent, layer=layer)
            agent_ids.append(agent_id)

        self._logger.info(
            "Agent pool spawned",
            agent_class=agent_class.__name__,
            count=count,
            layer=layer,
        )

        return agent_ids

    async def broadcast(self, message: AgentMessage) -> None:
        """
        Broadcast message to all agents or specific recipient.

        Args:
            message: Message to broadcast.
        """
        await self.message_bus.publish(message)
        self.messages_processed += 1

    async def broadcast_to_role(
        self,
        message: AgentMessage,
        role: AgentRole,
    ) -> None:
        """
        Broadcast to all agents with a specific role.

        Args:
            message: Message to broadcast.
            role: Target role.
        """
        await self.message_bus.publish_to_role(message, role)

    async def request_quantum_computation(
        self,
        circuit_spec: dict[str, Any],
        backend: str = "qiskit",
        priority: int = 8,
    ) -> str:
        """
        Request quantum computation from quantum executor agents.

        Args:
            circuit_spec: Circuit specification.
            backend: Quantum backend to use.
            priority: Task priority.

        Returns:
            Request message ID.
        """
        message = AgentMessage(
            message_type=MessageType.QUANTUM_REQUEST,
            payload={"circuit": circuit_spec, "backend": backend},
            priority=MessagePriority(min(priority, 10)),
        )
        await self.broadcast_to_role(message, AgentRole.QUANTUM_EXECUTOR)
        return str(message.id)

    async def request_neuromorphic_processing(
        self,
        signal_data: dict[str, Any],
        processor_type: str = "biosignal",
        priority: int = 7,
    ) -> str:
        """
        Request neuromorphic processing from neuromorphic agents.

        Args:
            signal_data: Signal data to process.
            processor_type: Type of processing.
            priority: Task priority.

        Returns:
            Request message ID.
        """
        message = AgentMessage(
            message_type=MessageType.NEUROMORPHIC_REQUEST,
            payload={"signal": signal_data, "processor": processor_type},
            priority=MessagePriority(min(priority, 10)),
        )
        await self.broadcast_to_role(message, AgentRole.NEUROMORPHIC_PROCESSOR)
        return str(message.id)

    async def _handle_agent_broadcast(self, message: AgentMessage) -> None:
        """Handle broadcast from an agent."""
        await self.broadcast(message)

    async def _decay_pheromones(self) -> None:
        """Decay all pheromone trails (biological evaporation)."""
        self.context.decay_pheromones(self.config.pheromone_decay_rate)

    async def _regenerate_agent_energy(self) -> None:
        """Regenerate energy for all agents."""
        for agent in self.agents.values():
            await agent.regenerate_energy(self.config.energy_regeneration_rate)

    async def _process_agent_cycle(self, agent: BaseAgent) -> dict[str, Any]:
        """Process one cycle for an agent."""
        if agent.state == AgentState.TERMINATED:
            return {"status": "terminated"}

        try:
            return await asyncio.wait_for(
                agent.run_cycle(),
                timeout=self.config.agent_timeout_seconds,
            )
        except asyncio.TimeoutError:
            agent.state = AgentState.ERROR
            return {"status": "timeout", "agent_id": str(agent.agent_id)[:8]}
        except Exception as e:
            agent.state = AgentState.ERROR
            self._logger.exception(
                "Agent cycle error",
                agent_id=str(agent.agent_id)[:8],
                error=str(e),
            )
            return {"status": "error", "error": str(e)}

    async def run_cycle(self) -> dict[str, Any]:
        """
        Run one cycle of the swarm.

        Returns:
            Cycle metrics.
        """
        self.cycle_count += 1

        # Process all agents concurrently
        tasks = [
            self._process_agent_cycle(agent)
            for agent in self.agents.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Environmental updates
        await self._decay_pheromones()
        await self._regenerate_agent_energy()

        # Calculate metrics
        active_count = sum(
            1 for a in self.agents.values()
            if a.state == AgentState.ACTIVE
        )

        return {
            "cycle": self.cycle_count,
            "active_agents": active_count,
            "total_agents": len(self.agents),
            "messages_processed": self.messages_processed,
            "pheromone_trails": len(self.context.pheromone_trails),
        }

    async def run(self, max_cycles: int | None = None) -> None:
        """
        Run the swarm continuously.

        Args:
            max_cycles: Maximum cycles to run. None for infinite.
        """
        self.running = True
        cycle = 0

        self._logger.info(
            "Swarm starting",
            total_agents=len(self.agents),
            max_cycles=max_cycles,
        )

        while self.running:
            if max_cycles and cycle >= max_cycles:
                break

            metrics = await self.run_cycle()
            cycle += 1

            if cycle % 100 == 0:
                self._logger.info(f"Swarm cycle {cycle}", **metrics)

            await asyncio.sleep(0.01)  # Yield to event loop

        self._logger.info("Swarm stopped", total_cycles=cycle)

    async def stop(self) -> None:
        """Stop the swarm gracefully."""
        self.running = False
        for agent in self.agents.values():
            agent.state = AgentState.TERMINATED

    def get_agent(self, agent_id: AgentID) -> BaseAgent | None:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_agents_by_role(self, role: AgentRole) -> list[BaseAgent]:
        """Get all agents with a specific role."""
        agent_ids = self.agent_pools.get(role, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def get_swarm_status(self) -> dict[str, Any]:
        """Get current swarm status."""
        state_counts = defaultdict(int)
        for agent in self.agents.values():
            state_counts[agent.state.value] += 1

        return {
            "total_agents": len(self.agents),
            "agents_by_role": {
                role.value: len(ids) for role, ids in self.agent_pools.items()
            },
            "agents_by_state": dict(state_counts),
            "strategic_agents": len(self.strategic_agents),
            "planning_agents": len(self.planning_agents),
            "execution_agents": len(self.execution_agents),
            "cycle_count": self.cycle_count,
            "messages_processed": self.messages_processed,
            "active_pheromone_trails": len(self.context.pheromone_trails),
            "quantum_jobs_completed": self.quantum_jobs_completed,
            "running": self.running,
            "message_bus": self.message_bus.get_metrics(),
            "coordinator": self.coordinator.get_metrics(),
        }
