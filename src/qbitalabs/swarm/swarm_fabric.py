"""
SWARM Fabric - High-Level Interface for QBitaLabs Swarm Operations

Provides a simplified interface for common swarm operations:
- Quick swarm setup
- Discovery workflows
- Result aggregation
- Monitoring and observability
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Type

import structlog

from qbitalabs.core.config import SwarmConfig
from qbitalabs.core.types import AgentID, AgentRole, ExecutionStatus
from qbitalabs.swarm.orchestrator import SwarmOrchestrator
from qbitalabs.swarm.base_agent import BaseAgent, AgentMessage
from qbitalabs.swarm.coordinator import Task, ConsensusResult, ConsensusType

logger = structlog.get_logger(__name__)


@dataclass
class DiscoveryTask:
    """A high-level discovery task."""

    task_id: str = ""
    name: str = ""
    description: str = ""
    target: str = ""
    objective: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    results: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


@dataclass
class SwarmMetrics:
    """Metrics for swarm performance."""

    total_agents: int = 0
    active_agents: int = 0
    messages_per_second: float = 0.0
    discoveries_made: int = 0
    consensus_success_rate: float = 0.0
    average_agent_energy: float = 0.0
    pheromone_trail_count: int = 0
    quantum_jobs_completed: int = 0
    uptime_seconds: float = 0.0


class SwarmFabric:
    """
    High-level interface for QBitaLabs swarm operations.

    Provides simplified methods for:
    - Creating and managing swarms
    - Running discovery tasks
    - Monitoring swarm health
    - Aggregating results

    Example:
        >>> fabric = SwarmFabric()
        >>> await fabric.initialize()
        >>> await fabric.quick_setup(agent_count=100)
        >>> result = await fabric.run_discovery(
        ...     target="SARS-CoV-2 Main Protease",
        ...     objective="Find novel inhibitors"
        ... )
    """

    def __init__(self, config: SwarmConfig | None = None):
        """
        Initialize the SwarmFabric.

        Args:
            config: Swarm configuration.
        """
        self.config = config or SwarmConfig()
        self.orchestrator = SwarmOrchestrator(self.config)
        self._start_time: datetime | None = None
        self._discovery_tasks: dict[str, DiscoveryTask] = {}
        self._logger = structlog.get_logger("swarm_fabric")

    async def initialize(self) -> None:
        """Initialize the fabric and underlying orchestrator."""
        await self.orchestrator.initialize()
        self._start_time = datetime.utcnow()
        self._logger.info("SwarmFabric initialized")

    async def shutdown(self) -> None:
        """Shutdown the fabric."""
        await self.orchestrator.shutdown()
        self._logger.info("SwarmFabric shutdown")

    async def quick_setup(
        self,
        agent_count: int = 100,
        agent_distribution: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """
        Quickly set up a balanced swarm.

        Args:
            agent_count: Total number of agents to create.
            agent_distribution: Custom distribution by role (values should sum to 1.0).

        Returns:
            Dictionary of role -> count of spawned agents.
        """
        # Import agent implementations
        from qbitalabs.swarm.agents import (
            MolecularAgent,
            PathwayAgent,
            HypothesisAgent,
            ValidationAgent,
            LiteratureAgent,
            CohortAgent,
        )

        # Default balanced distribution
        default_distribution = {
            "molecular": 0.30,
            "pathway": 0.20,
            "hypothesis": 0.15,
            "validation": 0.15,
            "literature": 0.10,
            "cohort": 0.10,
        }

        distribution = agent_distribution or default_distribution

        # Map roles to agent classes and role enums
        role_mapping = {
            "molecular": (MolecularAgent, AgentRole.MOLECULAR_MODELER, "execution"),
            "pathway": (PathwayAgent, AgentRole.PATHWAY_SIMULATOR, "execution"),
            "hypothesis": (HypothesisAgent, AgentRole.HYPOTHESIS_GENERATOR, "planning"),
            "validation": (ValidationAgent, AgentRole.VALIDATION_AGENT, "execution"),
            "literature": (LiteratureAgent, AgentRole.LITERATURE_REVIEWER, "planning"),
            "cohort": (CohortAgent, AgentRole.COHORT_MANAGER, "execution"),
        }

        spawned = {}
        for role_name, fraction in distribution.items():
            if role_name not in role_mapping:
                continue

            agent_class, role, layer = role_mapping[role_name]
            count = int(agent_count * fraction)

            if count > 0:
                await self.orchestrator.spawn_agent_pool(
                    agent_class,
                    count=count,
                    role=role,
                    layer=layer,
                )
                spawned[role_name] = count

        self._logger.info(
            "Quick setup complete",
            total_agents=sum(spawned.values()),
            distribution=spawned,
        )

        return spawned

    async def add_agent_pool(
        self,
        agent_class: Type[BaseAgent],
        count: int,
        role: AgentRole,
        layer: str = "execution",
    ) -> list[AgentID]:
        """
        Add a pool of agents to the swarm.

        Args:
            agent_class: Agent class to instantiate.
            count: Number of agents.
            role: Agent role.
            layer: Hierarchy layer.

        Returns:
            List of created agent IDs.
        """
        return await self.orchestrator.spawn_agent_pool(
            agent_class,
            count=count,
            role=role,
            layer=layer,
        )

    async def run_discovery(
        self,
        target: str,
        objective: str,
        constraints: dict[str, Any] | None = None,
        max_cycles: int = 1000,
        timeout_seconds: float = 300.0,
    ) -> DiscoveryTask:
        """
        Run a discovery task.

        Args:
            target: Discovery target (e.g., protein name).
            objective: What to discover.
            constraints: Additional constraints.
            max_cycles: Maximum swarm cycles.
            timeout_seconds: Overall timeout.

        Returns:
            DiscoveryTask with results.
        """
        from uuid import uuid4

        task = DiscoveryTask(
            task_id=str(uuid4()),
            name=f"Discovery: {objective[:50]}",
            target=target,
            objective=objective,
            constraints=constraints or {},
            status=ExecutionStatus.RUNNING,
        )

        self._discovery_tasks[task.task_id] = task

        self._logger.info(
            "Starting discovery",
            task_id=task.task_id[:8],
            target=target,
            objective=objective,
        )

        try:
            # Run the swarm
            await asyncio.wait_for(
                self.orchestrator.run(max_cycles=max_cycles),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._logger.warning("Discovery timeout", task_id=task.task_id[:8])
        except asyncio.CancelledError:
            self._logger.info("Discovery cancelled", task_id=task.task_id[:8])
        finally:
            await self.orchestrator.stop()

        # Collect results from context
        task.results = self._collect_discovery_results()
        task.status = ExecutionStatus.COMPLETED
        task.completed_at = datetime.utcnow()

        self._logger.info(
            "Discovery complete",
            task_id=task.task_id[:8],
            result_count=len(task.results),
        )

        return task

    def _collect_discovery_results(self) -> list[dict[str, Any]]:
        """Collect results from the discovery cache."""
        results = []

        # Collect from discovery cache
        for key, value in self.orchestrator.context.discovery_cache.items():
            results.append({"key": key, "value": value})

        # Collect from quantum results
        for key, value in self.orchestrator.context.quantum_results.items():
            results.append({"type": "quantum", "key": key, "value": value})

        return results

    async def request_consensus(
        self,
        question: str,
        options: list[Any],
        consensus_type: ConsensusType = ConsensusType.MAJORITY,
        roles: list[AgentRole] | None = None,
    ) -> ConsensusResult:
        """
        Request consensus from agents.

        Args:
            question: Question to decide.
            options: Available options.
            consensus_type: Consensus protocol.
            roles: Specific roles to involve (all if None).

        Returns:
            ConsensusResult.
        """
        from uuid import uuid4

        # Get participating agents
        if roles:
            agents = []
            for role in roles:
                agents.extend(self.orchestrator.get_agents_by_role(role))
        else:
            agents = list(self.orchestrator.agents.values())

        return await self.orchestrator.coordinator.initiate_consensus(
            consensus_id=str(uuid4()),
            question=question,
            options=options,
            agents=agents,
            consensus_type=consensus_type,
        )

    def get_metrics(self) -> SwarmMetrics:
        """Get current swarm metrics."""
        status = self.orchestrator.get_swarm_status()

        # Calculate average energy
        total_energy = sum(a.energy for a in self.orchestrator.agents.values())
        avg_energy = (
            total_energy / len(self.orchestrator.agents)
            if self.orchestrator.agents else 0
        )

        # Calculate uptime
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return SwarmMetrics(
            total_agents=status["total_agents"],
            active_agents=status.get("agents_by_state", {}).get("active", 0),
            messages_per_second=(
                status["messages_processed"] / uptime if uptime > 0 else 0
            ),
            discoveries_made=len(self.orchestrator.context.discovery_cache),
            consensus_success_rate=status["coordinator"]["consensus_success_rate"],
            average_agent_energy=avg_energy,
            pheromone_trail_count=status["active_pheromone_trails"],
            quantum_jobs_completed=status["quantum_jobs_completed"],
            uptime_seconds=uptime,
        )

    def get_status(self) -> dict[str, Any]:
        """Get full status report."""
        metrics = self.get_metrics()
        status = self.orchestrator.get_swarm_status()

        return {
            "metrics": {
                "total_agents": metrics.total_agents,
                "active_agents": metrics.active_agents,
                "messages_per_second": round(metrics.messages_per_second, 2),
                "discoveries_made": metrics.discoveries_made,
                "consensus_success_rate": round(metrics.consensus_success_rate, 3),
                "average_agent_energy": round(metrics.average_agent_energy, 3),
                "uptime_seconds": round(metrics.uptime_seconds, 1),
            },
            "agents": status["agents_by_role"],
            "hierarchy": {
                "strategic": status["strategic_agents"],
                "planning": status["planning_agents"],
                "execution": status["execution_agents"],
            },
            "running": status["running"],
            "cycle_count": status["cycle_count"],
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the swarm."""
        status = self.get_status()

        # Check for issues
        issues = []

        if status["metrics"]["total_agents"] == 0:
            issues.append("No agents registered")

        if status["metrics"]["average_agent_energy"] < 0.2:
            issues.append("Low average agent energy")

        if not status["running"]:
            issues.append("Swarm not running")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
