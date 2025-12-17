"""
Coordinator for QBitaLabs SWARM Architecture

Handles task coordination and consensus among agents:
- Task distribution and load balancing
- Consensus protocols (voting, weighted, Byzantine)
- Workflow orchestration
- Result aggregation
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import uuid4

import structlog

from qbitalabs.core.types import (
    AgentID,
    AgentRole,
    ExecutionStatus,
    MessagePriority,
    MessageType,
)
from qbitalabs.swarm.base_agent import AgentMessage, BaseAgent

logger = structlog.get_logger(__name__)


class ConsensusType(str, Enum):
    """Types of consensus protocols."""

    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Weighted by agent energy/reputation
    UNANIMOUS = "unanimous"  # All must agree
    QUORUM = "quorum"  # Minimum number must agree


@dataclass
class ConsensusResult:
    """Result of a consensus operation."""

    consensus_id: str
    achieved: bool
    decision: Any
    votes: dict[str, Any] = field(default_factory=dict)
    participation: float = 0.0  # Fraction of agents that voted
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Task:
    """A task to be coordinated."""

    task_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    priority: MessagePriority = MessagePriority.NORMAL
    assigned_agents: list[AgentID] = field(default_factory=list)
    results: dict[AgentID, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    timeout_seconds: float = 60.0


class Coordinator:
    """
    Coordinates tasks and consensus among SWARM agents.

    Features:
    - Task distribution with load balancing
    - Multiple consensus protocols
    - Result aggregation
    - Failure handling

    Example:
        >>> coordinator = Coordinator()
        >>> task = Task(name="analyze_molecule", payload={"smiles": "CCO"})
        >>> result = await coordinator.distribute_task(task, agents)
    """

    def __init__(
        self,
        consensus_threshold: float = 0.67,
        default_timeout: float = 60.0,
    ):
        """
        Initialize the coordinator.

        Args:
            consensus_threshold: Required agreement threshold (0-1).
            default_timeout: Default task timeout in seconds.
        """
        self.consensus_threshold = consensus_threshold
        self.default_timeout = default_timeout

        # Task tracking
        self._tasks: dict[str, Task] = {}
        self._task_queues: dict[AgentRole, asyncio.Queue[Task]] = defaultdict(asyncio.Queue)

        # Consensus tracking
        self._pending_consensus: dict[str, dict[AgentID, Any]] = {}
        self._consensus_waiters: dict[str, asyncio.Event] = {}

        # Agent tracking
        self._agent_loads: dict[AgentID, int] = defaultdict(int)

        # Metrics
        self.tasks_distributed = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.consensus_attempts = 0
        self.consensus_achieved = 0

        self._logger = structlog.get_logger("coordinator")

    async def distribute_task(
        self,
        task: Task,
        agents: list[BaseAgent],
        strategy: str = "load_balance",
    ) -> Task:
        """
        Distribute a task to agents.

        Args:
            task: Task to distribute.
            agents: Available agents.
            strategy: Distribution strategy ("load_balance", "broadcast", "random").

        Returns:
            Updated task with assignments.
        """
        if not agents:
            task.status = ExecutionStatus.FAILED
            return task

        self._tasks[task.task_id] = task
        task.status = ExecutionStatus.QUEUED

        # Select agents based on strategy
        if strategy == "broadcast":
            selected = agents
        elif strategy == "random":
            import random
            selected = [random.choice(agents)]
        else:  # load_balance
            selected = self._select_least_loaded(agents, count=min(3, len(agents)))

        task.assigned_agents = [a.agent_id for a in selected]

        # Send task to selected agents
        for agent in selected:
            message = AgentMessage(
                message_type=MessageType.COMMAND,
                payload={
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "data": task.payload,
                },
                priority=task.priority,
            )
            await agent.receive_message(message)
            self._agent_loads[agent.agent_id] += 1

        task.status = ExecutionStatus.RUNNING
        self.tasks_distributed += 1

        self._logger.info(
            "Task distributed",
            task_id=task.task_id[:8],
            assigned_count=len(selected),
        )

        return task

    def _select_least_loaded(
        self,
        agents: list[BaseAgent],
        count: int,
    ) -> list[BaseAgent]:
        """Select the least loaded agents."""
        # Sort by current load
        sorted_agents = sorted(
            agents,
            key=lambda a: (self._agent_loads.get(a.agent_id, 0), -a.energy),
        )
        return sorted_agents[:count]

    async def collect_result(
        self,
        task_id: str,
        agent_id: AgentID,
        result: Any,
    ) -> None:
        """
        Collect a result from an agent.

        Args:
            task_id: Task identifier.
            agent_id: Agent that completed the task.
            result: Task result.
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        task.results[agent_id] = result
        self._agent_loads[agent_id] = max(0, self._agent_loads.get(agent_id, 0) - 1)

        # Check if all assigned agents have reported
        if len(task.results) >= len(task.assigned_agents):
            task.status = ExecutionStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self.tasks_completed += 1

            self._logger.info(
                "Task completed",
                task_id=task_id[:8],
                result_count=len(task.results),
            )

    async def initiate_consensus(
        self,
        consensus_id: str,
        question: str,
        options: list[Any],
        agents: list[BaseAgent],
        consensus_type: ConsensusType = ConsensusType.MAJORITY,
        timeout: float | None = None,
    ) -> ConsensusResult:
        """
        Initiate a consensus vote among agents.

        Args:
            consensus_id: Unique identifier for this consensus.
            question: Question being decided.
            options: Available options to vote on.
            agents: Agents participating in consensus.
            consensus_type: Type of consensus protocol.
            timeout: Vote timeout in seconds.

        Returns:
            ConsensusResult with outcome.
        """
        if not agents:
            return ConsensusResult(
                consensus_id=consensus_id,
                achieved=False,
                decision=None,
            )

        timeout = timeout or self.default_timeout
        self._pending_consensus[consensus_id] = {}
        self._consensus_waiters[consensus_id] = asyncio.Event()

        # Send vote request to all agents
        vote_request = AgentMessage(
            message_type=MessageType.QUERY,
            payload={
                "type": "consensus_vote",
                "consensus_id": consensus_id,
                "question": question,
                "options": options,
            },
            priority=MessagePriority.HIGH,
        )

        for agent in agents:
            await agent.receive_message(vote_request)

        self.consensus_attempts += 1

        # Wait for votes with timeout
        try:
            await asyncio.wait_for(
                self._wait_for_consensus(consensus_id, len(agents)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "Consensus timeout",
                consensus_id=consensus_id[:8],
                votes_received=len(self._pending_consensus.get(consensus_id, {})),
            )

        # Tally votes
        result = self._tally_votes(
            consensus_id=consensus_id,
            total_agents=len(agents),
            consensus_type=consensus_type,
            agents=agents,
        )

        # Cleanup
        self._pending_consensus.pop(consensus_id, None)
        self._consensus_waiters.pop(consensus_id, None)

        if result.achieved:
            self.consensus_achieved += 1

        return result

    async def _wait_for_consensus(
        self,
        consensus_id: str,
        expected_votes: int,
    ) -> None:
        """Wait for consensus votes to arrive."""
        event = self._consensus_waiters.get(consensus_id)
        if not event:
            return

        while len(self._pending_consensus.get(consensus_id, {})) < expected_votes:
            try:
                await asyncio.wait_for(event.wait(), timeout=0.1)
                event.clear()
            except asyncio.TimeoutError:
                continue

    async def record_vote(
        self,
        consensus_id: str,
        agent_id: AgentID,
        vote: Any,
    ) -> None:
        """
        Record a vote from an agent.

        Args:
            consensus_id: Consensus identifier.
            agent_id: Voting agent.
            vote: Agent's vote.
        """
        if consensus_id not in self._pending_consensus:
            return

        self._pending_consensus[consensus_id][agent_id] = vote

        # Signal that a vote was received
        event = self._consensus_waiters.get(consensus_id)
        if event:
            event.set()

    def _tally_votes(
        self,
        consensus_id: str,
        total_agents: int,
        consensus_type: ConsensusType,
        agents: list[BaseAgent],
    ) -> ConsensusResult:
        """Tally votes and determine consensus."""
        votes = self._pending_consensus.get(consensus_id, {})
        participation = len(votes) / total_agents if total_agents > 0 else 0

        if not votes:
            return ConsensusResult(
                consensus_id=consensus_id,
                achieved=False,
                decision=None,
                participation=participation,
            )

        # Count votes
        vote_counts: dict[Any, float] = defaultdict(float)

        for agent_id, vote in votes.items():
            if consensus_type == ConsensusType.WEIGHTED:
                # Weight by agent energy
                agent = next((a for a in agents if a.agent_id == agent_id), None)
                weight = agent.energy if agent else 1.0
                vote_counts[vote] += weight
            else:
                vote_counts[vote] += 1

        # Determine winner
        if not vote_counts:
            return ConsensusResult(
                consensus_id=consensus_id,
                achieved=False,
                decision=None,
                votes=dict(votes),
                participation=participation,
            )

        winner = max(vote_counts, key=vote_counts.get)
        winner_count = vote_counts[winner]

        # Check if consensus achieved based on type
        achieved = False

        if consensus_type == ConsensusType.UNANIMOUS:
            achieved = len(set(votes.values())) == 1
        elif consensus_type == ConsensusType.MAJORITY:
            achieved = winner_count > len(votes) / 2
        elif consensus_type == ConsensusType.WEIGHTED:
            total_weight = sum(vote_counts.values())
            achieved = winner_count / total_weight >= self.consensus_threshold
        elif consensus_type == ConsensusType.QUORUM:
            achieved = (
                participation >= self.consensus_threshold
                and winner_count > len(votes) / 2
            )

        return ConsensusResult(
            consensus_id=consensus_id,
            achieved=achieved,
            decision=winner if achieved else None,
            votes=dict(votes),
            participation=participation,
        )

    async def aggregate_results(
        self,
        task_id: str,
        aggregation: str = "merge",
    ) -> dict[str, Any]:
        """
        Aggregate results from multiple agents.

        Args:
            task_id: Task identifier.
            aggregation: Aggregation strategy ("merge", "vote", "average").

        Returns:
            Aggregated result.
        """
        task = self._tasks.get(task_id)
        if not task or not task.results:
            return {}

        results = list(task.results.values())

        if aggregation == "vote":
            # Return most common result
            from collections import Counter
            result_strs = [str(r) for r in results]
            most_common = Counter(result_strs).most_common(1)
            if most_common:
                return {"decision": results[result_strs.index(most_common[0][0])]}

        elif aggregation == "average":
            # Average numeric results
            if all(isinstance(r, (int, float)) for r in results):
                return {"average": sum(results) / len(results)}

        # Default: merge dictionaries
        merged = {}
        for result in results:
            if isinstance(result, dict):
                merged.update(result)
            else:
                merged[f"result_{len(merged)}"] = result

        return merged

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics."""
        return {
            "tasks_distributed": self.tasks_distributed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "active_tasks": sum(
                1 for t in self._tasks.values()
                if t.status == ExecutionStatus.RUNNING
            ),
            "consensus_attempts": self.consensus_attempts,
            "consensus_achieved": self.consensus_achieved,
            "consensus_success_rate": (
                self.consensus_achieved / self.consensus_attempts
                if self.consensus_attempts > 0 else 0
            ),
        }
