"""
Base Agent for QBitaLabs SWARM Architecture

Inspired by protein behavior - agents coordinate through:
- Stigmergy (indirect communication via shared environment)
- Pheromone-like signal propagation
- Local interactions producing global behavior
- Self-organization without central control

Example:
    >>> class MyAgent(BaseAgent):
    ...     async def process(self, input_data):
    ...         # Agent-specific processing
    ...         return {"result": "processed"}
    ...
    ...     async def respond_to_signal(self, message):
    ...         if message.message_type == "query":
    ...             return AgentMessage(payload={"response": "data"})
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import structlog

from qbitalabs.core.types import (
    AgentID,
    AgentRole,
    AgentState,
    MessageID,
    MessagePriority,
    MessageType,
    generate_agent_id,
    generate_message_id,
)
from qbitalabs.core.exceptions import AgentError, AgentTimeoutError

logger = structlog.get_logger(__name__)


@dataclass
class AgentMessage:
    """
    Message passed between agents in the swarm.

    Messages support:
    - Direct messaging (with recipient_id)
    - Broadcasting (recipient_id=None)
    - Priority-based processing
    - Pheromone-like decay (strength decreases over time)

    Attributes:
        id: Unique message identifier.
        sender_id: ID of the sending agent.
        recipient_id: ID of the recipient agent (None for broadcast).
        message_type: Type of message (signal, command, query, etc.).
        payload: Message data.
        priority: Priority level (1-10, higher = more urgent).
        timestamp: When the message was created.
        pheromone_strength: Signal strength (decays over time).
        ttl: Time to live in processing cycles.
    """

    id: MessageID = field(default_factory=generate_message_id)
    sender_id: AgentID | None = None
    recipient_id: AgentID | None = None
    message_type: MessageType = MessageType.SIGNAL
    payload: dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pheromone_strength: float = 1.0
    ttl: int = 100
    correlation_id: MessageID | None = None  # For request-response tracking

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        return self.ttl <= 0

    def decay(self, rate: float = 0.05) -> None:
        """Apply decay to pheromone strength."""
        self.pheromone_strength *= 1 - rate
        self.ttl -= 1

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": str(self.id),
            "sender_id": str(self.sender_id) if self.sender_id else None,
            "recipient_id": str(self.recipient_id) if self.recipient_id else None,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "pheromone_strength": self.pheromone_strength,
            "ttl": self.ttl,
        }


@dataclass
class AgentContext:
    """
    Shared context accessible to all agents (stigmergy environment).

    This represents the shared "environment" that agents can read from
    and write to, enabling indirect communication (stigmergy).

    Attributes:
        global_state: Shared state dictionary.
        pheromone_trails: Active pheromone trails with strength values.
        discovery_cache: Cache for discovered information.
        quantum_results: Results from quantum computations.
        neuromorphic_signals: Signals from neuromorphic processing.
    """

    global_state: dict[str, Any] = field(default_factory=dict)
    pheromone_trails: dict[str, float] = field(default_factory=dict)
    discovery_cache: dict[str, Any] = field(default_factory=dict)
    quantum_results: dict[str, Any] = field(default_factory=dict)
    neuromorphic_signals: dict[str, Any] = field(default_factory=dict)
    blackboard: dict[str, Any] = field(default_factory=dict)  # Shared blackboard

    def get_trail_strength(self, trail_id: str) -> float:
        """Get the strength of a pheromone trail."""
        return self.pheromone_trails.get(trail_id, 0.0)

    def deposit_pheromone(self, trail_id: str, strength: float) -> None:
        """Deposit pheromone on a trail."""
        current = self.pheromone_trails.get(trail_id, 0)
        self.pheromone_trails[trail_id] = min(current + strength, 10.0)

    def decay_pheromones(self, rate: float = 0.05) -> None:
        """Apply decay to all pheromone trails."""
        expired = []
        for trail_id in self.pheromone_trails:
            self.pheromone_trails[trail_id] *= 1 - rate
            if self.pheromone_trails[trail_id] < 0.01:
                expired.append(trail_id)
        for trail_id in expired:
            del self.pheromone_trails[trail_id]


class BaseAgent(ABC):
    """
    Base class for all SWARM agents in QBitaLabs.

    Each agent operates like a protein in a cellular system:
    - Has a specific role/function
    - Responds to environmental signals
    - Can modify the shared environment
    - Coordinates through indirect communication

    Subclasses must implement:
    - process(): Main processing logic
    - respond_to_signal(): Handle incoming messages

    Example:
        >>> class MolecularAgent(BaseAgent):
        ...     async def process(self, input_data):
        ...         molecule = input_data.get("molecule")
        ...         energy = await self._calculate_energy(molecule)
        ...         await self.deposit_pheromone(f"energy:{molecule}", energy)
        ...         return {"energy": energy}
        ...
        ...     async def respond_to_signal(self, message):
        ...         if message.message_type == MessageType.QUERY:
        ...             return AgentMessage(
        ...                 recipient_id=message.sender_id,
        ...                 payload={"status": "ready"}
        ...             )
    """

    def __init__(
        self,
        agent_id: AgentID | None = None,
        role: AgentRole = AgentRole.MOLECULAR_MODELER,
        llm_model: str = "claude-sonnet-4-20250514",
        tools: list[Callable] | None = None,
        max_iterations: int = 10,
        name: str | None = None,
    ):
        """
        Initialize the agent.

        Args:
            agent_id: Unique agent identifier. Auto-generated if not provided.
            role: The agent's role in the swarm.
            llm_model: LLM model to use for reasoning.
            tools: List of tools available to the agent.
            max_iterations: Maximum iterations per processing cycle.
            name: Optional human-readable name.
        """
        self.agent_id: AgentID = agent_id or generate_agent_id()
        self.role = role
        self.llm_model = llm_model
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.name = name or f"{role.value}_{str(self.agent_id)[:8]}"

        # State
        self.state = AgentState.IDLE
        self._context: AgentContext | None = None
        self.iteration_count = 0

        # Energy model (like ATP in cells)
        self.energy = 1.0
        self.energy_consumption_rate = 0.01
        self.energy_regeneration_rate = 0.02

        # Message handling
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._pending_responses: dict[MessageID, asyncio.Future] = {}

        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.tasks_completed = 0

        # Logger
        self._logger = structlog.get_logger(
            agent_id=str(self.agent_id)[:8],
            role=self.role.value,
        )

        # Callbacks for orchestrator integration
        self._broadcast_callback: Callable[[AgentMessage], Any] | None = None

    @property
    def context(self) -> AgentContext:
        """Get the shared context."""
        if self._context is None:
            raise RuntimeError("Agent not connected to swarm context")
        return self._context

    @context.setter
    def context(self, ctx: AgentContext) -> None:
        """Set the shared context."""
        self._context = ctx

    @abstractmethod
    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Main processing logic for the agent.

        This method should implement the agent's core functionality.

        Args:
            input_data: Input data to process.

        Returns:
            Processing results.
        """
        pass

    @abstractmethod
    async def respond_to_signal(
        self, message: AgentMessage
    ) -> AgentMessage | None:
        """
        React to signals from other agents.

        Args:
            message: Incoming message to handle.

        Returns:
            Optional response message.
        """
        pass

    async def emit_signal(
        self,
        message_type: MessageType = MessageType.SIGNAL,
        payload: dict[str, Any] | None = None,
        recipient_id: AgentID | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> AgentMessage:
        """
        Emit a signal to the swarm.

        Args:
            message_type: Type of message to emit.
            payload: Message payload.
            recipient_id: Specific recipient (None for broadcast).
            priority: Message priority.

        Returns:
            The emitted message.
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload or {},
            priority=priority,
            pheromone_strength=self.energy * priority.value / 10,
        )

        if self._broadcast_callback:
            await self._broadcast_callback(message)

        self.messages_sent += 1
        self._logger.debug(
            "Signal emitted",
            message_type=message_type.value,
            recipient=str(recipient_id)[:8] if recipient_id else "broadcast",
        )

        return message

    async def request(
        self,
        recipient_id: AgentID,
        payload: dict[str, Any],
        timeout: float = 30.0,
    ) -> AgentMessage:
        """
        Send a request and wait for response.

        Args:
            recipient_id: Target agent ID.
            payload: Request payload.
            timeout: Response timeout in seconds.

        Returns:
            Response message.

        Raises:
            AgentTimeoutError: If no response within timeout.
        """
        correlation_id = generate_message_id()

        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=MessageType.QUERY,
            payload=payload,
            priority=MessagePriority.HIGH,
            correlation_id=correlation_id,
        )

        # Create future for response
        response_future: asyncio.Future[AgentMessage] = asyncio.Future()
        self._pending_responses[correlation_id] = response_future

        try:
            if self._broadcast_callback:
                await self._broadcast_callback(message)

            return await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            raise AgentTimeoutError(
                f"Request to {recipient_id} timed out",
                agent_id=str(self.agent_id),
                timeout_seconds=timeout,
            )
        finally:
            self._pending_responses.pop(correlation_id, None)

    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from the swarm.

        Args:
            message: Incoming message.
        """
        self.messages_received += 1

        # Check if this is a response to a pending request
        if message.correlation_id and message.correlation_id in self._pending_responses:
            self._pending_responses[message.correlation_id].set_result(message)
            return

        # Queue the message for processing
        await self.message_queue.put(message)

    async def deposit_pheromone(self, trail_id: str, strength: float) -> None:
        """
        Leave a pheromone trail for other agents to follow.

        Args:
            trail_id: Identifier for the trail.
            strength: Strength of the pheromone deposit.
        """
        if self._context:
            self._context.deposit_pheromone(trail_id, strength * self.energy)
            self._logger.debug("Pheromone deposited", trail_id=trail_id)

    async def sense_pheromone(self, trail_id: str) -> float:
        """
        Sense pheromone concentration at a trail.

        Args:
            trail_id: Trail to sense.

        Returns:
            Pheromone concentration.
        """
        if self._context:
            return self._context.get_trail_strength(trail_id)
        return 0.0

    async def consume_energy(self, amount: float) -> bool:
        """
        Consume metabolic energy for work.

        Args:
            amount: Amount of energy to consume.

        Returns:
            True if energy was available and consumed.
        """
        if self.energy >= amount:
            self.energy -= amount
            return True
        return False

    async def regenerate_energy(self, amount: float | None = None) -> None:
        """
        Regenerate energy over time (like ATP regeneration).

        Args:
            amount: Amount to regenerate. Uses default rate if not specified.
        """
        regen = amount if amount is not None else self.energy_regeneration_rate
        self.energy = min(self.energy + regen, 1.0)

    async def run_cycle(self) -> dict[str, Any]:
        """
        Run one processing cycle for the agent.

        Returns:
            Cycle metrics.
        """
        if self.state == AgentState.TERMINATED:
            return {"status": "terminated"}

        self.state = AgentState.ACTIVE
        messages_processed = 0

        try:
            # Process messages in queue
            while not self.message_queue.empty():
                try:
                    message = self.message_queue.get_nowait()
                    if not message.is_expired():
                        self.state = AgentState.PROCESSING
                        response = await self.respond_to_signal(message)
                        if response:
                            await self.emit_signal(
                                message_type=response.message_type,
                                payload=response.payload,
                                recipient_id=response.recipient_id,
                            )
                        messages_processed += 1
                except asyncio.QueueEmpty:
                    break

            # Consume energy for work done
            await self.consume_energy(
                self.energy_consumption_rate * max(messages_processed, 1)
            )

            self.iteration_count += 1
            self.state = AgentState.IDLE

        except Exception as e:
            self.state = AgentState.ERROR
            self._logger.exception("Error in agent cycle", error=str(e))
            raise AgentError(
                str(e),
                agent_id=str(self.agent_id),
                agent_role=self.role.value,
                state=self.state.value,
            )

        return {
            "agent_id": str(self.agent_id)[:8],
            "messages_processed": messages_processed,
            "energy": self.energy,
            "state": self.state.value,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "role": self.role.value,
            "state": self.state.value,
            "energy": self.energy,
            "iteration_count": self.iteration_count,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "tasks_completed": self.tasks_completed,
            "queue_size": self.message_queue.qsize(),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}("
            f"id={str(self.agent_id)[:8]}, "
            f"role={self.role.value}, "
            f"state={self.state.value})>"
        )
