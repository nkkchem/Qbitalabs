"""
Message Bus for QBitaLabs SWARM Architecture

Provides asynchronous message passing between agents:
- Priority-based message queuing
- Topic-based pub/sub
- Request-response patterns
- Message persistence (optional)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from heapq import heappop, heappush
from typing import Any, Callable, Awaitable
from uuid import uuid4

import structlog

from qbitalabs.core.types import (
    AgentID,
    AgentRole,
    MessageID,
    MessagePriority,
    MessageType,
)
from qbitalabs.swarm.base_agent import AgentMessage
from qbitalabs.core.exceptions import MessageBusError

logger = structlog.get_logger(__name__)


@dataclass(order=True)
class PrioritizedMessage:
    """Wrapper for priority queue ordering."""

    priority: int
    timestamp: float = field(compare=True)
    message: AgentMessage = field(compare=False)


class MessageBus:
    """
    Asynchronous message bus for SWARM agent communication.

    Features:
    - Priority-based message delivery
    - Topic-based subscriptions
    - Role-based routing
    - Dead letter queue for failed messages

    Example:
        >>> bus = MessageBus()
        >>> await bus.start()
        >>> await bus.publish(message)
        >>> bus.subscribe("topic", callback)
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        enable_persistence: bool = False,
    ):
        """
        Initialize the message bus.

        Args:
            max_queue_size: Maximum messages in queue.
            enable_persistence: Whether to persist messages.
        """
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence

        # Message queues
        self._global_queue: list[PrioritizedMessage] = []
        self._agent_queues: dict[AgentID, asyncio.Queue[AgentMessage]] = {}
        self._role_queues: dict[AgentRole, list[AgentID]] = defaultdict(list)

        # Subscriptions
        self._topic_subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._type_subscribers: dict[MessageType, list[Callable]] = defaultdict(list)

        # Message handlers
        self._handlers: dict[AgentID, Callable[[AgentMessage], Awaitable[None]]] = {}

        # Dead letter queue
        self._dead_letters: list[AgentMessage] = []
        self._max_dead_letters = 1000

        # State
        self._running = False
        self._processor_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Metrics
        self.messages_published = 0
        self.messages_delivered = 0
        self.messages_failed = 0

        self._logger = structlog.get_logger("message_bus")

    async def start(self) -> None:
        """Start the message bus processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
        self._logger.info("Message bus started")

    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Message bus stopped")

    def register_agent(
        self,
        agent_id: AgentID,
        role: AgentRole,
        handler: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """
        Register an agent with the message bus.

        Args:
            agent_id: Agent identifier.
            role: Agent role for role-based routing.
            handler: Callback for message delivery.
        """
        self._agent_queues[agent_id] = asyncio.Queue()
        self._role_queues[role].append(agent_id)
        self._handlers[agent_id] = handler
        self._logger.debug("Agent registered", agent_id=str(agent_id)[:8])

    def unregister_agent(self, agent_id: AgentID, role: AgentRole) -> None:
        """
        Unregister an agent from the message bus.

        Args:
            agent_id: Agent identifier.
            role: Agent role.
        """
        self._agent_queues.pop(agent_id, None)
        self._handlers.pop(agent_id, None)
        if agent_id in self._role_queues[role]:
            self._role_queues[role].remove(agent_id)
        self._logger.debug("Agent unregistered", agent_id=str(agent_id)[:8])

    async def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the bus.

        Args:
            message: Message to publish.

        Raises:
            MessageBusError: If queue is full.
        """
        if len(self._global_queue) >= self.max_queue_size:
            raise MessageBusError(
                "Message queue is full",
                message_id=str(message.id),
            )

        async with self._lock:
            # Negate priority for max-heap behavior (higher priority = lower number)
            priority_value = -message.priority.value
            heappush(
                self._global_queue,
                PrioritizedMessage(
                    priority=priority_value,
                    timestamp=message.timestamp.timestamp(),
                    message=message,
                ),
            )

        self.messages_published += 1

        # Notify topic subscribers
        topic = message.payload.get("topic")
        if topic and topic in self._topic_subscribers:
            for callback in self._topic_subscribers[topic]:
                try:
                    await callback(message)
                except Exception as e:
                    self._logger.exception("Topic callback error", error=str(e))

        # Notify type subscribers
        for callback in self._type_subscribers.get(message.message_type, []):
            try:
                await callback(message)
            except Exception as e:
                self._logger.exception("Type callback error", error=str(e))

    async def publish_to_role(
        self,
        message: AgentMessage,
        role: AgentRole,
    ) -> None:
        """
        Publish a message to all agents with a specific role.

        Args:
            message: Message to publish.
            role: Target role.
        """
        for agent_id in self._role_queues.get(role, []):
            if agent_id != message.sender_id:
                role_message = AgentMessage(
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    message_type=message.message_type,
                    payload=message.payload,
                    priority=message.priority,
                    pheromone_strength=message.pheromone_strength,
                    ttl=message.ttl,
                )
                await self.publish(role_message)

    def subscribe_topic(
        self,
        topic: str,
        callback: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """
        Subscribe to a topic.

        Args:
            topic: Topic to subscribe to.
            callback: Callback for messages.
        """
        self._topic_subscribers[topic].append(callback)

    def subscribe_type(
        self,
        message_type: MessageType,
        callback: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """
        Subscribe to a message type.

        Args:
            message_type: Message type to subscribe to.
            callback: Callback for messages.
        """
        self._type_subscribers[message_type].append(callback)

    def unsubscribe_topic(
        self,
        topic: str,
        callback: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """Unsubscribe from a topic."""
        if callback in self._topic_subscribers.get(topic, []):
            self._topic_subscribers[topic].remove(callback)

    async def _process_messages(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                if not self._global_queue:
                    await asyncio.sleep(0.001)  # Small sleep to prevent busy loop
                    continue

                async with self._lock:
                    if not self._global_queue:
                        continue
                    prioritized = heappop(self._global_queue)

                message = prioritized.message

                # Apply decay
                message.decay()

                # Skip expired messages
                if message.is_expired():
                    continue

                # Deliver message
                await self._deliver_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception("Error processing message", error=str(e))

    async def _deliver_message(self, message: AgentMessage) -> None:
        """
        Deliver a message to its recipient(s).

        Args:
            message: Message to deliver.
        """
        try:
            if message.recipient_id:
                # Direct delivery
                handler = self._handlers.get(message.recipient_id)
                if handler:
                    await handler(message)
                    self.messages_delivered += 1
                else:
                    self._add_to_dead_letter(message, "Recipient not found")
            else:
                # Broadcast to all agents
                for agent_id, handler in self._handlers.items():
                    if agent_id != message.sender_id:
                        try:
                            await handler(message)
                            self.messages_delivered += 1
                        except Exception as e:
                            self._logger.warning(
                                "Delivery failed",
                                agent_id=str(agent_id)[:8],
                                error=str(e),
                            )

        except Exception as e:
            self.messages_failed += 1
            self._add_to_dead_letter(message, str(e))
            self._logger.exception("Message delivery failed", error=str(e))

    def _add_to_dead_letter(self, message: AgentMessage, reason: str) -> None:
        """Add a message to the dead letter queue."""
        message.payload["_dead_letter_reason"] = reason
        message.payload["_dead_letter_time"] = datetime.utcnow().isoformat()
        self._dead_letters.append(message)

        # Trim dead letter queue if too large
        if len(self._dead_letters) > self._max_dead_letters:
            self._dead_letters = self._dead_letters[-self._max_dead_letters:]

    def get_dead_letters(self, limit: int = 100) -> list[AgentMessage]:
        """Get messages from the dead letter queue."""
        return self._dead_letters[-limit:]

    def get_metrics(self) -> dict[str, Any]:
        """Get message bus metrics."""
        return {
            "messages_published": self.messages_published,
            "messages_delivered": self.messages_delivered,
            "messages_failed": self.messages_failed,
            "queue_size": len(self._global_queue),
            "dead_letter_count": len(self._dead_letters),
            "registered_agents": len(self._handlers),
            "running": self._running,
        }

    async def drain(self, timeout: float = 5.0) -> int:
        """
        Wait for all messages to be processed.

        Args:
            timeout: Maximum time to wait.

        Returns:
            Number of messages remaining.
        """
        start = asyncio.get_event_loop().time()
        while self._global_queue and (asyncio.get_event_loop().time() - start) < timeout:
            await asyncio.sleep(0.01)
        return len(self._global_queue)
