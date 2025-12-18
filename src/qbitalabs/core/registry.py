"""
Component Registry for QBitaLabs Platform

Provides a centralized registry for platform components:
- Agent registration
- Backend registration
- Service discovery
- Dependency injection support
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Generic, TypeVar

import structlog

from qbitalabs.core.base import Component, Singleton
from qbitalabs.core.types import AgentID, AgentRole

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class RegistryEntry(Generic[T]):
    """Entry in the registry."""

    name: str
    instance: T
    factory: Callable[..., T] | None = None
    singleton: bool = True
    tags: dict[str, str] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry(metaclass=Singleton):
    """
    Centralized registry for platform components.

    Supports:
    - Singleton and factory registration
    - Tag-based lookup
    - Type-safe retrieval
    - Lifecycle management

    Example:
        >>> registry = Registry()
        >>> registry.register("quantum_backend", QiskitBackend())
        >>> backend = registry.get("quantum_backend")
    """

    def __init__(self):
        """Initialize the registry."""
        self._entries: dict[str, RegistryEntry] = {}
        self._by_type: dict[type, list[str]] = defaultdict(list)
        self._by_tag: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._lock = Lock()
        self._logger = structlog.get_logger("registry")

    def register(
        self,
        name: str,
        instance: T | None = None,
        factory: Callable[..., T] | None = None,
        singleton: bool = True,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a component.

        Args:
            name: Unique name for the component.
            instance: Instance of the component (for singletons).
            factory: Factory function to create instances.
            singleton: Whether to cache the instance.
            tags: Tags for categorization and lookup.
            metadata: Additional metadata.
            overwrite: Whether to overwrite existing registration.

        Raises:
            ValueError: If neither instance nor factory provided, or name exists.
        """
        if instance is None and factory is None:
            raise ValueError("Either instance or factory must be provided")

        with self._lock:
            if name in self._entries and not overwrite:
                raise ValueError(f"Component '{name}' already registered")

            entry = RegistryEntry(
                name=name,
                instance=instance,
                factory=factory,
                singleton=singleton,
                tags=tags or {},
                metadata=metadata or {},
            )

            self._entries[name] = entry

            # Index by type
            if instance is not None:
                self._by_type[type(instance)].append(name)

            # Index by tags
            for tag_key, tag_value in entry.tags.items():
                self._by_tag[tag_key][tag_value].append(name)

            self._logger.info(
                "Component registered",
                name=name,
                singleton=singleton,
                tags=entry.tags,
            )

    def unregister(self, name: str) -> None:
        """
        Unregister a component.

        Args:
            name: Name of the component to unregister.
        """
        with self._lock:
            if name not in self._entries:
                return

            entry = self._entries[name]

            # Remove from type index
            if entry.instance is not None:
                instance_type = type(entry.instance)
                if name in self._by_type.get(instance_type, []):
                    self._by_type[instance_type].remove(name)

            # Remove from tag index
            for tag_key, tag_value in entry.tags.items():
                if name in self._by_tag.get(tag_key, {}).get(tag_value, []):
                    self._by_tag[tag_key][tag_value].remove(name)

            del self._entries[name]

            self._logger.info("Component unregistered", name=name)

    def get(self, name: str, default: T | None = None) -> T | None:
        """
        Get a component by name.

        Args:
            name: Name of the component.
            default: Default value if not found.

        Returns:
            Component instance or default.
        """
        with self._lock:
            entry = self._entries.get(name)

            if entry is None:
                return default

            if entry.instance is not None:
                return entry.instance

            if entry.factory is not None:
                instance = entry.factory()
                if entry.singleton:
                    entry.instance = instance
                return instance

            return default

    def get_or_raise(self, name: str) -> Any:
        """
        Get a component by name or raise an error.

        Args:
            name: Name of the component.

        Returns:
            Component instance.

        Raises:
            KeyError: If component not found.
        """
        result = self.get(name)
        if result is None:
            raise KeyError(f"Component '{name}' not found in registry")
        return result

    def get_by_type(self, component_type: type[T]) -> list[T]:
        """
        Get all components of a specific type.

        Args:
            component_type: Type of components to retrieve.

        Returns:
            List of matching component instances.
        """
        with self._lock:
            names = self._by_type.get(component_type, [])
            return [self.get(name) for name in names if self.get(name) is not None]

    def get_by_tag(self, tag_key: str, tag_value: str) -> list[Any]:
        """
        Get all components with a specific tag.

        Args:
            tag_key: Tag key to match.
            tag_value: Tag value to match.

        Returns:
            List of matching component instances.
        """
        with self._lock:
            names = self._by_tag.get(tag_key, {}).get(tag_value, [])
            return [self.get(name) for name in names if self.get(name) is not None]

    def has(self, name: str) -> bool:
        """
        Check if a component is registered.

        Args:
            name: Name of the component.

        Returns:
            True if component is registered.
        """
        return name in self._entries

    def list_all(self) -> list[str]:
        """
        List all registered component names.

        Returns:
            List of component names.
        """
        return list(self._entries.keys())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """
        Get metadata for a component.

        Args:
            name: Name of the component.

        Returns:
            Component metadata.
        """
        entry = self._entries.get(name)
        if entry is None:
            return {}
        return {
            "name": entry.name,
            "singleton": entry.singleton,
            "tags": entry.tags,
            "registered_at": entry.registered_at.isoformat(),
            "metadata": entry.metadata,
            "has_instance": entry.instance is not None,
            "has_factory": entry.factory is not None,
        }

    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._entries.clear()
            self._by_type.clear()
            self._by_tag.clear()
            self._logger.info("Registry cleared")


class AgentRegistry:
    """
    Specialized registry for SWARM agents.

    Provides agent-specific functionality:
    - Role-based lookup
    - Pool management
    - Agent lifecycle tracking
    """

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: dict[AgentID, Any] = {}
        self._by_role: dict[AgentRole, list[AgentID]] = defaultdict(list)
        self._pools: dict[str, list[AgentID]] = defaultdict(list)
        self._lock = Lock()
        self._logger = structlog.get_logger("agent_registry")

    def register_agent(
        self,
        agent_id: AgentID,
        agent: Any,
        role: AgentRole,
        pool_name: str | None = None,
    ) -> None:
        """
        Register an agent.

        Args:
            agent_id: Unique agent ID.
            agent: Agent instance.
            role: Agent role.
            pool_name: Optional pool name.
        """
        with self._lock:
            self._agents[agent_id] = agent
            self._by_role[role].append(agent_id)

            if pool_name:
                self._pools[pool_name].append(agent_id)

            self._logger.debug(
                "Agent registered",
                agent_id=str(agent_id)[:8],
                role=role.value,
                pool=pool_name,
            )

    def unregister_agent(self, agent_id: AgentID) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID to unregister.
        """
        with self._lock:
            if agent_id not in self._agents:
                return

            # Remove from role index
            for role, agents in self._by_role.items():
                if agent_id in agents:
                    agents.remove(agent_id)

            # Remove from pools
            for pool_name, agents in self._pools.items():
                if agent_id in agents:
                    agents.remove(agent_id)

            del self._agents[agent_id]

            self._logger.debug("Agent unregistered", agent_id=str(agent_id)[:8])

    def get_agent(self, agent_id: AgentID) -> Any | None:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_agents_by_role(self, role: AgentRole) -> list[Any]:
        """Get all agents with a specific role."""
        agent_ids = self._by_role.get(role, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_pool(self, pool_name: str) -> list[Any]:
        """Get all agents in a pool."""
        agent_ids = self._pools.get(pool_name, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def count(self) -> int:
        """Get total number of registered agents."""
        return len(self._agents)

    def count_by_role(self, role: AgentRole) -> int:
        """Get number of agents with a specific role."""
        return len(self._by_role.get(role, []))

    def list_agent_ids(self) -> list[AgentID]:
        """List all registered agent IDs."""
        return list(self._agents.keys())

    def clear(self) -> None:
        """Clear all agent registrations."""
        with self._lock:
            self._agents.clear()
            self._by_role.clear()
            self._pools.clear()


# Global registry instance
_registry: Registry | None = None
_agent_registry: AgentRegistry | None = None


def get_registry() -> Registry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


def reset_registries() -> None:
    """Reset all global registries (useful for testing)."""
    global _registry, _agent_registry
    if _registry is not None:
        _registry.clear()
    if _agent_registry is not None:
        _agent_registry.clear()
    _registry = None
    _agent_registry = None
