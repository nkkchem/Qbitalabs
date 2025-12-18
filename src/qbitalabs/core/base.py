"""
Base Classes for QBitaLabs Platform

Provides foundational classes used throughout the platform:
- Component: Base class for all platform components
- Configurable: Mixin for components that accept configuration
- Observable: Mixin for components that emit events
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar
from uuid import uuid4

import structlog

T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


@dataclass
class ComponentMetadata:
    """Metadata for platform components."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    version: str = "0.1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: dict[str, str] = field(default_factory=dict)


class Component(ABC):
    """
    Base class for all QBitaLabs platform components.

    Provides:
    - Unique identification
    - Logging infrastructure
    - Lifecycle management (initialize, start, stop)
    - Health checking

    Example:
        >>> class MyAgent(Component):
        ...     async def initialize(self):
        ...         self.logger.info("Initializing agent")
        ...
        ...     async def start(self):
        ...         self.logger.info("Starting agent")
        ...
        ...     async def stop(self):
        ...         self.logger.info("Stopping agent")
    """

    def __init__(self, name: str | None = None):
        """
        Initialize the component.

        Args:
            name: Optional name for the component. Defaults to class name.
        """
        self._metadata = ComponentMetadata(
            name=name or self.__class__.__name__,
        )
        self._logger = structlog.get_logger(
            component=self._metadata.name,
            component_id=self._metadata.id[:8],
        )
        self._initialized = False
        self._running = False

    @property
    def id(self) -> str:
        """Get the unique component ID."""
        return self._metadata.id

    @property
    def name(self) -> str:
        """Get the component name."""
        return self._metadata.name

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get the component logger."""
        return self._logger

    @property
    def is_initialized(self) -> bool:
        """Check if the component is initialized."""
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if the component is running."""
        return self._running

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the component.

        Called once before the component is started.
        Override this method to perform initialization tasks.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Start the component.

        Called after initialization to begin component operation.
        Override this method to implement startup logic.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the component.

        Called to gracefully shutdown the component.
        Override this method to implement cleanup logic.
        """
        pass

    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the component.

        Returns:
            Dictionary containing health status information.
        """
        return {
            "component": self.name,
            "id": self.id[:8],
            "initialized": self._initialized,
            "running": self._running,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}("
            f"id={self.id[:8]}, "
            f"name={self.name}, "
            f"running={self._running})>"
        )


class Configurable(Generic[ConfigT]):
    """
    Mixin for components that accept configuration.

    Provides configuration validation and access.

    Example:
        >>> @dataclass
        ... class MyConfig:
        ...     max_retries: int = 3
        ...
        >>> class MyService(Component, Configurable[MyConfig]):
        ...     def __init__(self, config: MyConfig):
        ...         super().__init__()
        ...         self.set_config(config)
    """

    _config: ConfigT | None = None

    def set_config(self, config: ConfigT) -> None:
        """
        Set the configuration.

        Args:
            config: Configuration object to set.
        """
        self._config = config

    @property
    def config(self) -> ConfigT:
        """
        Get the configuration.

        Returns:
            The configuration object.

        Raises:
            RuntimeError: If configuration is not set.
        """
        if self._config is None:
            raise RuntimeError(
                f"Configuration not set for {self.__class__.__name__}. "
                "Call set_config() first."
            )
        return self._config


EventCallback = Callable[[str, dict[str, Any]], None]


class Observable:
    """
    Mixin for components that emit events.

    Implements the observer pattern for event-driven communication.

    Example:
        >>> class MyAgent(Component, Observable):
        ...     async def process(self, data):
        ...         result = await self._do_processing(data)
        ...         await self.emit("processing_complete", {"result": result})
        ...
        >>> agent = MyAgent()
        >>> agent.subscribe("processing_complete", lambda e, d: print(d))
    """

    def __init__(self) -> None:
        """Initialize the observable."""
        self._observers: dict[str, list[EventCallback]] = {}
        self._async_observers: dict[str, list[Callable]] = {}

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """
        Subscribe to an event.

        Args:
            event: Name of the event to subscribe to.
            callback: Function to call when event is emitted.
        """
        if event not in self._observers:
            self._observers[event] = []
        self._observers[event].append(callback)

    def subscribe_async(
        self,
        event: str,
        callback: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        """
        Subscribe to an event with an async callback.

        Args:
            event: Name of the event to subscribe to.
            callback: Async function to call when event is emitted.
        """
        if event not in self._async_observers:
            self._async_observers[event] = []
        self._async_observers[event].append(callback)

    def unsubscribe(self, event: str, callback: EventCallback) -> None:
        """
        Unsubscribe from an event.

        Args:
            event: Name of the event to unsubscribe from.
            callback: The callback to remove.
        """
        if event in self._observers:
            try:
                self._observers[event].remove(callback)
            except ValueError:
                pass

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event: Name of the event to emit.
            data: Data to pass to subscribers.
        """
        data = data or {}

        # Sync observers
        for callback in self._observers.get(event, []):
            try:
                callback(event, data)
            except Exception as e:
                logging.exception(f"Error in event callback for {event}: {e}")

        # Async observers
        tasks = []
        for callback in self._async_observers.get(event, []):
            tasks.append(asyncio.create_task(callback(event, data)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class Singleton(type):
    """
    Metaclass for creating singleton classes.

    Example:
        >>> class MySingleton(metaclass=Singleton):
        ...     pass
        >>> a = MySingleton()
        >>> b = MySingleton()
        >>> a is b
        True
    """

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def reset(mcs, cls: type) -> None:
        """Reset the singleton instance for a class."""
        if cls in mcs._instances:
            del mcs._instances[cls]
