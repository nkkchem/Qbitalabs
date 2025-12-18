"""
QBitaLabs Core Module

Core components and utilities shared across the platform:
- Configuration management
- Base classes and interfaces
- Custom exceptions
- Type definitions
- Component registry
"""

from __future__ import annotations

from qbitalabs.core.base import Component, Configurable, Observable
from qbitalabs.core.config import (
    QBitaConfig,
    SwarmConfig,
    QuantumConfig,
    NeuromorphicConfig,
    get_config,
    load_config,
)
from qbitalabs.core.exceptions import (
    QBitaError,
    ConfigurationError,
    SwarmError,
    AgentError,
    QuantumError,
    NeuromorphicError,
    ValidationError,
)
from qbitalabs.core.types import (
    AgentID,
    MessageID,
    JobID,
    CircuitID,
    TwinID,
    ComputeBackend,
    ExecutionStatus,
)
from qbitalabs.core.registry import Registry, get_registry

__all__ = [
    # Base
    "Component",
    "Configurable",
    "Observable",
    # Config
    "QBitaConfig",
    "SwarmConfig",
    "QuantumConfig",
    "NeuromorphicConfig",
    "get_config",
    "load_config",
    # Exceptions
    "QBitaError",
    "ConfigurationError",
    "SwarmError",
    "AgentError",
    "QuantumError",
    "NeuromorphicError",
    "ValidationError",
    # Types
    "AgentID",
    "MessageID",
    "JobID",
    "CircuitID",
    "TwinID",
    "ComputeBackend",
    "ExecutionStatus",
    # Registry
    "Registry",
    "get_registry",
]
