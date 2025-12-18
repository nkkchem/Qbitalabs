"""
Configuration Management for QBitaLabs Platform

Provides centralized configuration management:
- Environment-based configuration
- YAML configuration files
- Type-safe configuration dataclasses
- Configuration validation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SwarmTopology(str, Enum):
    """Swarm topology types."""

    FLAT = "flat"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    PROTEIN_CLUSTER = "protein_cluster"


class QuantumBackendType(str, Enum):
    """Available quantum backend types."""

    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    IONQ = "ionq"
    SIMULATOR = "simulator"


class NeuromorphicBackendType(str, Enum):
    """Available neuromorphic backend types."""

    AKIDA = "akida"
    LOIHI = "loihi"
    SYNSENSE = "synsense"
    BRIAN2 = "brian2"
    SIMULATOR = "simulator"


@dataclass
class SwarmConfig:
    """Configuration for the SWARM agent system."""

    max_agents: int = 1000
    topology: SwarmTopology = SwarmTopology.HIERARCHICAL
    pheromone_decay_rate: float = 0.05
    energy_regeneration_rate: float = 0.02
    message_ttl_default: int = 100
    consensus_threshold: float = 0.67
    max_concurrent_tasks: int = 100
    quantum_task_priority: int = 8
    neuromorphic_task_priority: int = 7
    default_llm_model: str = "claude-sonnet-4-20250514"
    agent_timeout_seconds: float = 30.0
    enable_metrics: bool = True

    def __post_init__(self):
        if self.max_agents < 1:
            raise ValueError("max_agents must be at least 1")
        if not 0 <= self.pheromone_decay_rate <= 1:
            raise ValueError("pheromone_decay_rate must be between 0 and 1")
        if not 0 <= self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be between 0 and 1")


@dataclass
class QuantumConfig:
    """Configuration for quantum computing layer."""

    default_backend: QuantumBackendType = QuantumBackendType.SIMULATOR
    default_shots: int = 4096
    optimization_level: int = 3
    resilience_level: int = 1
    use_error_mitigation: bool = True

    # IBM Qiskit settings
    ibm_channel: str = "ibm_quantum"
    ibm_instance: str = "ibm-q/open/main"
    ibm_backend_name: str = "ibm_brisbane"

    # Cirq settings
    cirq_use_simulator: bool = True

    # PennyLane settings
    pennylane_device: str = "default.qubit"

    # Timeouts
    job_timeout_seconds: float = 300.0
    max_retries: int = 3


@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic computing layer."""

    default_backend: NeuromorphicBackendType = NeuromorphicBackendType.BRIAN2
    enable_hardware_detection: bool = True

    # Akida settings
    akida_input_scaling: tuple[int, int] = (0, 255)

    # SNN settings
    default_time_step_ms: float = 1.0
    default_simulation_duration_ms: float = 100.0

    # Biosignal processing
    ecg_sampling_rate: int = 250
    eeg_sampling_rate: int = 256
    emg_sampling_rate: int = 1000


@dataclass
class APIConfig:
    """Configuration for the API layer."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable_cors: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    enable_docs: bool = True
    api_prefix: str = "/api/v1"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""

    url: str = "postgresql://localhost:5432/qbitalabs"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class RedisConfig:
    """Configuration for Redis."""

    url: str = "redis://localhost:6379/0"
    max_connections: int = 10


class QBitaConfig(BaseModel):
    """
    Main configuration for the QBitaLabs platform.

    Combines all sub-configurations and provides environment-aware defaults.
    """

    environment: Environment = Field(default=Environment.DEVELOPMENT)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    debug: bool = Field(default=False)

    # Sub-configurations (using dict for pydantic compatibility)
    swarm: dict[str, Any] = Field(default_factory=dict)
    quantum: dict[str, Any] = Field(default_factory=dict)
    neuromorphic: dict[str, Any] = Field(default_factory=dict)
    api: dict[str, Any] = Field(default_factory=dict)
    database: dict[str, Any] = Field(default_factory=dict)
    redis: dict[str, Any] = Field(default_factory=dict)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str | Environment) -> Environment:
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str | LogLevel) -> LogLevel:
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v

    def get_swarm_config(self) -> SwarmConfig:
        """Get SwarmConfig from dict."""
        return SwarmConfig(**self.swarm)

    def get_quantum_config(self) -> QuantumConfig:
        """Get QuantumConfig from dict."""
        return QuantumConfig(**self.quantum)

    def get_neuromorphic_config(self) -> NeuromorphicConfig:
        """Get NeuromorphicConfig from dict."""
        return NeuromorphicConfig(**self.neuromorphic)

    def get_api_config(self) -> APIConfig:
        """Get APIConfig from dict."""
        return APIConfig(**self.api)


# Global configuration instance
_config: QBitaConfig | None = None


def load_config(
    config_path: str | Path | None = None,
    env: str | None = None,
) -> QBitaConfig:
    """
    Load configuration from file and environment.

    Args:
        config_path: Path to YAML config file. If None, looks for default locations.
        env: Environment override.

    Returns:
        QBitaConfig instance.

    Example:
        >>> config = load_config("configs/production.yaml")
        >>> print(config.environment)
    """
    global _config

    # Determine environment
    environment = env or os.getenv("QBITALABS_ENV", "development")

    # Find config file
    if config_path is None:
        possible_paths = [
            Path(f"configs/{environment}.yaml"),
            Path("configs/default.yaml"),
            Path("config.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    # Load YAML config if exists
    config_data: dict[str, Any] = {"environment": environment}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)
            if yaml_data:
                config_data.update(yaml_data)

    # Override with environment variables
    env_overrides = {
        "QBITALABS_LOG_LEVEL": "log_level",
        "QBITALABS_DEBUG": "debug",
    }

    for env_var, config_key in env_overrides.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            if config_key == "debug":
                config_data[config_key] = value.lower() in ("true", "1", "yes")
            else:
                config_data[config_key] = value

    _config = QBitaConfig(**config_data)
    return _config


def get_config() -> QBitaConfig:
    """
    Get the global configuration instance.

    Returns:
        QBitaConfig instance.

    Raises:
        RuntimeError: If configuration has not been loaded.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
