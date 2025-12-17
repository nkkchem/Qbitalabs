"""
Type Definitions for QBitaLabs Platform

Provides type aliases and custom types used throughout the platform:
- ID types for various entities
- Enums for state and status
- Protocol definitions
- Generic type aliases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, NewType, Protocol, TypeVar, runtime_checkable
from uuid import uuid4


# ID Types - NewType for type safety
AgentID = NewType("AgentID", str)
MessageID = NewType("MessageID", str)
JobID = NewType("JobID", str)
CircuitID = NewType("CircuitID", str)
TwinID = NewType("TwinID", str)
PathwayID = NewType("PathwayID", str)
MoleculeID = NewType("MoleculeID", str)
CohortID = NewType("CohortID", str)


def generate_agent_id() -> AgentID:
    """Generate a new agent ID."""
    return AgentID(str(uuid4()))


def generate_message_id() -> MessageID:
    """Generate a new message ID."""
    return MessageID(str(uuid4()))


def generate_job_id() -> JobID:
    """Generate a new job ID."""
    return JobID(str(uuid4()))


def generate_circuit_id() -> CircuitID:
    """Generate a new circuit ID."""
    return CircuitID(str(uuid4()))


def generate_twin_id() -> TwinID:
    """Generate a new twin ID."""
    return TwinID(str(uuid4()))


# Enums


class ExecutionStatus(str, Enum):
    """Status of task/job execution."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ComputeBackend(str, Enum):
    """Available compute backends."""

    CPU = "cpu"
    GPU = "gpu"
    QUANTUM = "quantum"
    NEUROMORPHIC = "neuromorphic"


class AgentState(str, Enum):
    """State of a SWARM agent."""

    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    PROCESSING = "processing"
    SIGNALING = "signaling"
    TERMINATED = "terminated"
    ERROR = "error"


class AgentRole(str, Enum):
    """Role of a SWARM agent."""

    MOLECULAR_MODELER = "molecular_modeler"
    PATHWAY_SIMULATOR = "pathway_simulator"
    PATIENT_RISK = "patient_risk"
    TRIAL_DESIGNER = "trial_designer"
    LITERATURE_REVIEWER = "literature_reviewer"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    VALIDATION_AGENT = "validation_agent"
    COHORT_MANAGER = "cohort_manager"
    QUANTUM_EXECUTOR = "quantum_executor"
    NEUROMORPHIC_PROCESSOR = "neuromorphic_processor"


class MessageType(str, Enum):
    """Types of messages in the SWARM."""

    SIGNAL = "signal"
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    QUANTUM_REQUEST = "quantum_request"
    QUANTUM_RESULT = "quantum_result"
    NEUROMORPHIC_REQUEST = "neuromorphic_request"
    NEUROMORPHIC_RESULT = "neuromorphic_result"


class MessagePriority(int, Enum):
    """Priority levels for messages."""

    LOW = 1
    NORMAL = 5
    HIGH = 7
    CRITICAL = 10


class ProteinState(str, Enum):
    """State of a protein-like agent."""

    UNFOLDED = "unfolded"
    FOLDING = "folding"
    FOLDED = "folded"
    ACTIVE = "active"
    BOUND = "bound"
    DEGRADING = "degrading"


class SignalType(str, Enum):
    """Types of biological-inspired signals."""

    PHEROMONE = "pheromone"
    CONFORMATIONAL = "conformational"
    ACTIVATION = "activation"
    INHIBITION = "inhibition"
    RECRUITMENT = "recruitment"


# Data Classes


@dataclass
class Coordinates:
    """3D coordinates."""

    x: float
    y: float
    z: float

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class MolecularGeometry:
    """Molecular geometry specification."""

    atoms: list[tuple[str, Coordinates]] = field(default_factory=list)

    def to_string(self) -> str:
        """Convert to string format for quantum chemistry."""
        parts = []
        for atom, coord in self.atoms:
            parts.append(f"{atom} {coord.x} {coord.y} {coord.z}")
        return "; ".join(parts)


@dataclass
class QuantumJobResult:
    """Result from a quantum job."""

    job_id: JobID
    status: ExecutionStatus
    result: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float | None = None
    shots: int | None = None
    backend: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


@dataclass
class BiosignalData:
    """Container for biosignal data."""

    signal_type: str  # ecg, eeg, emg, ppg
    samples: list[float] = field(default_factory=list)
    sampling_rate: int = 250
    channels: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Protocols for duck typing


@runtime_checkable
class Processable(Protocol):
    """Protocol for objects that can be processed."""

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process input data and return results."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Create instance from dictionary."""
        ...


@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with an ID."""

    @property
    def id(self) -> str:
        """Get the unique identifier."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for objects that support health checks."""

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        ...


# Generic type variables
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

# Type aliases
JSON = dict[str, Any]
JSONList = list[dict[str, Any]]
Callback = type[Any]  # Callable type alias
