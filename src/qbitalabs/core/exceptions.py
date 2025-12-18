"""
Custom Exceptions for QBitaLabs Platform

Provides a hierarchy of exceptions for different platform components:
- QBitaError: Base exception for all platform errors
- Component-specific exceptions for swarm, quantum, neuromorphic, etc.
"""

from __future__ import annotations

from typing import Any


class QBitaError(Exception):
    """
    Base exception for all QBitaLabs platform errors.

    Attributes:
        message: Human-readable error message.
        code: Error code for programmatic handling.
        details: Additional error details.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            code: Error code for programmatic handling.
            details: Additional error details.
        """
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        if self.details:
            return f"{self.code}: {self.message} - {self.details}"
        return f"{self.code}: {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r})"


# Configuration Errors


class ConfigurationError(QBitaError):
    """Raised when there is a configuration error."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual

        super().__init__(message, code="CONFIG_ERROR", details=details)


class ValidationError(QBitaError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        constraints: list[str] | None = None,
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if constraints:
            details["constraints"] = constraints

        super().__init__(message, code="VALIDATION_ERROR", details=details)


# SWARM Errors


class SwarmError(QBitaError):
    """Base exception for SWARM-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "SWARM_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code=code, details=details)


class AgentError(SwarmError):
    """Raised when an agent encounters an error."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        agent_role: str | None = None,
        state: str | None = None,
    ):
        details = {}
        if agent_id:
            details["agent_id"] = agent_id
        if agent_role:
            details["agent_role"] = agent_role
        if state:
            details["state"] = state

        super().__init__(message, code="AGENT_ERROR", details=details)


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        timeout_seconds: float | None = None,
    ):
        details = {}
        if agent_id:
            details["agent_id"] = agent_id
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(message, agent_id=agent_id)
        self.code = "AGENT_TIMEOUT"
        self.details.update(details)


class SwarmCapacityError(SwarmError):
    """Raised when swarm capacity is exceeded."""

    def __init__(
        self,
        message: str,
        current_agents: int | None = None,
        max_agents: int | None = None,
    ):
        details = {}
        if current_agents is not None:
            details["current_agents"] = current_agents
        if max_agents is not None:
            details["max_agents"] = max_agents

        super().__init__(message, code="SWARM_CAPACITY_ERROR", details=details)


class MessageBusError(SwarmError):
    """Raised when message bus encounters an error."""

    def __init__(
        self,
        message: str,
        message_id: str | None = None,
        sender_id: str | None = None,
        recipient_id: str | None = None,
    ):
        details = {}
        if message_id:
            details["message_id"] = message_id
        if sender_id:
            details["sender_id"] = sender_id
        if recipient_id:
            details["recipient_id"] = recipient_id

        super().__init__(message, code="MESSAGE_BUS_ERROR", details=details)


# Quantum Errors


class QuantumError(QBitaError):
    """Base exception for quantum computing errors."""

    def __init__(
        self,
        message: str,
        code: str = "QUANTUM_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code=code, details=details)


class QuantumBackendError(QuantumError):
    """Raised when a quantum backend encounters an error."""

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        backend_status: str | None = None,
    ):
        details = {}
        if backend:
            details["backend"] = backend
        if backend_status:
            details["backend_status"] = backend_status

        super().__init__(message, code="QUANTUM_BACKEND_ERROR", details=details)


class QuantumCircuitError(QuantumError):
    """Raised when there is an error with a quantum circuit."""

    def __init__(
        self,
        message: str,
        circuit_id: str | None = None,
        num_qubits: int | None = None,
        depth: int | None = None,
    ):
        details = {}
        if circuit_id:
            details["circuit_id"] = circuit_id
        if num_qubits is not None:
            details["num_qubits"] = num_qubits
        if depth is not None:
            details["depth"] = depth

        super().__init__(message, code="QUANTUM_CIRCUIT_ERROR", details=details)


class QuantumJobError(QuantumError):
    """Raised when a quantum job fails."""

    def __init__(
        self,
        message: str,
        job_id: str | None = None,
        job_status: str | None = None,
    ):
        details = {}
        if job_id:
            details["job_id"] = job_id
        if job_status:
            details["job_status"] = job_status

        super().__init__(message, code="QUANTUM_JOB_ERROR", details=details)


class HamiltonianError(QuantumError):
    """Raised when there is an error constructing a Hamiltonian."""

    def __init__(
        self,
        message: str,
        molecule: str | None = None,
        basis: str | None = None,
    ):
        details = {}
        if molecule:
            details["molecule"] = molecule
        if basis:
            details["basis"] = basis

        super().__init__(message, code="HAMILTONIAN_ERROR", details=details)


# Neuromorphic Errors


class NeuromorphicError(QBitaError):
    """Base exception for neuromorphic computing errors."""

    def __init__(
        self,
        message: str,
        code: str = "NEUROMORPHIC_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code=code, details=details)


class NeuromorphicBackendError(NeuromorphicError):
    """Raised when a neuromorphic backend encounters an error."""

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        device_available: bool | None = None,
    ):
        details = {}
        if backend:
            details["backend"] = backend
        if device_available is not None:
            details["device_available"] = device_available

        super().__init__(message, code="NEUROMORPHIC_BACKEND_ERROR", details=details)


class SNNError(NeuromorphicError):
    """Raised when there is an error with a spiking neural network."""

    def __init__(
        self,
        message: str,
        num_neurons: int | None = None,
        num_layers: int | None = None,
    ):
        details = {}
        if num_neurons is not None:
            details["num_neurons"] = num_neurons
        if num_layers is not None:
            details["num_layers"] = num_layers

        super().__init__(message, code="SNN_ERROR", details=details)


class BiosignalError(NeuromorphicError):
    """Raised when there is an error processing biosignals."""

    def __init__(
        self,
        message: str,
        signal_type: str | None = None,
        sampling_rate: int | None = None,
    ):
        details = {}
        if signal_type:
            details["signal_type"] = signal_type
        if sampling_rate is not None:
            details["sampling_rate"] = sampling_rate

        super().__init__(message, code="BIOSIGNAL_ERROR", details=details)


# Digital Twin Errors


class DigitalTwinError(QBitaError):
    """Base exception for digital twin errors."""

    def __init__(
        self,
        message: str,
        code: str = "DIGITAL_TWIN_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code=code, details=details)


class TwinNotFoundError(DigitalTwinError):
    """Raised when a digital twin is not found."""

    def __init__(self, message: str, twin_id: str | None = None):
        details = {}
        if twin_id:
            details["twin_id"] = twin_id

        super().__init__(message, code="TWIN_NOT_FOUND", details=details)


class SimulationError(DigitalTwinError):
    """Raised when a simulation fails."""

    def __init__(
        self,
        message: str,
        simulation_id: str | None = None,
        step: int | None = None,
    ):
        details = {}
        if simulation_id:
            details["simulation_id"] = simulation_id
        if step is not None:
            details["step"] = step

        super().__init__(message, code="SIMULATION_ERROR", details=details)


# API Errors


class APIError(QBitaError):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "API_ERROR",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code=code, details=details)
        self.status_code = status_code


class NotFoundError(APIError):
    """Raised when a resource is not found."""

    def __init__(self, message: str, resource: str | None = None):
        details = {}
        if resource:
            details["resource"] = resource

        super().__init__(message, status_code=404, code="NOT_FOUND", details=details)


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401, code="AUTHENTICATION_ERROR")


class AuthorizationError(APIError):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, status_code=403, code="AUTHORIZATION_ERROR")


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ):
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after

        super().__init__(
            message, status_code=429, code="RATE_LIMIT_ERROR", details=details
        )
