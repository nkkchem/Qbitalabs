"""
Variational Quantum Circuits for QBitaLabs

Implements parameterized circuits for variational algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CircuitParameter:
    """A parameter in a variational circuit."""

    name: str
    value: float = 0.0
    bounds: tuple[float, float] = (-np.pi, np.pi)
    trainable: bool = True


class VariationalCircuit(ABC):
    """
    Abstract base class for variational quantum circuits.

    Provides interface for:
    - Parameter management
    - Circuit construction
    - Gradient computation
    """

    def __init__(self, n_qubits: int, name: str = "variational"):
        """
        Initialize variational circuit.

        Args:
            n_qubits: Number of qubits.
            name: Circuit name.
        """
        self.n_qubits = n_qubits
        self.name = name
        self._parameters: list[CircuitParameter] = []
        self._logger = structlog.get_logger(f"circuit.{name}")

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the circuit."""
        return len(self._parameters)

    @property
    def parameter_values(self) -> np.ndarray:
        """Get current parameter values."""
        return np.array([p.value for p in self._parameters])

    @parameter_values.setter
    def parameter_values(self, values: np.ndarray) -> None:
        """Set parameter values."""
        for p, v in zip(self._parameters, values):
            p.value = v

    def add_parameter(
        self,
        name: str,
        initial_value: float = 0.0,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        trainable: bool = True,
    ) -> CircuitParameter:
        """Add a parameter to the circuit."""
        param = CircuitParameter(
            name=name,
            value=initial_value,
            bounds=bounds,
            trainable=trainable,
        )
        self._parameters.append(param)
        return param

    def initialize_random(self, seed: int | None = None) -> None:
        """Initialize parameters randomly."""
        rng = np.random.default_rng(seed)
        for p in self._parameters:
            p.value = rng.uniform(p.bounds[0], p.bounds[1])

    @abstractmethod
    def build(self, backend: Any) -> Any:
        """
        Build the circuit using a specific backend.

        Args:
            backend: Quantum backend to use.

        Returns:
            Backend-specific circuit object.
        """
        pass

    @abstractmethod
    def get_gradient_circuits(self) -> list[tuple[Any, float]]:
        """
        Get circuits for parameter-shift gradient computation.

        Returns:
            List of (circuit, coefficient) tuples.
        """
        pass


class HardwareEfficientAnsatz(VariationalCircuit):
    """
    Hardware-efficient variational ansatz.

    Uses native gates with minimal depth for NISQ devices.

    Structure per layer:
    1. Single-qubit rotations (RY, RZ)
    2. Entangling layer (CNOT ladder)

    Example:
        >>> ansatz = HardwareEfficientAnsatz(n_qubits=4, depth=3)
        >>> ansatz.initialize_random()
        >>> circuit = ansatz.build(backend)
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int = 2,
        entanglement: str = "linear",
        rotation_blocks: list[str] | None = None,
    ):
        """
        Initialize hardware-efficient ansatz.

        Args:
            n_qubits: Number of qubits.
            depth: Number of variational layers.
            entanglement: Entanglement pattern (linear, full, circular).
            rotation_blocks: Rotation gates to use (default: ["ry", "rz"]).
        """
        super().__init__(n_qubits, "hardware_efficient")

        self.depth = depth
        self.entanglement = entanglement
        self.rotation_blocks = rotation_blocks or ["ry", "rz"]

        # Create parameters
        for layer in range(depth):
            for qubit in range(n_qubits):
                for rot in self.rotation_blocks:
                    self.add_parameter(f"{rot}_{layer}_{qubit}")

    def build(self, backend: Any) -> Any:
        """Build the circuit."""
        circuit = backend.create_circuit(self.n_qubits, self.name)
        param_idx = 0

        for layer in range(self.depth):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                for rot in self.rotation_blocks:
                    theta = self._parameters[param_idx].value
                    if rot == "ry":
                        backend.add_ry_gate(circuit, qubit, theta)
                    elif rot == "rz":
                        backend.add_rz_gate(circuit, qubit, theta)
                    elif rot == "rx":
                        backend.add_rx_gate(circuit, qubit, theta)
                    param_idx += 1

            # Entangling layer
            self._add_entanglement(backend, circuit, layer)

        return circuit

    def _add_entanglement(self, backend: Any, circuit: Any, layer: int) -> None:
        """Add entanglement layer."""
        if self.entanglement == "linear":
            for i in range(self.n_qubits - 1):
                backend.add_cx_gate(circuit, i, i + 1)

        elif self.entanglement == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    backend.add_cx_gate(circuit, i, j)

        elif self.entanglement == "circular":
            for i in range(self.n_qubits):
                backend.add_cx_gate(circuit, i, (i + 1) % self.n_qubits)

    def get_gradient_circuits(self) -> list[tuple[Any, float]]:
        """Get parameter-shift gradient circuits."""
        # For each parameter, create shifted circuits
        gradients = []

        for i, param in enumerate(self._parameters):
            if not param.trainable:
                continue

            # Plus shift
            plus_params = self.parameter_values.copy()
            plus_params[i] += np.pi / 2

            # Minus shift
            minus_params = self.parameter_values.copy()
            minus_params[i] -= np.pi / 2

            gradients.append({
                "parameter": param.name,
                "plus_shift": plus_params,
                "minus_shift": minus_params,
                "coefficient": 0.5,  # Parameter shift rule coefficient
            })

        return gradients


class UCCSDAnsatz(VariationalCircuit):
    """
    Unitary Coupled Cluster Singles and Doubles ansatz.

    Used for molecular ground state calculations with VQE.
    Implements fermionic excitations mapped to qubit operations.

    Example:
        >>> ansatz = UCCSDAnsatz(n_qubits=4, n_electrons=2)
        >>> ansatz.initialize_random()
        >>> circuit = ansatz.build(backend)
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        include_singles: bool = True,
        include_doubles: bool = True,
    ):
        """
        Initialize UCCSD ansatz.

        Args:
            n_qubits: Number of qubits (spin orbitals).
            n_electrons: Number of electrons.
            include_singles: Include single excitations.
            include_doubles: Include double excitations.
        """
        super().__init__(n_qubits, "uccsd")

        self.n_electrons = n_electrons
        self.include_singles = include_singles
        self.include_doubles = include_doubles

        # Generate excitation operators
        self._singles = self._generate_singles()
        self._doubles = self._generate_doubles()

        # Create parameters for each excitation
        for i, (p, q) in enumerate(self._singles):
            if include_singles:
                self.add_parameter(f"t1_{p}_{q}")

        for i, (p, q, r, s) in enumerate(self._doubles):
            if include_doubles:
                self.add_parameter(f"t2_{p}_{q}_{r}_{s}")

    def _generate_singles(self) -> list[tuple[int, int]]:
        """Generate single excitation indices."""
        occupied = list(range(self.n_electrons))
        virtual = list(range(self.n_electrons, self.n_qubits))

        singles = []
        for i in occupied:
            for a in virtual:
                singles.append((i, a))

        return singles

    def _generate_doubles(self) -> list[tuple[int, int, int, int]]:
        """Generate double excitation indices."""
        occupied = list(range(self.n_electrons))
        virtual = list(range(self.n_electrons, self.n_qubits))

        doubles = []
        for i in occupied:
            for j in occupied:
                if i < j:
                    for a in virtual:
                        for b in virtual:
                            if a < b:
                                doubles.append((i, j, a, b))

        return doubles

    def build(self, backend: Any) -> Any:
        """Build UCCSD circuit."""
        circuit = backend.create_circuit(self.n_qubits, self.name)

        # Initial Hartree-Fock state
        for i in range(self.n_electrons):
            backend.add_x_gate(circuit, i)

        param_idx = 0

        # Singles excitations
        if self.include_singles:
            for p, q in self._singles:
                theta = self._parameters[param_idx].value
                self._add_single_excitation(backend, circuit, p, q, theta)
                param_idx += 1

        # Doubles excitations
        if self.include_doubles:
            for p, q, r, s in self._doubles:
                theta = self._parameters[param_idx].value
                self._add_double_excitation(backend, circuit, p, q, r, s, theta)
                param_idx += 1

        return circuit

    def _add_single_excitation(
        self,
        backend: Any,
        circuit: Any,
        p: int,
        q: int,
        theta: float,
    ) -> None:
        """Add single excitation operator."""
        # Implement using Jordan-Wigner transformation
        # a†_p a_q - a†_q a_p

        # Simplified implementation using Givens rotation
        backend.add_ry_gate(circuit, p, theta / 2)
        backend.add_cx_gate(circuit, p, q)
        backend.add_ry_gate(circuit, p, -theta / 2)
        backend.add_cx_gate(circuit, p, q)

    def _add_double_excitation(
        self,
        backend: Any,
        circuit: Any,
        p: int,
        q: int,
        r: int,
        s: int,
        theta: float,
    ) -> None:
        """Add double excitation operator."""
        # Simplified implementation
        # Full UCCSD requires more complex decomposition

        backend.add_cx_gate(circuit, p, q)
        backend.add_cx_gate(circuit, r, s)
        backend.add_ry_gate(circuit, q, theta / 4)
        backend.add_cx_gate(circuit, s, q)
        backend.add_ry_gate(circuit, q, -theta / 4)
        backend.add_cx_gate(circuit, s, q)
        backend.add_cx_gate(circuit, r, s)
        backend.add_cx_gate(circuit, p, q)

    def get_gradient_circuits(self) -> list[tuple[Any, float]]:
        """Get gradient circuits using parameter shift rule."""
        gradients = []

        for i, param in enumerate(self._parameters):
            if not param.trainable:
                continue

            plus_params = self.parameter_values.copy()
            plus_params[i] += np.pi / 2

            minus_params = self.parameter_values.copy()
            minus_params[i] -= np.pi / 2

            gradients.append({
                "parameter": param.name,
                "plus_shift": plus_params,
                "minus_shift": minus_params,
                "coefficient": 0.5,
            })

        return gradients


@dataclass
class VariationalLayer:
    """A layer in a variational circuit."""

    layer_type: str  # "rotation", "entanglement", "pooling"
    qubits: list[int] = field(default_factory=list)
    gates: list[str] = field(default_factory=list)
    parameters: list[float] = field(default_factory=list)


class CircuitBuilder:
    """
    Fluent builder for variational circuits.

    Example:
        >>> builder = CircuitBuilder(4)
        >>> circuit = (builder
        ...     .add_layer("rotation", gates=["ry", "rz"])
        ...     .add_layer("entanglement", pattern="linear")
        ...     .repeat(3)
        ...     .build())
    """

    def __init__(self, n_qubits: int):
        """Initialize builder."""
        self.n_qubits = n_qubits
        self.layers: list[VariationalLayer] = []
        self._repeat_count = 1

    def add_rotation_layer(
        self,
        gates: list[str] | None = None,
        qubits: list[int] | None = None,
    ) -> "CircuitBuilder":
        """Add a rotation layer."""
        gates = gates or ["ry", "rz"]
        qubits = qubits or list(range(self.n_qubits))

        self.layers.append(VariationalLayer(
            layer_type="rotation",
            qubits=qubits,
            gates=gates,
        ))
        return self

    def add_entanglement_layer(
        self,
        pattern: str = "linear",
        gate: str = "cx",
    ) -> "CircuitBuilder":
        """Add an entanglement layer."""
        if pattern == "linear":
            qubits = [(i, i + 1) for i in range(self.n_qubits - 1)]
        elif pattern == "circular":
            qubits = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        elif pattern == "full":
            qubits = [
                (i, j)
                for i in range(self.n_qubits)
                for j in range(i + 1, self.n_qubits)
            ]
        else:
            qubits = []

        self.layers.append(VariationalLayer(
            layer_type="entanglement",
            qubits=qubits,
            gates=[gate],
        ))
        return self

    def repeat(self, count: int) -> "CircuitBuilder":
        """Repeat the current layer structure."""
        self._repeat_count = count
        return self

    def build(self) -> list[VariationalLayer]:
        """Build and return the layer structure."""
        result = []
        for _ in range(self._repeat_count):
            result.extend(self.layers)
        return result
