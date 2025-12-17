"""
Simulator Backend for QBitaLabs

Provides a pure NumPy-based quantum simulator for testing
and development without external quantum computing libraries.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import numpy as np
import structlog

from qbitalabs.quantum.backends.base_backend import (
    BackendType,
    BaseQuantumBackend,
    CircuitResult,
    ExecutionMode,
    QuantumJob,
)

logger = structlog.get_logger(__name__)


class SimulatorBackend(BaseQuantumBackend):
    """
    Pure NumPy quantum simulator backend.

    Features:
    - No external dependencies (beyond NumPy)
    - State vector simulation
    - Density matrix simulation for noise
    - Educational and testing purposes

    Example:
        >>> backend = SimulatorBackend()
        >>> await backend.initialize()
        >>> circuit = backend.create_circuit(4)
        >>> circuit.h(0).cx(0, 1)
        >>> result = await backend.execute(circuit)
    """

    # Standard quantum gates
    GATES = {
        "I": np.array([[1, 0], [0, 1]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        "S": np.array([[1, 0], [0, 1j]], dtype=complex),
        "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        "CNOT": np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex),
        "CZ": np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ], dtype=complex),
        "SWAP": np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex),
    }

    def __init__(
        self,
        shots: int = 1024,
        seed: int | None = None,
        use_density_matrix: bool = False,
    ):
        """
        Initialize simulator backend.

        Args:
            shots: Measurement shots.
            seed: Random seed for reproducibility.
            use_density_matrix: Use density matrix simulation.
        """
        super().__init__(
            backend_type=BackendType.SIMULATOR,
            execution_mode=ExecutionMode.SIMULATOR,
            shots=shots,
            seed=seed,
        )

        self.use_density_matrix = use_density_matrix
        self._rng = np.random.default_rng(seed)

    async def initialize(self) -> None:
        """Initialize simulator."""
        self._initialized = True
        self._logger.info(
            "Simulator backend initialized",
            shots=self.shots,
            density_matrix=self.use_density_matrix,
        )

    async def shutdown(self) -> None:
        """Shutdown simulator."""
        self._initialized = False
        self._logger.info("Simulator backend shutdown")

    def create_circuit(self, n_qubits: int, name: str = "circuit") -> "SimulatorCircuit":
        """Create a simulator circuit."""
        return SimulatorCircuit(n_qubits, name)

    async def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        **kwargs: Any,
    ) -> CircuitResult:
        """Execute a quantum circuit."""
        if not self._initialized:
            await self.initialize()

        shots = shots or self.shots
        job_id = str(uuid4())[:8]

        job = QuantumJob(
            job_id=job_id,
            circuit_name=getattr(circuit, "name", "simulator_circuit"),
            backend_type=self.backend_type,
            shots=shots,
        )

        try:
            import time

            start_time = time.time()

            if self.use_density_matrix:
                counts = self._simulate_density_matrix(circuit, shots)
            else:
                counts = self._simulate_state_vector(circuit, shots)

            execution_time = time.time() - start_time

            job.status = "completed"
            job.results = {"counts": counts}

            probabilities = {k: v / shots for k, v in counts.items()}

            circuit_result = CircuitResult(
                counts=counts,
                probabilities=probabilities,
                execution_time=execution_time,
                shots=shots,
                backend="qbitalabs_simulator",
            )

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self._logger.error("Circuit execution failed", error=str(e))
            raise

        finally:
            self._record_job(job)

        return circuit_result

    def _simulate_state_vector(
        self, circuit: "SimulatorCircuit", shots: int
    ) -> dict[str, int]:
        """Simulate using state vector."""
        n_qubits = circuit.n_qubits

        # Initialize |0...0>
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0

        # Apply gates
        for gate in circuit.gates:
            state = self._apply_gate_to_state(state, gate, n_qubits)

        # Compute probabilities
        probabilities = np.abs(state) ** 2
        probabilities /= probabilities.sum()

        # Sample measurements
        outcomes = self._rng.choice(
            2**n_qubits,
            size=shots,
            p=probabilities,
        )

        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{n_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def _simulate_density_matrix(
        self, circuit: "SimulatorCircuit", shots: int
    ) -> dict[str, int]:
        """Simulate using density matrix (supports noise)."""
        n_qubits = circuit.n_qubits
        dim = 2**n_qubits

        # Initialize |0><0|
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0

        # Apply gates as superoperators
        for gate in circuit.gates:
            rho = self._apply_gate_to_density_matrix(rho, gate, n_qubits)

        # Extract probabilities from diagonal
        probabilities = np.real(np.diag(rho))
        probabilities = np.maximum(probabilities, 0)  # Fix numerical errors
        probabilities /= probabilities.sum()

        # Sample measurements
        outcomes = self._rng.choice(
            dim,
            size=shots,
            p=probabilities,
        )

        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{n_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def _apply_gate_to_state(
        self,
        state: np.ndarray,
        gate: dict[str, Any],
        n_qubits: int,
    ) -> np.ndarray:
        """Apply gate to state vector."""
        gate_name = gate["gate"]
        targets = gate.get("targets", [gate.get("target", 0)])
        params = gate.get("params", {})

        if gate_name in ["H", "X", "Y", "Z", "S", "T", "I"]:
            matrix = self.GATES[gate_name]
            target = targets[0] if isinstance(targets, list) else targets
            return self._apply_single_qubit(state, matrix, target, n_qubits)

        elif gate_name in ["RX", "RY", "RZ"]:
            theta = params.get("theta", 0)
            matrix = self._rotation_gate(gate_name, theta)
            target = targets[0] if isinstance(targets, list) else targets
            return self._apply_single_qubit(state, matrix, target, n_qubits)

        elif gate_name in ["CNOT", "CX"]:
            return self._apply_cnot(state, targets, n_qubits)

        elif gate_name == "CZ":
            return self._apply_cz(state, targets, n_qubits)

        elif gate_name == "SWAP":
            return self._apply_swap(state, targets, n_qubits)

        return state

    def _apply_gate_to_density_matrix(
        self,
        rho: np.ndarray,
        gate: dict[str, Any],
        n_qubits: int,
    ) -> np.ndarray:
        """Apply gate to density matrix: rho -> U @ rho @ U†"""
        dim = 2**n_qubits
        gate_name = gate["gate"]
        targets = gate.get("targets", [gate.get("target", 0)])
        params = gate.get("params", {})

        # Build full unitary
        if gate_name in ["H", "X", "Y", "Z", "S", "T", "I"]:
            matrix = self.GATES[gate_name]
            target = targets[0] if isinstance(targets, list) else targets
            U = self._embed_single_qubit(matrix, target, n_qubits)
        elif gate_name in ["RX", "RY", "RZ"]:
            theta = params.get("theta", 0)
            matrix = self._rotation_gate(gate_name, theta)
            target = targets[0] if isinstance(targets, list) else targets
            U = self._embed_single_qubit(matrix, target, n_qubits)
        elif gate_name in ["CNOT", "CX"]:
            U = self._embed_two_qubit(self.GATES["CNOT"], targets, n_qubits)
        elif gate_name == "CZ":
            U = self._embed_two_qubit(self.GATES["CZ"], targets, n_qubits)
        else:
            U = np.eye(dim, dtype=complex)

        return U @ rho @ U.conj().T

    def _rotation_gate(self, gate_name: str, theta: float) -> np.ndarray:
        """Create rotation gate matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        if gate_name == "RX":
            return np.array([
                [c, -1j * s],
                [-1j * s, c],
            ], dtype=complex)
        elif gate_name == "RY":
            return np.array([
                [c, -s],
                [s, c],
            ], dtype=complex)
        elif gate_name == "RZ":
            return np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ], dtype=complex)

        return np.eye(2, dtype=complex)

    def _apply_single_qubit(
        self,
        state: np.ndarray,
        matrix: np.ndarray,
        target: int,
        n_qubits: int,
    ) -> np.ndarray:
        """Apply single-qubit gate using tensor contraction."""
        # Reshape state to tensor
        shape = [2] * n_qubits
        tensor = state.reshape(shape)

        # Apply gate along target axis
        tensor = np.tensordot(matrix, tensor, axes=([1], [target]))

        # Move axis back
        tensor = np.moveaxis(tensor, 0, target)

        return tensor.reshape(-1)

    def _embed_single_qubit(
        self,
        matrix: np.ndarray,
        target: int,
        n_qubits: int,
    ) -> np.ndarray:
        """Embed single-qubit gate in full Hilbert space."""
        result = np.array([[1]], dtype=complex)

        for i in range(n_qubits):
            if i == target:
                result = np.kron(result, matrix)
            else:
                result = np.kron(result, self.GATES["I"])

        return result

    def _embed_two_qubit(
        self,
        matrix: np.ndarray,
        targets: list[int],
        n_qubits: int,
    ) -> np.ndarray:
        """Embed two-qubit gate in full Hilbert space."""
        dim = 2**n_qubits
        q0, q1 = targets[0], targets[1]

        # Build permutation
        U = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            b0 = (i >> (n_qubits - 1 - q0)) & 1
            b1 = (i >> (n_qubits - 1 - q1)) & 1
            idx = 2 * b0 + b1

            for j in range(4):
                new_b0 = (j >> 1) & 1
                new_b1 = j & 1

                new_i = i
                new_i = (new_i & ~(1 << (n_qubits - 1 - q0))) | (new_b0 << (n_qubits - 1 - q0))
                new_i = (new_i & ~(1 << (n_qubits - 1 - q1))) | (new_b1 << (n_qubits - 1 - q1))

                U[new_i, i] += matrix[j, idx]

        return U

    def _apply_cnot(
        self,
        state: np.ndarray,
        targets: list[int],
        n_qubits: int,
    ) -> np.ndarray:
        """Apply CNOT gate."""
        control, target = targets[0], targets[1]
        new_state = state.copy()

        for i in range(2**n_qubits):
            control_bit = (i >> (n_qubits - 1 - control)) & 1
            if control_bit == 1:
                partner = i ^ (1 << (n_qubits - 1 - target))
                new_state[i], new_state[partner] = state[partner], state[i]

        return new_state

    def _apply_cz(
        self,
        state: np.ndarray,
        targets: list[int],
        n_qubits: int,
    ) -> np.ndarray:
        """Apply CZ gate."""
        q0, q1 = targets[0], targets[1]
        new_state = state.copy()

        for i in range(2**n_qubits):
            b0 = (i >> (n_qubits - 1 - q0)) & 1
            b1 = (i >> (n_qubits - 1 - q1)) & 1
            if b0 == 1 and b1 == 1:
                new_state[i] *= -1

        return new_state

    def _apply_swap(
        self,
        state: np.ndarray,
        targets: list[int],
        n_qubits: int,
    ) -> np.ndarray:
        """Apply SWAP gate."""
        q0, q1 = targets[0], targets[1]
        new_state = state.copy()

        for i in range(2**n_qubits):
            b0 = (i >> (n_qubits - 1 - q0)) & 1
            b1 = (i >> (n_qubits - 1 - q1)) & 1

            if b0 != b1:
                # Swap the bits
                partner = i ^ (1 << (n_qubits - 1 - q0)) ^ (1 << (n_qubits - 1 - q1))
                new_state[i], new_state[partner] = state[partner], state[i]

        return new_state

    async def run_vqe(
        self,
        hamiltonian: Any,
        ansatz: Any,
        optimizer: str = "COBYLA",
        initial_params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run VQE using simulator."""
        from scipy.optimize import minimize

        if initial_params is None:
            initial_params = self._rng.uniform(-np.pi, np.pi, 10)

        def cost_function(params: np.ndarray) -> float:
            circuit = ansatz(params)
            result = self._simulate_state_vector(circuit, 1)

            # Simple energy calculation
            energy = 0.0
            for bitstring, count in result.items():
                energy += sum(1 if b == "1" else -1 for b in bitstring)

            return energy

        result = minimize(
            cost_function,
            initial_params,
            method=optimizer,
            options={"maxiter": 500},
        )

        return {
            "eigenvalue": float(result.fun),
            "optimal_params": result.x.tolist(),
            "success": result.success,
            "nfev": result.nfev,
        }

    async def run_qaoa(
        self,
        cost_hamiltonian: Any,
        p: int = 1,
        initial_params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run QAOA using simulator."""
        if initial_params is None:
            initial_params = self._rng.uniform(-np.pi, np.pi, 2 * p)

        def qaoa_ansatz(params: np.ndarray) -> "SimulatorCircuit":
            circuit = SimulatorCircuit(4, f"qaoa_p{p}")

            # Initial superposition
            for q in range(4):
                circuit.h(q)

            # QAOA layers
            for layer in range(p):
                gamma = params[layer]
                beta = params[p + layer]

                # Cost layer (ZZ interactions)
                for i in range(3):
                    circuit.cnot(i, i + 1)
                    circuit.rz(gamma, i + 1)
                    circuit.cnot(i, i + 1)

                # Mixer layer
                for q in range(4):
                    circuit.rx(2 * beta, q)

            return circuit

        return await self.run_vqe(
            cost_hamiltonian,
            qaoa_ansatz,
            initial_params=initial_params,
        )

    def get_state_vector(self, circuit: "SimulatorCircuit") -> np.ndarray:
        """Get the final state vector without measurement."""
        n_qubits = circuit.n_qubits
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0

        for gate in circuit.gates:
            state = self._apply_gate_to_state(state, gate, n_qubits)

        return state

    def get_density_matrix(self, circuit: "SimulatorCircuit") -> np.ndarray:
        """Get the final density matrix."""
        state = self.get_state_vector(circuit)
        return np.outer(state, state.conj())


class SimulatorCircuit:
    """Simple circuit representation for the simulator."""

    def __init__(self, n_qubits: int, name: str = "circuit"):
        """Initialize circuit."""
        self.n_qubits = n_qubits
        self.name = name
        self.gates: list[dict[str, Any]] = []

    def h(self, target: int) -> "SimulatorCircuit":
        """Add Hadamard gate."""
        self.gates.append({"gate": "H", "target": target})
        return self

    def x(self, target: int) -> "SimulatorCircuit":
        """Add Pauli-X gate."""
        self.gates.append({"gate": "X", "target": target})
        return self

    def y(self, target: int) -> "SimulatorCircuit":
        """Add Pauli-Y gate."""
        self.gates.append({"gate": "Y", "target": target})
        return self

    def z(self, target: int) -> "SimulatorCircuit":
        """Add Pauli-Z gate."""
        self.gates.append({"gate": "Z", "target": target})
        return self

    def s(self, target: int) -> "SimulatorCircuit":
        """Add S gate."""
        self.gates.append({"gate": "S", "target": target})
        return self

    def t(self, target: int) -> "SimulatorCircuit":
        """Add T gate."""
        self.gates.append({"gate": "T", "target": target})
        return self

    def rx(self, theta: float, target: int) -> "SimulatorCircuit":
        """Add RX rotation."""
        self.gates.append({
            "gate": "RX",
            "target": target,
            "params": {"theta": theta},
        })
        return self

    def ry(self, theta: float, target: int) -> "SimulatorCircuit":
        """Add RY rotation."""
        self.gates.append({
            "gate": "RY",
            "target": target,
            "params": {"theta": theta},
        })
        return self

    def rz(self, theta: float, target: int) -> "SimulatorCircuit":
        """Add RZ rotation."""
        self.gates.append({
            "gate": "RZ",
            "target": target,
            "params": {"theta": theta},
        })
        return self

    def cnot(self, control: int, target: int) -> "SimulatorCircuit":
        """Add CNOT gate."""
        self.gates.append({
            "gate": "CNOT",
            "targets": [control, target],
        })
        return self

    def cx(self, control: int, target: int) -> "SimulatorCircuit":
        """Add CX (CNOT) gate."""
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> "SimulatorCircuit":
        """Add CZ gate."""
        self.gates.append({
            "gate": "CZ",
            "targets": [control, target],
        })
        return self

    def swap(self, q0: int, q1: int) -> "SimulatorCircuit":
        """Add SWAP gate."""
        self.gates.append({
            "gate": "SWAP",
            "targets": [q0, q1],
        })
        return self

    def measure_all(self) -> "SimulatorCircuit":
        """Mark circuit for measurement (implicit in execute)."""
        return self

    def __repr__(self) -> str:
        """String representation."""
        return f"SimulatorCircuit({self.name}, {self.n_qubits} qubits, {len(self.gates)} gates)"
