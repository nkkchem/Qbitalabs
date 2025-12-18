"""
PennyLane Backend for QBitaLabs

Provides integration with Xanadu's PennyLane quantum ML framework.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable
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

# Lazy imports
pennylane = None
qml = None


def _import_pennylane() -> None:
    """Lazily import PennyLane."""
    global pennylane, qml
    if pennylane is None:
        try:
            import pennylane as _pennylane

            pennylane = _pennylane
            qml = _pennylane
        except ImportError as e:
            raise ImportError(
                "PennyLane not installed. Install with: pip install pennylane"
            ) from e


class PennyLaneBackend(BaseQuantumBackend):
    """
    PennyLane-based quantum computing backend.

    Supports:
    - Automatic differentiation for quantum ML
    - Multiple device backends
    - Hybrid quantum-classical optimization
    - Integration with PyTorch/TensorFlow

    Example:
        >>> backend = PennyLaneBackend()
        >>> await backend.initialize()
        >>> @backend.qnode
        ... def circuit(params):
        ...     qml.RY(params[0], wires=0)
        ...     return qml.expval(qml.PauliZ(0))
    """

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.SIMULATOR,
        shots: int = 1024,
        seed: int | None = None,
        device_name: str = "default.qubit",
        interface: str = "numpy",
    ):
        """
        Initialize PennyLane backend.

        Args:
            execution_mode: Simulator or hardware.
            shots: Measurement shots.
            seed: Random seed.
            device_name: PennyLane device name.
            interface: ML framework interface (numpy, torch, tf, jax).
        """
        super().__init__(
            backend_type=BackendType.PENNYLANE,
            execution_mode=execution_mode,
            shots=shots,
            seed=seed,
        )

        self.device_name = device_name
        self.interface = interface
        self._device = None
        self._n_qubits = 4

    async def initialize(self) -> None:
        """Initialize PennyLane backend."""
        _import_pennylane()

        # Create device
        device_kwargs = {"wires": self._n_qubits}

        if self.shots:
            device_kwargs["shots"] = self.shots

        if self.device_name in ["default.qubit", "lightning.qubit"]:
            self._device = qml.device(self.device_name, **device_kwargs)
        else:
            # Try to create the requested device
            try:
                self._device = qml.device(self.device_name, **device_kwargs)
            except Exception as e:
                self._logger.warning(
                    f"Failed to create device {self.device_name}, "
                    "falling back to default.qubit",
                    error=str(e),
                )
                self._device = qml.device("default.qubit", **device_kwargs)

        self._initialized = True
        self._logger.info(
            "PennyLane backend initialized",
            device=self.device_name,
            interface=self.interface,
        )

    async def shutdown(self) -> None:
        """Shutdown PennyLane backend."""
        self._device = None
        self._initialized = False
        self._logger.info("PennyLane backend shutdown")

    def set_n_qubits(self, n_qubits: int) -> None:
        """Set number of qubits and recreate device."""
        self._n_qubits = n_qubits
        if self._initialized:
            _import_pennylane()
            device_kwargs = {"wires": n_qubits}
            if self.shots:
                device_kwargs["shots"] = self.shots
            self._device = qml.device(self.device_name, **device_kwargs)

    def create_circuit(self, n_qubits: int, name: str = "circuit") -> Any:
        """Create a PennyLane circuit wrapper."""
        _import_pennylane()

        self.set_n_qubits(n_qubits)

        # Return a circuit builder class
        return PennyLaneCircuitBuilder(n_qubits, name)

    def qnode(
        self,
        func: Callable,
        diff_method: str = "best",
    ) -> Callable:
        """
        Create a QNode from a quantum function.

        Args:
            func: Quantum function.
            diff_method: Differentiation method.

        Returns:
            QNode callable.
        """
        _import_pennylane()

        return qml.QNode(
            func,
            self._device,
            interface=self.interface,
            diff_method=diff_method,
        )

    async def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        **kwargs: Any,
    ) -> CircuitResult:
        """Execute a quantum circuit."""
        if not self._initialized:
            await self.initialize()

        _import_pennylane()

        shots = shots or self.shots
        job_id = str(uuid4())[:8]

        job = QuantumJob(
            job_id=job_id,
            circuit_name=getattr(circuit, "name", "pennylane_circuit"),
            backend_type=self.backend_type,
            shots=shots,
        )

        try:
            import time

            start_time = time.time()

            # Build and execute circuit
            if isinstance(circuit, PennyLaneCircuitBuilder):
                results = circuit.execute(self._device, shots)
            else:
                # Assume it's already a QNode or callable
                results = circuit()

            execution_time = time.time() - start_time

            # Convert to counts format
            if isinstance(results, np.ndarray):
                # Probabilistic result
                counts = {
                    format(i, f"0{self._n_qubits}b"): int(p * shots)
                    for i, p in enumerate(results)
                    if p > 0
                }
            else:
                counts = {"0" * self._n_qubits: shots}

            probabilities = {k: v / shots for k, v in counts.items()}

            job.status = "completed"
            job.results = {"counts": counts}

            circuit_result = CircuitResult(
                counts=counts,
                probabilities=probabilities,
                execution_time=execution_time,
                shots=shots,
                backend=self.device_name,
                raw_data=results,
            )

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self._logger.error("Circuit execution failed", error=str(e))
            raise

        finally:
            self._record_job(job)

        return circuit_result

    async def run_vqe(
        self,
        hamiltonian: Any,
        ansatz: Any,
        optimizer: str = "GradientDescent",
        initial_params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run VQE using PennyLane's differentiable framework."""
        if not self._initialized:
            await self.initialize()

        _import_pennylane()

        # Create cost function
        @qml.qnode(self._device, interface=self.interface)
        def cost_fn(params):
            ansatz(params)
            return qml.expval(hamiltonian)

        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, 10)

        params = initial_params.copy()

        # Select optimizer
        if optimizer == "GradientDescent":
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
        elif optimizer == "Adam":
            opt = qml.AdamOptimizer(stepsize=0.1)
        elif optimizer == "QNG":
            opt = qml.QNGOptimizer(stepsize=0.1)
        else:
            opt = qml.GradientDescentOptimizer(stepsize=0.1)

        # Optimization loop
        history = []
        for step in range(200):
            params, cost = opt.step_and_cost(cost_fn, params)
            history.append(float(cost))

            if step > 10 and abs(history[-1] - history[-2]) < 1e-6:
                break

        return {
            "eigenvalue": float(cost_fn(params)),
            "optimal_params": params.tolist(),
            "iterations": len(history),
            "history": history,
        }

    async def run_qaoa(
        self,
        cost_hamiltonian: Any,
        p: int = 1,
        initial_params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run QAOA using PennyLane."""
        if not self._initialized:
            await self.initialize()

        _import_pennylane()

        n_qubits = self._n_qubits

        # Build QAOA circuit
        def qaoa_layer(gamma, beta):
            # Cost layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(gamma, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

            # Mixer layer
            for i in range(n_qubits):
                qml.RX(2 * beta, wires=i)

        @qml.qnode(self._device, interface=self.interface)
        def qaoa_circuit(params):
            # Initialize in superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            # QAOA layers
            for layer in range(p):
                qaoa_layer(params[layer], params[p + layer])

            return qml.expval(cost_hamiltonian)

        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, 2 * p)

        params = initial_params.copy()

        # Optimize
        opt = qml.AdamOptimizer(stepsize=0.1)
        history = []

        for step in range(200):
            params, cost = opt.step_and_cost(qaoa_circuit, params)
            history.append(float(cost))

            if step > 10 and abs(history[-1] - history[-2]) < 1e-6:
                break

        return {
            "eigenvalue": float(qaoa_circuit(params)),
            "optimal_params": params.tolist(),
            "gamma": params[:p].tolist(),
            "beta": params[p:].tolist(),
            "iterations": len(history),
            "history": history,
        }

    def create_variational_ansatz(
        self,
        n_qubits: int,
        depth: int = 2,
    ) -> Callable:
        """Create a variational ansatz function."""
        _import_pennylane()

        def ansatz(params):
            param_idx = 0
            for layer in range(depth):
                for i in range(n_qubits):
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        return ansatz

    def create_hamiltonian(
        self,
        coeffs: list[float],
        observables: list[str],
    ) -> Any:
        """
        Create a Hamiltonian from coefficients and Pauli strings.

        Args:
            coeffs: Coefficients for each term.
            observables: Pauli strings (e.g., ["ZZ", "XI", "IX"]).

        Returns:
            PennyLane Hamiltonian.
        """
        _import_pennylane()

        ops = []
        for obs_str in observables:
            pauli_ops = []
            for i, p in enumerate(obs_str):
                if p == "X":
                    pauli_ops.append(qml.PauliX(i))
                elif p == "Y":
                    pauli_ops.append(qml.PauliY(i))
                elif p == "Z":
                    pauli_ops.append(qml.PauliZ(i))
                elif p == "I":
                    pauli_ops.append(qml.Identity(i))

            if len(pauli_ops) == 1:
                ops.append(pauli_ops[0])
            else:
                ops.append(pauli_ops[0] @ pauli_ops[1])

        return qml.Hamiltonian(coeffs, ops)

    def quantum_natural_gradient(
        self,
        qnode: Callable,
        params: np.ndarray,
    ) -> np.ndarray:
        """Compute quantum natural gradient."""
        _import_pennylane()

        metric_tensor = qml.metric_tensor(qnode)(params)
        grad = qml.grad(qnode)(params)

        # Solve Fubini-Study metric equation
        qng = np.linalg.solve(
            metric_tensor + 0.01 * np.eye(len(params)),
            grad,
        )

        return qng

    def _get_supported_gates(self) -> list[str]:
        """Get PennyLane-specific supported gates."""
        return [
            "Hadamard", "PauliX", "PauliY", "PauliZ",
            "CNOT", "CZ", "SWAP", "Toffoli",
            "RX", "RY", "RZ", "Rot",
            "U1", "U2", "U3",
            "QubitUnitary", "ControlledPhaseShift",
        ]


class PennyLaneCircuitBuilder:
    """Helper class for building PennyLane circuits."""

    def __init__(self, n_qubits: int, name: str = "circuit"):
        """Initialize circuit builder."""
        self.n_qubits = n_qubits
        self.name = name
        self.operations: list[tuple[str, list[Any], dict[str, Any]]] = []

    def h(self, wire: int) -> "PennyLaneCircuitBuilder":
        """Add Hadamard gate."""
        self.operations.append(("Hadamard", [wire], {}))
        return self

    def x(self, wire: int) -> "PennyLaneCircuitBuilder":
        """Add Pauli-X gate."""
        self.operations.append(("PauliX", [wire], {}))
        return self

    def y(self, wire: int) -> "PennyLaneCircuitBuilder":
        """Add Pauli-Y gate."""
        self.operations.append(("PauliY", [wire], {}))
        return self

    def z(self, wire: int) -> "PennyLaneCircuitBuilder":
        """Add Pauli-Z gate."""
        self.operations.append(("PauliZ", [wire], {}))
        return self

    def cnot(self, control: int, target: int) -> "PennyLaneCircuitBuilder":
        """Add CNOT gate."""
        self.operations.append(("CNOT", [[control, target]], {}))
        return self

    def rx(self, theta: float, wire: int) -> "PennyLaneCircuitBuilder":
        """Add RX gate."""
        self.operations.append(("RX", [theta, wire], {}))
        return self

    def ry(self, theta: float, wire: int) -> "PennyLaneCircuitBuilder":
        """Add RY gate."""
        self.operations.append(("RY", [theta, wire], {}))
        return self

    def rz(self, theta: float, wire: int) -> "PennyLaneCircuitBuilder":
        """Add RZ gate."""
        self.operations.append(("RZ", [theta, wire], {}))
        return self

    def execute(self, device: Any, shots: int = 1024) -> np.ndarray:
        """Execute the built circuit."""
        _import_pennylane()

        @qml.qnode(device)
        def circuit():
            for op_name, args, kwargs in self.operations:
                gate = getattr(qml, op_name)
                if op_name in ["RX", "RY", "RZ"]:
                    gate(args[0], wires=args[1])
                elif op_name == "CNOT":
                    gate(wires=args[0])
                else:
                    gate(wires=args[0])
            return qml.probs(wires=range(self.n_qubits))

        return circuit()
