"""
QAOA Circuits for QBitaLabs

Implements Quantum Approximate Optimization Algorithm circuits
for combinatorial optimization problems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from qbitalabs.quantum.circuits.variational import VariationalCircuit

logger = structlog.get_logger(__name__)


class QAOACircuit(VariationalCircuit):
    """
    Quantum Approximate Optimization Algorithm circuit.

    QAOA uses alternating cost and mixer layers:
    1. Initialize in |+⟩ superposition
    2. Apply cost unitary: exp(-i γ C)
    3. Apply mixer unitary: exp(-i β B)
    4. Repeat p times

    Example:
        >>> qaoa = QAOACircuit(n_qubits=4, p=2)
        >>> qaoa.set_cost_hamiltonian(hamiltonian)
        >>> circuit = qaoa.build(backend)
    """

    def __init__(
        self,
        n_qubits: int,
        p: int = 1,
        mixer_type: str = "x",  # "x", "xy", "grover"
    ):
        """
        Initialize QAOA circuit.

        Args:
            n_qubits: Number of qubits.
            p: Number of QAOA layers.
            mixer_type: Type of mixer Hamiltonian.
        """
        super().__init__(n_qubits, f"qaoa_p{p}")

        self.p = p
        self.mixer_type = mixer_type

        # Cost Hamiltonian coefficients
        self._cost_terms: list[tuple[list[int], float]] = []

        # Create gamma and beta parameters
        for layer in range(p):
            self.add_parameter(f"gamma_{layer}")
            self.add_parameter(f"beta_{layer}")

    @property
    def gammas(self) -> np.ndarray:
        """Get gamma parameters."""
        return np.array([
            self._parameters[2 * i].value
            for i in range(self.p)
        ])

    @property
    def betas(self) -> np.ndarray:
        """Get beta parameters."""
        return np.array([
            self._parameters[2 * i + 1].value
            for i in range(self.p)
        ])

    def set_cost_hamiltonian(
        self,
        terms: list[tuple[list[int], float]],
    ) -> None:
        """
        Set the cost Hamiltonian.

        Args:
            terms: List of (qubits, coefficient) tuples.
                   e.g., [([0, 1], 1.0), ([1, 2], 0.5)] for Z0Z1 + 0.5*Z1Z2
        """
        self._cost_terms = terms

    def build(self, backend: Any) -> Any:
        """Build QAOA circuit."""
        circuit = backend.create_circuit(self.n_qubits, self.name)

        # Initial superposition
        for q in range(self.n_qubits):
            backend.add_h_gate(circuit, q)

        # QAOA layers
        for layer in range(self.p):
            gamma = self._parameters[2 * layer].value
            beta = self._parameters[2 * layer + 1].value

            # Cost layer
            self._add_cost_layer(backend, circuit, gamma)

            # Mixer layer
            self._add_mixer_layer(backend, circuit, beta)

        return circuit

    def _add_cost_layer(
        self,
        backend: Any,
        circuit: Any,
        gamma: float,
    ) -> None:
        """Add cost Hamiltonian layer."""
        for qubits, coeff in self._cost_terms:
            if len(qubits) == 1:
                # Single-qubit Z term
                backend.add_rz_gate(circuit, qubits[0], 2 * gamma * coeff)

            elif len(qubits) == 2:
                # Two-qubit ZZ term
                q0, q1 = qubits
                backend.add_cx_gate(circuit, q0, q1)
                backend.add_rz_gate(circuit, q1, 2 * gamma * coeff)
                backend.add_cx_gate(circuit, q0, q1)

            else:
                # Multi-qubit Z terms require more complex decomposition
                self._add_multiqubit_z_rotation(
                    backend, circuit, qubits, 2 * gamma * coeff
                )

    def _add_mixer_layer(
        self,
        backend: Any,
        circuit: Any,
        beta: float,
    ) -> None:
        """Add mixer Hamiltonian layer."""
        if self.mixer_type == "x":
            # Standard X mixer
            for q in range(self.n_qubits):
                backend.add_rx_gate(circuit, q, 2 * beta)

        elif self.mixer_type == "xy":
            # XY mixer (preserves Hamming weight)
            for i in range(self.n_qubits - 1):
                self._add_xy_mixer(backend, circuit, i, i + 1, beta)

        elif self.mixer_type == "grover":
            # Grover mixer
            self._add_grover_mixer(backend, circuit, beta)

    def _add_multiqubit_z_rotation(
        self,
        backend: Any,
        circuit: Any,
        qubits: list[int],
        angle: float,
    ) -> None:
        """Add rotation for multi-qubit Z term."""
        # Use CNOT ladder
        for i in range(len(qubits) - 1):
            backend.add_cx_gate(circuit, qubits[i], qubits[i + 1])

        backend.add_rz_gate(circuit, qubits[-1], angle)

        # Uncompute
        for i in range(len(qubits) - 2, -1, -1):
            backend.add_cx_gate(circuit, qubits[i], qubits[i + 1])

    def _add_xy_mixer(
        self,
        backend: Any,
        circuit: Any,
        q0: int,
        q1: int,
        beta: float,
    ) -> None:
        """Add XY mixer between two qubits."""
        # exp(-i β (XX + YY))
        backend.add_h_gate(circuit, q0)
        backend.add_h_gate(circuit, q1)
        backend.add_cx_gate(circuit, q0, q1)
        backend.add_rz_gate(circuit, q1, 2 * beta)
        backend.add_cx_gate(circuit, q0, q1)
        backend.add_h_gate(circuit, q0)
        backend.add_h_gate(circuit, q1)

    def _add_grover_mixer(
        self,
        backend: Any,
        circuit: Any,
        beta: float,
    ) -> None:
        """Add Grover diffusion operator as mixer."""
        # H⊗n (2|0><0| - I) H⊗n
        for q in range(self.n_qubits):
            backend.add_h_gate(circuit, q)

        for q in range(self.n_qubits):
            backend.add_x_gate(circuit, q)

        # Multi-controlled Z
        if self.n_qubits > 1:
            backend.add_h_gate(circuit, self.n_qubits - 1)
            # Simplified: in practice need multi-controlled gate
            backend.add_cx_gate(circuit, 0, self.n_qubits - 1)
            backend.add_h_gate(circuit, self.n_qubits - 1)

        for q in range(self.n_qubits):
            backend.add_x_gate(circuit, q)

        for q in range(self.n_qubits):
            backend.add_h_gate(circuit, q)

    def get_gradient_circuits(self) -> list[tuple[Any, float]]:
        """Get gradient circuits."""
        gradients = []

        for i, param in enumerate(self._parameters):
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


class MaxCutQAOA(QAOACircuit):
    """
    QAOA specialized for MaxCut problems.

    MaxCut: partition graph vertices into two sets
    maximizing edges between sets.

    Example:
        >>> edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> qaoa = MaxCutQAOA(n_qubits=4, edges=edges, p=2)
        >>> result = await backend.run_qaoa(qaoa)
    """

    def __init__(
        self,
        n_qubits: int,
        edges: list[tuple[int, int]],
        weights: list[float] | None = None,
        p: int = 1,
    ):
        """
        Initialize MaxCut QAOA.

        Args:
            n_qubits: Number of vertices/qubits.
            edges: List of edges (i, j).
            weights: Optional edge weights.
            p: QAOA depth.
        """
        super().__init__(n_qubits, p, mixer_type="x")

        self.edges = edges
        self.weights = weights or [1.0] * len(edges)

        # Build cost Hamiltonian
        # MaxCut: C = Σ w_ij (1 - Z_i Z_j) / 2
        # Equivalent to maximizing: Σ w_ij Z_i Z_j (with sign flip)
        cost_terms = []
        for (i, j), w in zip(edges, self.weights):
            cost_terms.append(([i, j], -w / 2))

        self.set_cost_hamiltonian(cost_terms)

    def evaluate_cut(self, bitstring: str) -> float:
        """Evaluate the cut value for a given bitstring."""
        cut_value = 0.0
        bits = [int(b) for b in bitstring]

        for (i, j), w in zip(self.edges, self.weights):
            if bits[i] != bits[j]:
                cut_value += w

        return cut_value

    def get_optimal_angles_heuristic(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get heuristic initial angles.

        Based on analytical results for p=1 on regular graphs.

        Returns:
            Tuple of (gammas, betas) arrays.
        """
        if self.p == 1:
            # Optimal for regular graphs
            gamma = np.array([0.25 * np.pi])
            beta = np.array([0.25 * np.pi])
        else:
            # Linear interpolation heuristic
            gamma = np.linspace(0.1 * np.pi, 0.5 * np.pi, self.p)
            beta = np.linspace(0.5 * np.pi, 0.1 * np.pi, self.p)

        return gamma, beta


@dataclass
class QAOAResult:
    """Result from QAOA optimization."""

    optimal_bitstring: str
    optimal_value: float
    optimal_params: np.ndarray
    convergence_history: list[float]
    all_samples: dict[str, int]
    approximation_ratio: float | None = None


class PortfolioQAOA(QAOACircuit):
    """
    QAOA for portfolio optimization.

    Minimize risk subject to return constraints.
    Uses Markowitz model encoded as QUBO.
    """

    def __init__(
        self,
        n_assets: int,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_factor: float = 0.5,
        budget: int | None = None,
        p: int = 1,
    ):
        """
        Initialize portfolio QAOA.

        Args:
            n_assets: Number of assets.
            expected_returns: Expected returns vector.
            covariance_matrix: Covariance matrix.
            risk_factor: Trade-off between risk and return.
            budget: Number of assets to select.
            p: QAOA depth.
        """
        mixer_type = "xy" if budget else "x"
        super().__init__(n_assets, p, mixer_type=mixer_type)

        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_factor = risk_factor
        self.budget = budget

        # Build cost Hamiltonian
        cost_terms = self._build_portfolio_hamiltonian()
        self.set_cost_hamiltonian(cost_terms)

    def _build_portfolio_hamiltonian(self) -> list[tuple[list[int], float]]:
        """Build portfolio cost Hamiltonian."""
        terms = []
        n = self.n_qubits

        # Linear terms: -μ_i x_i (mapped to Z)
        for i in range(n):
            coeff = -self.expected_returns[i] + self.risk_factor * self.covariance_matrix[i, i]
            terms.append(([i], coeff))

        # Quadratic terms: λ Σ_ij w_ij x_i x_j
        for i in range(n):
            for j in range(i + 1, n):
                coeff = self.risk_factor * self.covariance_matrix[i, j]
                if abs(coeff) > 1e-10:
                    terms.append(([i, j], coeff))

        return terms

    def evaluate_portfolio(self, bitstring: str) -> dict[str, float]:
        """Evaluate portfolio for a given selection."""
        selection = np.array([int(b) for b in bitstring])

        # Calculate return
        portfolio_return = np.dot(selection, self.expected_returns)

        # Calculate risk
        portfolio_risk = np.sqrt(
            selection @ self.covariance_matrix @ selection
        )

        return {
            "return": float(portfolio_return),
            "risk": float(portfolio_risk),
            "sharpe_ratio": float(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0,
            "n_assets": int(sum(selection)),
        }


class TSPCircuit(QAOACircuit):
    """
    QAOA circuit for Traveling Salesman Problem.

    Encodes TSP as a quadratic optimization problem.
    """

    def __init__(
        self,
        n_cities: int,
        distance_matrix: np.ndarray,
        p: int = 1,
    ):
        """
        Initialize TSP QAOA.

        Args:
            n_cities: Number of cities.
            distance_matrix: Distance matrix.
            p: QAOA depth.
        """
        # Need n^2 qubits for position encoding
        n_qubits = n_cities * n_cities
        super().__init__(n_qubits, p, mixer_type="x")

        self.n_cities = n_cities
        self.distance_matrix = distance_matrix

        # Build cost Hamiltonian
        cost_terms = self._build_tsp_hamiltonian()
        self.set_cost_hamiltonian(cost_terms)

    def _build_tsp_hamiltonian(self) -> list[tuple[list[int], float]]:
        """Build TSP cost Hamiltonian."""
        terms = []
        n = self.n_cities
        penalty = 10.0  # Constraint penalty

        # Cost: Σ d_ij x_i,p x_j,p+1
        for p in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        q1 = i * n + p
                        q2 = j * n + ((p + 1) % n)
                        coeff = self.distance_matrix[i, j]
                        terms.append(([q1, q2], coeff))

        # Constraint: each city visited once
        for i in range(n):
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    q1 = i * n + p1
                    q2 = i * n + p2
                    terms.append(([q1, q2], penalty))

        # Constraint: each position has one city
        for p in range(n):
            for i1 in range(n):
                for i2 in range(i1 + 1, n):
                    q1 = i1 * n + p
                    q2 = i2 * n + p
                    terms.append(([q1, q2], penalty))

        return terms

    def decode_solution(self, bitstring: str) -> list[int]:
        """Decode bitstring to tour."""
        n = self.n_cities
        tour = []

        for p in range(n):
            for i in range(n):
                if bitstring[i * n + p] == "1":
                    tour.append(i)
                    break

        return tour

    def evaluate_tour(self, tour: list[int]) -> float:
        """Calculate tour distance."""
        total = 0.0
        for i in range(len(tour)):
            total += self.distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return total
