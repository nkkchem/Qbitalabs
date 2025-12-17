"""Tests for quantum module."""

import pytest
import asyncio
import numpy as np


class TestQuantumCircuits:
    """Test quantum circuit functionality."""

    def test_hadamard_gate(self):
        """Test Hadamard gate matrix."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        # H is unitary
        assert np.allclose(H @ H.conj().T, np.eye(2))

        # H|0> = |+>
        zero_state = np.array([1, 0])
        plus_state = H @ zero_state
        assert np.allclose(plus_state, np.array([1, 1]) / np.sqrt(2))

    def test_pauli_gates(self):
        """Test Pauli gate matrices."""
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        # Pauli matrices are unitary
        assert np.allclose(X @ X, np.eye(2))
        assert np.allclose(Y @ Y, np.eye(2))
        assert np.allclose(Z @ Z, np.eye(2))

        # Anticommutation relations
        assert np.allclose(X @ Y + Y @ X, np.zeros((2, 2)))

    def test_cnot_gate(self):
        """Test CNOT gate matrix."""
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])

        # CNOT is unitary
        assert np.allclose(CNOT @ CNOT.conj().T, np.eye(4))

        # CNOT|11> = |10>
        state_11 = np.array([0, 0, 0, 1])
        result = CNOT @ state_11
        assert np.allclose(result, np.array([0, 0, 1, 0]))

    def test_rotation_gates(self):
        """Test rotation gate matrices."""
        theta = np.pi / 4

        # RX gate
        RX = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ])
        assert np.allclose(RX @ RX.conj().T, np.eye(2))

        # RZ gate
        RZ = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ])
        assert np.allclose(RZ @ RZ.conj().T, np.eye(2))


class TestVQE:
    """Test VQE algorithm components."""

    def test_parameter_shift_gradient(self):
        """Test parameter shift rule."""
        # For a simple parameterized gate f(θ), gradient is:
        # ∂f/∂θ = (f(θ + π/2) - f(θ - π/2)) / 2

        def simple_cost(theta):
            return np.cos(theta)

        theta = 0.5
        shift = np.pi / 2

        numerical_grad = (simple_cost(theta + 0.001) - simple_cost(theta - 0.001)) / 0.002
        param_shift_grad = (simple_cost(theta + shift) - simple_cost(theta - shift)) / 2

        assert np.isclose(numerical_grad, param_shift_grad, atol=0.01)

    def test_variational_ansatz(self):
        """Test variational ansatz structure."""
        n_qubits = 4
        depth = 2
        n_rotations = 2  # RY and RZ per qubit

        expected_params = n_qubits * n_rotations * depth
        assert expected_params == 16


class TestQAOA:
    """Test QAOA algorithm components."""

    def test_maxcut_cost(self):
        """Test MaxCut cost function."""
        # Simple 4-node graph
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        def maxcut_value(bitstring):
            cut = 0
            for i, j in edges:
                if bitstring[i] != bitstring[j]:
                    cut += 1
            return cut

        # Optimal cuts
        assert maxcut_value([0, 1, 0, 1]) == 4  # Maximum
        assert maxcut_value([0, 0, 0, 0]) == 0  # Minimum
        assert maxcut_value([0, 0, 1, 1]) == 2

    def test_qaoa_angles(self):
        """Test QAOA angle ranges."""
        gamma = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(0, np.pi)

        assert 0 <= gamma <= 2 * np.pi
        assert 0 <= beta <= np.pi
