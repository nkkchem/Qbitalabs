"""
QBitaLabs Test Configuration and Fixtures.

This module provides shared fixtures and configuration for all test types.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================================
# Async Support
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Core Fixtures
# ============================================================================


@pytest.fixture
def sample_config() -> dict:
    """Provide a sample configuration dictionary."""
    return {
        "project": {"name": "qbitalabs-test", "version": "0.1.0"},
        "quantum": {
            "default_backend": "simulator",
            "backends": {"simulator": {"num_qubits": 4, "shots": 100}},
        },
        "swarm": {"max_agents": 10, "coordination_pattern": "hierarchical"},
    }


@pytest.fixture
def temp_data_dir(tmp_path) -> str:
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "models").mkdir()
    return str(data_dir)


# ============================================================================
# SWARM Agent Fixtures
# ============================================================================


@pytest.fixture
def mock_message_bus() -> MagicMock:
    """Create a mock message bus for agent testing."""
    bus = MagicMock()
    bus.publish = MagicMock(return_value=None)
    bus.subscribe = MagicMock(return_value=None)
    bus.get_messages = MagicMock(return_value=[])
    return bus


@pytest.fixture
def sample_agent_config() -> dict:
    """Provide sample agent configuration."""
    return {
        "agent_id": "test-agent-001",
        "specialization": "molecular",
        "capabilities": ["analyze", "optimize"],
        "timeout_seconds": 30,
    }


# ============================================================================
# Quantum Computing Fixtures
# ============================================================================


@pytest.fixture
def simple_hamiltonian() -> dict:
    """Provide a simple Hamiltonian for testing."""
    return {
        "terms": {
            "ZZ": 0.5,
            "XX": 0.3,
            "II": -0.1,
        },
        "num_qubits": 2,
    }


@pytest.fixture
def h2_molecule_data() -> dict:
    """Provide H2 molecule test data."""
    return {
        "name": "H2",
        "atoms": [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
        "charge": 0,
        "multiplicity": 1,
        "basis": "sto-3g",
        "expected_energy": -1.137,  # Approximate ground state energy
    }


# ============================================================================
# Neuromorphic Computing Fixtures
# ============================================================================


@pytest.fixture
def spike_train_data() -> dict:
    """Provide sample spike train data."""
    import numpy as np

    np.random.seed(42)
    return {
        "times": np.sort(np.random.uniform(0, 100, 20)),
        "neuron_ids": np.random.randint(0, 10, 20),
        "num_neurons": 10,
        "duration_ms": 100,
    }


@pytest.fixture
def lif_neuron_params() -> dict:
    """Provide LIF neuron parameters."""
    return {
        "tau_m": 10.0,  # Membrane time constant (ms)
        "v_rest": -65.0,  # Resting potential (mV)
        "v_thresh": -50.0,  # Threshold potential (mV)
        "v_reset": -70.0,  # Reset potential (mV)
        "tau_ref": 2.0,  # Refractory period (ms)
    }


# ============================================================================
# Digital Twin Fixtures
# ============================================================================


@pytest.fixture
def sample_patient_profile() -> dict:
    """Provide a sample patient profile."""
    return {
        "patient_id": "test-patient-001",
        "age": 45,
        "sex": "male",
        "height_cm": 175,
        "weight_kg": 80,
        "genomics": {
            "APOE": "e3/e3",
            "CYP2D6": "*1/*1",
        },
        "biomarkers": {
            "glucose_fasting": 95,
            "hba1c": 5.4,
            "ldl_cholesterol": 120,
            "hdl_cholesterol": 55,
        },
    }


@pytest.fixture
def sample_twin_state() -> dict:
    """Provide a sample digital twin state."""
    return {
        "timestamp": 0.0,
        "metabolic": {
            "glucose": 95.0,
            "insulin": 10.0,
        },
        "cardiovascular": {
            "heart_rate": 72,
            "systolic_bp": 120,
            "diastolic_bp": 80,
        },
        "biological_age": 45.0,
    }


# ============================================================================
# Biology Module Fixtures
# ============================================================================


@pytest.fixture
def sample_omics_data() -> dict:
    """Provide sample omics data."""
    import numpy as np

    np.random.seed(42)
    return {
        "type": "transcriptomics",
        "genes": [f"GENE_{i}" for i in range(100)],
        "expression": np.random.lognormal(0, 1, 100),
        "sample_id": "sample-001",
    }


@pytest.fixture
def sample_pathway_data() -> dict:
    """Provide sample pathway data."""
    return {
        "pathway_id": "hsa04910",
        "name": "Insulin signaling pathway",
        "genes": ["INS", "INSR", "IRS1", "PIK3CA", "AKT1", "GSK3B"],
        "reactions": [
            {"source": "INS", "target": "INSR", "type": "binding"},
            {"source": "INSR", "target": "IRS1", "type": "phosphorylation"},
        ],
    }


# ============================================================================
# ML Model Fixtures
# ============================================================================


@pytest.fixture
def sample_molecular_graph() -> dict:
    """Provide a sample molecular graph for GNN testing."""
    import numpy as np

    # Simple molecule graph (ethanol-like)
    return {
        "node_features": np.array(
            [[6, 0, 0], [6, 0, 0], [8, 0, 1]]  # C  # C  # O
        ),
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": np.array([[1], [1], [1], [1]]),  # Bond orders
        "num_nodes": 3,
        "num_edges": 4,
    }


# ============================================================================
# API Fixtures
# ============================================================================


@pytest.fixture
def api_client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    from qbitalabs.api import app

    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
    """Provide authentication headers for API testing."""
    return {
        "Authorization": "Bearer test-token-12345",
        "Content-Type": "application/json",
    }


# ============================================================================
# Utility Functions
# ============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "quantum: marks tests requiring quantum simulation")
    config.addinivalue_line("markers", "neuromorphic: marks tests requiring neuromorphic simulation")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")
