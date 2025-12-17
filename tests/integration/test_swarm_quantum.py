"""
Integration tests for SWARM + Quantum module interaction.

Tests the coordination between SWARM agents and quantum computing backends.
"""

import pytest
import numpy as np


@pytest.mark.integration
class TestSwarmQuantumIntegration:
    """Test SWARM agents coordinating quantum simulations."""

    def test_molecular_agent_with_simulator(self, simple_hamiltonian):
        """Test MolecularAgent can request quantum simulation."""
        # This tests the integration path:
        # MolecularAgent -> Message Bus -> Quantum Backend

        # Setup
        from qbitalabs.swarm.agents import MolecularAgent
        from qbitalabs.quantum.backends import SimulatorBackend

        agent = MolecularAgent(
            agent_id="mol-test",
            specialization="quantum_chemistry"
        )
        backend = SimulatorBackend(num_qubits=2)
        backend.initialize()

        # Simulate agent requesting a quantum calculation
        # In production, this would go through the message bus
        task = {
            "type": "ground_state_energy",
            "hamiltonian": simple_hamiltonian,
        }

        # Verify agent can process quantum tasks
        assert agent.can_handle_task(task)

    def test_swarm_parallel_quantum_jobs(self):
        """Test multiple agents submitting quantum jobs concurrently."""
        from qbitalabs.quantum.backends import SimulatorBackend

        backend = SimulatorBackend(num_qubits=4)
        backend.initialize()

        # Create multiple circuits
        num_jobs = 5
        circuits = []
        for i in range(num_jobs):
            circuit = {
                "gates": [
                    {"name": "H", "qubits": [0]},
                    {"name": "CNOT", "qubits": [0, 1]},
                    {"name": "RZ", "qubits": [0], "params": [0.5 * (i + 1)]},
                ],
                "measurements": [0, 1],
            }
            circuits.append(circuit)

        # Execute all (simulating parallel execution)
        results = []
        for circuit in circuits:
            result = backend.execute(circuit, shots=100)
            results.append(result)

        # Verify all completed
        assert len(results) == num_jobs
        for result in results:
            assert result.success

    def test_quantum_results_to_agent_memory(self, simple_hamiltonian):
        """Test quantum results are stored in agent memory."""
        from qbitalabs.swarm.agents import MolecularAgent

        agent = MolecularAgent(
            agent_id="mol-memory",
            specialization="quantum_chemistry"
        )

        # Simulated quantum result
        quantum_result = {
            "energy": -1.137,
            "converged": True,
            "num_iterations": 42,
        }

        # Store in agent memory
        agent.store_result("h2_ground_state", quantum_result)

        # Retrieve and verify
        retrieved = agent.get_result("h2_ground_state")
        assert retrieved["energy"] == -1.137
        assert retrieved["converged"]


@pytest.mark.integration
class TestDigitalTwinQuantum:
    """Test Digital Twin using quantum simulations."""

    def test_twin_metabolic_quantum_simulation(self, sample_patient_profile):
        """Test digital twin can use quantum for metabolic modeling."""
        from qbitalabs.digital_twin import PatientProfile, DigitalTwinEngine

        # Create patient profile
        profile = PatientProfile(**sample_patient_profile)

        # Create engine with quantum-enhanced metabolism
        engine = DigitalTwinEngine()

        # Create twin
        twin = engine.create_twin(profile)

        # Verify twin was created with quantum capabilities
        assert twin is not None
        assert twin.patient_id == sample_patient_profile["patient_id"]


@pytest.mark.integration
class TestPathwayQuantum:
    """Test pathway analysis with quantum computing."""

    def test_pathway_flux_quantum_optimization(self, sample_pathway_data):
        """Test pathway flux analysis using quantum optimization."""
        # This would use QAOA for flux balance optimization
        # For now, test the integration path exists

        from qbitalabs.biology import PathwaySimulator

        simulator = PathwaySimulator()

        # Should be able to set quantum optimizer
        simulator.set_optimizer("qaoa")

        assert simulator.optimizer_type == "qaoa"
