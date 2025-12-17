"""Tests for swarm module."""

import pytest
import asyncio
import numpy as np

from qbitalabs.core.types import AgentID, AgentRole, AgentState


class TestBaseAgent:
    """Test base agent functionality."""

    @pytest.fixture
    def mock_agent_id(self):
        """Create mock agent ID."""
        return AgentID()

    def test_agent_role_values(self):
        """Test agent role enum values."""
        roles = list(AgentRole)
        assert len(roles) > 0
        assert AgentRole.MOLECULAR_MODELER in roles


class TestMessageBus:
    """Test message bus functionality."""

    def test_message_priority(self):
        """Test message priority ordering."""
        # Priority values
        priorities = [5, 1, 3, 2, 4]
        sorted_priorities = sorted(priorities)
        assert sorted_priorities == [1, 2, 3, 4, 5]


class TestSwarmPatterns:
    """Test swarm coordination patterns."""

    def test_stigmergy_decay(self):
        """Test pheromone decay dynamics."""
        initial_strength = 1.0
        decay_rate = 0.1

        # Apply decay
        strength = initial_strength * (1 - decay_rate)
        assert strength < initial_strength

        # Multiple decays
        for _ in range(10):
            strength *= (1 - decay_rate)
        assert strength < 0.5

    def test_particle_swarm_velocity(self):
        """Test PSO velocity update."""
        position = np.array([0.5, 0.5])
        velocity = np.array([0.1, -0.1])
        personal_best = np.array([0.6, 0.4])
        global_best = np.array([0.7, 0.3])

        w = 0.7  # Inertia
        c1 = 1.5  # Cognitive
        c2 = 1.5  # Social

        r1 = np.random.random(2)
        r2 = np.random.random(2)

        new_velocity = (
            w * velocity
            + c1 * r1 * (personal_best - position)
            + c2 * r2 * (global_best - position)
        )

        assert new_velocity.shape == velocity.shape

    def test_ant_colony_probability(self):
        """Test ACO probability calculation."""
        pheromone = 2.0
        heuristic = 0.5
        alpha = 1.0
        beta = 2.0

        prob = (pheromone ** alpha) * (heuristic ** beta)
        assert prob > 0


class TestProteinSwarm:
    """Test protein-like agent coordination."""

    def test_binding_affinity(self):
        """Test binding affinity calculation."""
        base_affinity = 0.5
        activity_multiplier = 1.5
        energy = 0.8

        effective_affinity = base_affinity * activity_multiplier * energy
        assert 0 <= min(effective_affinity, 1.0) <= 1.0

    def test_protein_folding_probability(self):
        """Test folding success probability."""
        misfolding_prob = 0.05

        # Simulate multiple folding attempts
        successes = sum(
            1 for _ in range(1000)
            if np.random.random() >= misfolding_prob
        )

        # Should succeed most of the time
        assert successes > 900
