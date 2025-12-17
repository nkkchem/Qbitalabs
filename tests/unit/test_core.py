"""Tests for core module."""

import pytest
import asyncio
from uuid import UUID

from qbitalabs.core.types import AgentID, AgentState, AgentRole, MessageType
from qbitalabs.core.config import QBitaLabsConfig, SwarmConfig, QuantumConfig


class TestTypes:
    """Test type definitions."""

    def test_agent_id_creation(self):
        """Test AgentID is a valid UUID."""
        agent_id = AgentID()
        assert isinstance(agent_id, UUID)

    def test_agent_state_enum(self):
        """Test AgentState enum values."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.ACTIVE.value == "active"
        assert AgentState.WAITING.value == "waiting"

    def test_agent_role_enum(self):
        """Test AgentRole enum values."""
        assert AgentRole.MOLECULAR_MODELER.value == "molecular_modeler"
        assert AgentRole.HYPOTHESIS_GENERATOR.value == "hypothesis_generator"

    def test_message_type_enum(self):
        """Test MessageType enum values."""
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.BROADCAST.value == "broadcast"


class TestConfig:
    """Test configuration classes."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = QBitaLabsConfig()
        assert config.environment == "development"
        assert config.log_level == "INFO"

    def test_swarm_config(self):
        """Test swarm configuration."""
        config = SwarmConfig()
        assert config.max_agents > 0
        assert config.message_queue_size > 0
        assert config.coordination_strategy in ["hierarchical", "stigmergy", "hybrid"]

    def test_quantum_config(self):
        """Test quantum configuration."""
        config = QuantumConfig()
        assert config.default_backend in ["qiskit", "cirq", "pennylane", "simulator"]
        assert config.default_shots > 0


class TestConfigValidation:
    """Test configuration validation."""

    def test_config_to_dict(self):
        """Test config serialization."""
        config = QBitaLabsConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "environment" in config_dict

    def test_config_from_dict(self):
        """Test config deserialization."""
        data = {
            "environment": "production",
            "log_level": "WARNING",
        }
        config = QBitaLabsConfig.from_dict(data)

        assert config.environment == "production"
        assert config.log_level == "WARNING"
