"""
Stigmergy Pattern for QBitaLabs

Implements indirect coordination through the environment:
- Pheromone trails for path optimization
- Environmental markers for information sharing
- Decay dynamics for temporal coordination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from qbitalabs.core.types import AgentID

logger = structlog.get_logger(__name__)


@dataclass
class PheromoneTrail:
    """A pheromone trail in the environment."""

    trail_id: str
    strength: float = 1.0
    created_by: AgentID | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reinforced: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float, by_agent: AgentID | None = None) -> None:
        """Reinforce the pheromone trail."""
        self.strength = min(10.0, self.strength + amount)
        self.last_reinforced = datetime.utcnow()
        if by_agent:
            self.metadata["last_reinforced_by"] = str(by_agent)

    def decay(self, rate: float) -> None:
        """Apply decay to the trail."""
        self.strength *= 1 - rate

    def is_expired(self, threshold: float = 0.01) -> bool:
        """Check if trail has decayed below threshold."""
        return self.strength < threshold


class StigmergyPattern:
    """
    Manages stigmergy-based coordination in the swarm.

    Stigmergy enables agents to:
    - Leave information in the environment (pheromones)
    - Follow trails left by other agents
    - Coordinate without direct communication
    - Build collective knowledge

    Example:
        >>> stigmergy = StigmergyPattern()
        >>> stigmergy.deposit("path_to_target", 2.0, agent_id)
        >>> best_path = stigmergy.get_strongest_trails(n=5)
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        evaporation_threshold: float = 0.01,
    ):
        """
        Initialize the stigmergy pattern.

        Args:
            decay_rate: Rate of pheromone decay per cycle.
            evaporation_threshold: Minimum strength before trail removal.
        """
        self.decay_rate = decay_rate
        self.evaporation_threshold = evaporation_threshold

        self._trails: dict[str, PheromoneTrail] = {}
        self._markers: dict[str, Any] = {}  # Static environmental markers

        self._logger = structlog.get_logger("stigmergy")

    def deposit(
        self,
        trail_id: str,
        strength: float,
        agent_id: AgentID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Deposit pheromone on a trail.

        Args:
            trail_id: Identifier for the trail.
            strength: Amount of pheromone to deposit.
            agent_id: Agent making the deposit.
            metadata: Additional trail metadata.
        """
        if trail_id in self._trails:
            self._trails[trail_id].reinforce(strength, agent_id)
        else:
            self._trails[trail_id] = PheromoneTrail(
                trail_id=trail_id,
                strength=strength,
                created_by=agent_id,
                metadata=metadata or {},
            )

        self._logger.debug(
            "Pheromone deposited",
            trail_id=trail_id,
            strength=self._trails[trail_id].strength,
        )

    def sense(self, trail_id: str) -> float:
        """
        Sense pheromone strength at a trail.

        Args:
            trail_id: Trail to sense.

        Returns:
            Pheromone concentration (0 if not found).
        """
        trail = self._trails.get(trail_id)
        return trail.strength if trail else 0.0

    def sense_pattern(self, pattern: str) -> dict[str, float]:
        """
        Sense all trails matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard).

        Returns:
            Dictionary of trail_id -> strength.
        """
        result = {}

        if "*" in pattern:
            prefix = pattern.split("*")[0]
            for trail_id, trail in self._trails.items():
                if trail_id.startswith(prefix):
                    result[trail_id] = trail.strength
        else:
            if pattern in self._trails:
                result[pattern] = self._trails[pattern].strength

        return result

    def get_strongest_trails(self, n: int = 5, prefix: str = "") -> list[PheromoneTrail]:
        """
        Get the strongest pheromone trails.

        Args:
            n: Number of trails to return.
            prefix: Optional prefix filter.

        Returns:
            List of strongest trails.
        """
        trails = list(self._trails.values())

        if prefix:
            trails = [t for t in trails if t.trail_id.startswith(prefix)]

        trails.sort(key=lambda t: t.strength, reverse=True)
        return trails[:n]

    def choose_probabilistic(
        self, options: list[str], temperature: float = 1.0
    ) -> str | None:
        """
        Choose an option probabilistically based on pheromone strength.

        Higher pheromone = higher probability of selection.

        Args:
            options: List of trail_ids to choose from.
            temperature: Controls randomness (higher = more random).

        Returns:
            Selected trail_id or None if no valid options.
        """
        strengths = [self.sense(opt) for opt in options]

        if sum(strengths) == 0:
            return options[0] if options else None

        # Apply temperature
        adjusted = np.array(strengths) ** (1 / temperature)
        probabilities = adjusted / adjusted.sum()

        return np.random.choice(options, p=probabilities)

    def decay_all(self) -> int:
        """
        Apply decay to all trails.

        Returns:
            Number of trails evaporated.
        """
        evaporated = []

        for trail_id, trail in self._trails.items():
            trail.decay(self.decay_rate)
            if trail.is_expired(self.evaporation_threshold):
                evaporated.append(trail_id)

        for trail_id in evaporated:
            del self._trails[trail_id]

        if evaporated:
            self._logger.debug(
                "Trails evaporated",
                count=len(evaporated),
            )

        return len(evaporated)

    def set_marker(self, marker_id: str, value: Any) -> None:
        """
        Set a static environmental marker.

        Markers don't decay like pheromones.

        Args:
            marker_id: Marker identifier.
            value: Marker value.
        """
        self._markers[marker_id] = value

    def get_marker(self, marker_id: str) -> Any:
        """Get a static marker value."""
        return self._markers.get(marker_id)

    def clear_marker(self, marker_id: str) -> None:
        """Clear a static marker."""
        self._markers.pop(marker_id, None)

    def get_statistics(self) -> dict[str, Any]:
        """Get stigmergy statistics."""
        if not self._trails:
            return {
                "total_trails": 0,
                "avg_strength": 0,
                "max_strength": 0,
                "markers_count": len(self._markers),
            }

        strengths = [t.strength for t in self._trails.values()]

        return {
            "total_trails": len(self._trails),
            "avg_strength": np.mean(strengths),
            "max_strength": max(strengths),
            "min_strength": min(strengths),
            "markers_count": len(self._markers),
        }

    def export_state(self) -> dict[str, Any]:
        """Export the current stigmergy state."""
        return {
            "trails": {
                tid: {
                    "strength": t.strength,
                    "created_by": str(t.created_by) if t.created_by else None,
                    "metadata": t.metadata,
                }
                for tid, t in self._trails.items()
            },
            "markers": self._markers.copy(),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Import a stigmergy state."""
        self._trails.clear()
        self._markers.clear()

        for tid, data in state.get("trails", {}).items():
            self._trails[tid] = PheromoneTrail(
                trail_id=tid,
                strength=data["strength"],
                metadata=data.get("metadata", {}),
            )

        self._markers.update(state.get("markers", {}))
