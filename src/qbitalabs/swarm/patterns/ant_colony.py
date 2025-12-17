"""
Ant Colony Pattern for QBitaLabs

Implements Ant Colony Optimization (ACO) for:
- Path optimization
- Resource allocation
- Solution space exploration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from qbitalabs.swarm.patterns.stigmergy import StigmergyPattern

logger = structlog.get_logger(__name__)


@dataclass
class AntPath:
    """A path taken by an ant (agent)."""

    path_id: str = ""
    nodes: list[str] = field(default_factory=list)
    cost: float = float("inf")
    pheromone_deposited: float = 0.0


class AntColonyPattern:
    """
    Implements Ant Colony Optimization for the swarm.

    Features:
    - Pheromone-based path selection
    - Exploration vs exploitation balance
    - Evaporation and reinforcement
    - Solution convergence

    Example:
        >>> aco = AntColonyPattern()
        >>> aco.initialize_graph(nodes, edges)
        >>> best_path = await aco.optimize(n_iterations=100)
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,  # Heuristic importance
        evaporation_rate: float = 0.1,
        q: float = 100.0,  # Pheromone deposit factor
    ):
        """
        Initialize the ACO pattern.

        Args:
            alpha: Weight of pheromone influence.
            beta: Weight of heuristic (distance) influence.
            evaporation_rate: Rate of pheromone evaporation.
            q: Constant for pheromone deposit calculation.
        """
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q

        self._stigmergy = StigmergyPattern(decay_rate=evaporation_rate)
        self._graph: dict[str, dict[str, float]] = {}  # node -> {neighbor: distance}
        self._best_path: AntPath | None = None

        self._logger = structlog.get_logger("ant_colony")

    def initialize_graph(
        self,
        nodes: list[str],
        edges: list[tuple[str, str, float]],
    ) -> None:
        """
        Initialize the graph for optimization.

        Args:
            nodes: List of node identifiers.
            edges: List of (from, to, distance) tuples.
        """
        self._graph = {node: {} for node in nodes}

        for from_node, to_node, distance in edges:
            if from_node in self._graph:
                self._graph[from_node][to_node] = distance
            if to_node in self._graph:
                self._graph[to_node][from_node] = distance

        # Initialize pheromones
        for from_node, neighbors in self._graph.items():
            for to_node in neighbors:
                trail_id = f"{from_node}_{to_node}"
                self._stigmergy.deposit(trail_id, 0.1)

        self._logger.info(
            "Graph initialized",
            nodes=len(nodes),
            edges=len(edges),
        )

    def construct_path(
        self,
        start: str,
        end: str,
        visited: set[str] | None = None,
    ) -> AntPath:
        """
        Construct a path using probabilistic selection.

        Args:
            start: Starting node.
            end: Target node.
            visited: Already visited nodes.

        Returns:
            Constructed path.
        """
        path = AntPath(path_id=f"path_{start}_{end}")
        current = start
        path.nodes.append(current)
        visited = visited or set()
        visited.add(current)
        total_cost = 0.0

        while current != end:
            neighbors = self._get_unvisited_neighbors(current, visited)

            if not neighbors:
                # Dead end
                path.cost = float("inf")
                return path

            # Select next node probabilistically
            next_node = self._select_next_node(current, neighbors)
            distance = self._graph[current].get(next_node, 1.0)

            path.nodes.append(next_node)
            visited.add(next_node)
            total_cost += distance
            current = next_node

        path.cost = total_cost
        return path

    def _get_unvisited_neighbors(
        self, node: str, visited: set[str]
    ) -> list[str]:
        """Get unvisited neighbors of a node."""
        neighbors = self._graph.get(node, {}).keys()
        return [n for n in neighbors if n not in visited]

    def _select_next_node(
        self, current: str, candidates: list[str]
    ) -> str:
        """Select next node using ACO probability formula."""
        if len(candidates) == 1:
            return candidates[0]

        probabilities = []

        for candidate in candidates:
            trail_id = f"{current}_{candidate}"
            pheromone = max(self._stigmergy.sense(trail_id), 0.01)
            distance = self._graph[current].get(candidate, 1.0)
            heuristic = 1.0 / max(distance, 0.01)

            # ACO probability formula
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)

        # Normalize
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return np.random.choice(candidates, p=probabilities)

    def deposit_pheromone(self, path: AntPath) -> None:
        """
        Deposit pheromone along a path.

        Args:
            path: Path to deposit pheromone on.
        """
        if path.cost == float("inf"):
            return

        deposit_amount = self.q / path.cost

        for i in range(len(path.nodes) - 1):
            trail_id = f"{path.nodes[i]}_{path.nodes[i + 1]}"
            self._stigmergy.deposit(trail_id, deposit_amount)

        path.pheromone_deposited = deposit_amount

    async def optimize(
        self,
        start: str,
        end: str,
        n_ants: int = 10,
        n_iterations: int = 100,
    ) -> AntPath:
        """
        Run ACO optimization.

        Args:
            start: Starting node.
            end: Target node.
            n_ants: Number of ants per iteration.
            n_iterations: Number of iterations.

        Returns:
            Best path found.
        """
        self._best_path = None

        for iteration in range(n_iterations):
            iteration_paths = []

            # Each ant constructs a path
            for _ in range(n_ants):
                path = self.construct_path(start, end)
                iteration_paths.append(path)

                if self._best_path is None or path.cost < self._best_path.cost:
                    self._best_path = path

            # Evaporate pheromones
            self._stigmergy.decay_all()

            # Deposit pheromone for successful paths
            for path in iteration_paths:
                if path.cost < float("inf"):
                    self.deposit_pheromone(path)

            # Elite ant: extra deposit for best path
            if self._best_path:
                self.deposit_pheromone(self._best_path)

            if iteration % 10 == 0:
                self._logger.debug(
                    "ACO iteration",
                    iteration=iteration,
                    best_cost=self._best_path.cost if self._best_path else None,
                )

        self._logger.info(
            "ACO optimization complete",
            iterations=n_iterations,
            best_cost=self._best_path.cost if self._best_path else None,
        )

        return self._best_path

    def get_best_path(self) -> AntPath | None:
        """Get the best path found."""
        return self._best_path

    def get_pheromone_map(self) -> dict[str, float]:
        """Get current pheromone levels."""
        return {
            tid: trail.strength
            for tid, trail in self._stigmergy._trails.items()
        }
