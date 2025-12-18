"""
Particle Swarm Pattern for QBitaLabs

Implements Particle Swarm Optimization (PSO) for:
- Continuous optimization problems
- Parameter tuning
- Drug property optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Particle:
    """A particle in the swarm."""

    particle_id: str = ""
    position: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    best_position: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = float("inf")
    current_fitness: float = float("inf")


class ParticleSwarmPattern:
    """
    Implements Particle Swarm Optimization for the swarm.

    Features:
    - Velocity-based exploration
    - Personal and global best tracking
    - Inertia weight adaptation
    - Boundary handling

    Example:
        >>> pso = ParticleSwarmPattern(n_dimensions=5)
        >>> pso.set_bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
        >>> best = await pso.optimize(objective_function, n_particles=20)
    """

    def __init__(
        self,
        n_dimensions: int,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive coefficient
        c2: float = 1.5,  # Social coefficient
    ):
        """
        Initialize PSO.

        Args:
            n_dimensions: Number of dimensions in solution space.
            w: Inertia weight (momentum).
            c1: Personal best attraction.
            c2: Global best attraction.
        """
        self.n_dimensions = n_dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self._particles: list[Particle] = []
        self._global_best_position: np.ndarray | None = None
        self._global_best_fitness: float = float("inf")

        self._bounds_lower: np.ndarray | None = None
        self._bounds_upper: np.ndarray | None = None

        self._logger = structlog.get_logger("particle_swarm")

    def set_bounds(
        self,
        lower: list[float],
        upper: list[float],
    ) -> None:
        """
        Set the bounds for the search space.

        Args:
            lower: Lower bounds for each dimension.
            upper: Upper bounds for each dimension.
        """
        self._bounds_lower = np.array(lower)
        self._bounds_upper = np.array(upper)

    def initialize_particles(self, n_particles: int) -> None:
        """
        Initialize the particle swarm.

        Args:
            n_particles: Number of particles.
        """
        self._particles = []

        for i in range(n_particles):
            # Random initial position
            if self._bounds_lower is not None and self._bounds_upper is not None:
                position = np.random.uniform(
                    self._bounds_lower,
                    self._bounds_upper,
                    self.n_dimensions,
                )
            else:
                position = np.random.uniform(-1, 1, self.n_dimensions)

            # Random initial velocity
            velocity = np.random.uniform(-0.1, 0.1, self.n_dimensions)

            particle = Particle(
                particle_id=f"particle_{i}",
                position=position,
                velocity=velocity,
                best_position=position.copy(),
            )
            self._particles.append(particle)

        self._logger.info(
            "Particles initialized",
            n_particles=n_particles,
            dimensions=self.n_dimensions,
        )

    def update_particle(
        self,
        particle: Particle,
        objective: Callable[[np.ndarray], float],
    ) -> None:
        """
        Update a particle's position and velocity.

        Args:
            particle: Particle to update.
            objective: Objective function to minimize.
        """
        # Evaluate fitness
        fitness = objective(particle.position)
        particle.current_fitness = fitness

        # Update personal best
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()

        # Update global best
        if fitness < self._global_best_fitness:
            self._global_best_fitness = fitness
            self._global_best_position = particle.position.copy()

        # Update velocity
        r1 = np.random.random(self.n_dimensions)
        r2 = np.random.random(self.n_dimensions)

        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (
            (self._global_best_position if self._global_best_position is not None else particle.position)
            - particle.position
        )

        particle.velocity = self.w * particle.velocity + cognitive + social

        # Velocity clamping
        max_velocity = 0.2 * (
            (self._bounds_upper - self._bounds_lower)
            if self._bounds_lower is not None and self._bounds_upper is not None
            else np.ones(self.n_dimensions)
        )
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)

        # Update position
        particle.position = particle.position + particle.velocity

        # Boundary handling
        if self._bounds_lower is not None and self._bounds_upper is not None:
            # Reflect particles at boundaries
            for d in range(self.n_dimensions):
                if particle.position[d] < self._bounds_lower[d]:
                    particle.position[d] = self._bounds_lower[d]
                    particle.velocity[d] *= -0.5
                elif particle.position[d] > self._bounds_upper[d]:
                    particle.position[d] = self._bounds_upper[d]
                    particle.velocity[d] *= -0.5

    async def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        n_particles: int = 20,
        n_iterations: int = 100,
        early_stop_tolerance: float = 1e-6,
        early_stop_patience: int = 20,
    ) -> dict[str, Any]:
        """
        Run PSO optimization.

        Args:
            objective: Function to minimize.
            n_particles: Number of particles.
            n_iterations: Maximum iterations.
            early_stop_tolerance: Improvement threshold for early stopping.
            early_stop_patience: Iterations without improvement before stopping.

        Returns:
            Optimization results.
        """
        self.initialize_particles(n_particles)

        best_fitness_history = []
        no_improvement_count = 0
        previous_best = float("inf")

        for iteration in range(n_iterations):
            # Update all particles
            for particle in self._particles:
                self.update_particle(particle, objective)

            best_fitness_history.append(self._global_best_fitness)

            # Check for early stopping
            improvement = previous_best - self._global_best_fitness
            if improvement < early_stop_tolerance:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            previous_best = self._global_best_fitness

            if no_improvement_count >= early_stop_patience:
                self._logger.info(
                    "Early stopping",
                    iteration=iteration,
                    best_fitness=self._global_best_fitness,
                )
                break

            # Adaptive inertia weight
            self.w = max(0.4, self.w * 0.99)

            if iteration % 10 == 0:
                self._logger.debug(
                    "PSO iteration",
                    iteration=iteration,
                    best_fitness=self._global_best_fitness,
                )

        self._logger.info(
            "PSO optimization complete",
            iterations=len(best_fitness_history),
            best_fitness=self._global_best_fitness,
        )

        return {
            "best_position": self._global_best_position.tolist() if self._global_best_position is not None else None,
            "best_fitness": self._global_best_fitness,
            "iterations": len(best_fitness_history),
            "fitness_history": best_fitness_history,
            "final_inertia": self.w,
        }

    def get_swarm_state(self) -> dict[str, Any]:
        """Get current swarm state."""
        return {
            "n_particles": len(self._particles),
            "global_best_fitness": self._global_best_fitness,
            "global_best_position": (
                self._global_best_position.tolist()
                if self._global_best_position is not None
                else None
            ),
            "particle_positions": [
                p.position.tolist() for p in self._particles
            ],
            "particle_fitnesses": [
                p.current_fitness for p in self._particles
            ],
        }
