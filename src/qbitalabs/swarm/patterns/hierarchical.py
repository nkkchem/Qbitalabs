"""
Hierarchical Pattern for QBitaLabs

Implements multi-level agent organization:
- Strategic layer: High-level goal setting
- Planning layer: Task decomposition
- Execution layer: Task execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from qbitalabs.core.types import AgentID, ExecutionStatus
from qbitalabs.swarm.base_agent import BaseAgent, AgentMessage

logger = structlog.get_logger(__name__)


class HierarchyLevel(str, Enum):
    """Levels in the hierarchy."""

    STRATEGIC = "strategic"
    PLANNING = "planning"
    EXECUTION = "execution"


@dataclass
class Goal:
    """A high-level goal from strategic layer."""

    goal_id: str = field(default_factory=lambda: str(uuid4())[:8])
    description: str = ""
    priority: int = 5
    deadline: float | None = None  # Cycles until deadline
    status: ExecutionStatus = ExecutionStatus.PENDING
    sub_tasks: list[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    """A plan created by planning layer."""

    plan_id: str = field(default_factory=lambda: str(uuid4())[:8])
    goal_id: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    assigned_agents: list[AgentID] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress: float = 0.0


class HierarchicalPattern:
    """
    Manages hierarchical coordination in the swarm.

    Three levels:
    1. Strategic: Sets goals, monitors progress
    2. Planning: Decomposes goals into tasks
    3. Execution: Performs actual work

    Example:
        >>> hierarchy = HierarchicalPattern()
        >>> goal = hierarchy.create_goal("Find drug candidates")
        >>> plan = await hierarchy.create_plan(goal)
        >>> await hierarchy.execute_plan(plan)
    """

    def __init__(self):
        """Initialize the hierarchical pattern."""
        self._goals: dict[str, Goal] = {}
        self._plans: dict[str, TaskPlan] = {}
        self._agent_levels: dict[AgentID, HierarchyLevel] = {}

        self._logger = structlog.get_logger("hierarchy")

    def assign_level(self, agent_id: AgentID, level: HierarchyLevel) -> None:
        """
        Assign an agent to a hierarchy level.

        Args:
            agent_id: Agent to assign.
            level: Level to assign to.
        """
        self._agent_levels[agent_id] = level
        self._logger.info(
            "Agent assigned to level",
            agent_id=str(agent_id)[:8],
            level=level.value,
        )

    def get_level(self, agent_id: AgentID) -> HierarchyLevel:
        """Get the level of an agent."""
        return self._agent_levels.get(agent_id, HierarchyLevel.EXECUTION)

    def create_goal(
        self,
        description: str,
        priority: int = 5,
        deadline: float | None = None,
    ) -> Goal:
        """
        Create a new goal (strategic layer).

        Args:
            description: Goal description.
            priority: Priority (1-10).
            deadline: Cycles until deadline.

        Returns:
            Created goal.
        """
        goal = Goal(
            description=description,
            priority=priority,
            deadline=deadline,
            status=ExecutionStatus.PENDING,
        )

        self._goals[goal.goal_id] = goal

        self._logger.info(
            "Goal created",
            goal_id=goal.goal_id,
            description=description[:50],
        )

        return goal

    async def create_plan(
        self,
        goal: Goal,
        planning_agents: list[BaseAgent] | None = None,
    ) -> TaskPlan:
        """
        Create a plan for a goal (planning layer).

        Args:
            goal: Goal to plan for.
            planning_agents: Agents for planning.

        Returns:
            Task plan.
        """
        # Decompose goal into steps
        steps = self._decompose_goal(goal)

        plan = TaskPlan(
            goal_id=goal.goal_id,
            steps=steps,
            status=ExecutionStatus.PENDING,
        )

        self._plans[plan.plan_id] = plan
        goal.sub_tasks.append(plan.plan_id)

        self._logger.info(
            "Plan created",
            plan_id=plan.plan_id,
            goal_id=goal.goal_id,
            steps=len(steps),
        )

        return plan

    def _decompose_goal(self, goal: Goal) -> list[dict[str, Any]]:
        """Decompose a goal into execution steps."""
        # Simple rule-based decomposition
        # In production, use LLM for intelligent decomposition

        base_steps = [
            {
                "step_id": f"{goal.goal_id}_1",
                "action": "research",
                "description": "Gather relevant information",
                "role": "literature_reviewer",
            },
            {
                "step_id": f"{goal.goal_id}_2",
                "action": "analyze",
                "description": "Analyze gathered data",
                "role": "molecular_modeler",
            },
            {
                "step_id": f"{goal.goal_id}_3",
                "action": "synthesize",
                "description": "Synthesize findings",
                "role": "hypothesis_generator",
            },
            {
                "step_id": f"{goal.goal_id}_4",
                "action": "validate",
                "description": "Validate results",
                "role": "validation_agent",
            },
        ]

        return base_steps

    async def execute_plan(
        self,
        plan: TaskPlan,
        execution_agents: list[BaseAgent],
    ) -> dict[str, Any]:
        """
        Execute a plan (execution layer).

        Args:
            plan: Plan to execute.
            execution_agents: Available execution agents.

        Returns:
            Execution results.
        """
        plan.status = ExecutionStatus.RUNNING
        results = []
        completed_steps = 0

        for step in plan.steps:
            # Find suitable agent for step
            agent = self._find_agent_for_step(step, execution_agents)

            if agent:
                try:
                    result = await agent.process(step)
                    results.append({
                        "step_id": step["step_id"],
                        "status": "completed",
                        "result": result,
                    })
                    completed_steps += 1
                except Exception as e:
                    results.append({
                        "step_id": step["step_id"],
                        "status": "failed",
                        "error": str(e),
                    })
            else:
                results.append({
                    "step_id": step["step_id"],
                    "status": "skipped",
                    "reason": "No suitable agent",
                })

            plan.progress = completed_steps / len(plan.steps)

        plan.status = (
            ExecutionStatus.COMPLETED
            if completed_steps == len(plan.steps)
            else ExecutionStatus.FAILED
        )

        self._logger.info(
            "Plan execution complete",
            plan_id=plan.plan_id,
            completed=completed_steps,
            total=len(plan.steps),
        )

        return {
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "progress": plan.progress,
            "results": results,
        }

    def _find_agent_for_step(
        self,
        step: dict[str, Any],
        agents: list[BaseAgent],
    ) -> BaseAgent | None:
        """Find an agent suitable for a step."""
        required_role = step.get("role", "")

        for agent in agents:
            if agent.role.value == required_role:
                return agent

        # Return first available if no role match
        return agents[0] if agents else None

    def update_goal_status(self, goal_id: str, status: ExecutionStatus) -> None:
        """Update a goal's status."""
        if goal_id in self._goals:
            self._goals[goal_id].status = status

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_plan(self, plan_id: str) -> TaskPlan | None:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def get_active_goals(self) -> list[Goal]:
        """Get all active goals."""
        return [
            g for g in self._goals.values()
            if g.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get hierarchy statistics."""
        level_counts = {}
        for level in HierarchyLevel:
            level_counts[level.value] = sum(
                1 for l in self._agent_levels.values() if l == level
            )

        return {
            "total_goals": len(self._goals),
            "active_goals": len(self.get_active_goals()),
            "total_plans": len(self._plans),
            "agents_by_level": level_counts,
        }
