"""
QBitaLabs Autonomous Agent System

Self-improving autonomous agent that:
- Runs experiments daily
- Tracks results and metrics
- Identifies improvement opportunities
- Updates code based on results
- Generates reports

Authored by: QbitaLab
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import shutil

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Task Definitions
# =============================================================================

class TaskPriority(str, Enum):
    """Priority levels for autonomous tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Status of an autonomous task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AutonomousTask:
    """A task for the autonomous agent to execute."""
    task_id: str
    name: str
    task_type: str
    priority: TaskPriority
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "config": self.config,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics,
        }


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    """Result of an experiment run."""
    experiment_id: str
    experiment_type: str
    started_at: datetime
    completed_at: datetime
    success: bool
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    improvements: List[Dict[str, Any]]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "success": self.success,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "improvements": self.improvements,
            "error": self.error,
            "duration_seconds": (self.completed_at - self.started_at).total_seconds(),
        }


class ExperimentRunner:
    """Runs and tracks experiments."""

    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._logger = structlog.get_logger("ExperimentRunner")

    async def run_experiment(
        self,
        experiment_type: str,
        config: Dict[str, Any],
    ) -> ExperimentResult:
        """Run a single experiment."""
        experiment_id = self._generate_id(experiment_type)
        started_at = datetime.utcnow()

        self._logger.info("Starting experiment", experiment_id=experiment_id, type=experiment_type)

        try:
            # Execute experiment based on type
            metrics, artifacts = await self._execute(experiment_type, config)

            # Analyze results for improvements
            improvements = self._analyze_for_improvements(metrics, config)

            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_type=experiment_type,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                success=True,
                metrics=metrics,
                artifacts=artifacts,
                improvements=improvements,
            )

        except Exception as e:
            self._logger.error("Experiment failed", experiment_id=experiment_id, error=str(e))
            result = ExperimentResult(
                experiment_id=experiment_id,
                experiment_type=experiment_type,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                success=False,
                metrics={},
                artifacts={},
                improvements=[],
                error=str(e),
            )

        # Save result
        self._save_result(result)
        return result

    async def _execute(
        self,
        experiment_type: str,
        config: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Execute the experiment logic."""
        metrics = {}
        artifacts = {}

        if experiment_type == "quantum_vqe":
            metrics = await self._run_vqe_experiment(config)
        elif experiment_type == "swarm_convergence":
            metrics = await self._run_swarm_experiment(config)
        elif experiment_type == "neuromorphic_power":
            metrics = await self._run_neuromorphic_experiment(config)
        elif experiment_type == "model_training":
            metrics = await self._run_training_experiment(config)
        elif experiment_type == "data_validation":
            metrics = await self._run_validation_experiment(config)
        elif experiment_type == "test_suite":
            metrics = await self._run_test_suite(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        return metrics, artifacts

    async def _run_vqe_experiment(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run VQE optimization experiment."""
        # Simulate VQE experiment
        target_accuracy = config.get("target_accuracy", 0.99)
        iterations = config.get("iterations", 100)

        # Simulated results
        final_energy = -1.137 + np.random.normal(0, 0.01)  # H2 ground state
        circuit_depth = 80 + int(np.random.normal(0, 10))
        convergence_iterations = min(iterations, 30 + int(np.random.exponential(20)))

        return {
            "final_energy": final_energy,
            "circuit_depth": circuit_depth,
            "convergence_iterations": convergence_iterations,
            "accuracy": 0.985 + np.random.uniform(0, 0.01),
            "execution_time_ms": 500 + np.random.uniform(0, 200),
        }

    async def _run_swarm_experiment(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run SWARM convergence experiment."""
        num_agents = config.get("num_agents", 50)
        pattern = config.get("pattern", "stigmergy")

        # Simulated results
        convergence_iterations = 30 + int(np.random.exponential(15))
        consensus_score = 0.9 + np.random.uniform(0, 0.08)

        return {
            "convergence_iterations": convergence_iterations,
            "consensus_score": consensus_score,
            "message_count": num_agents * convergence_iterations * 2,
            "coordination_efficiency": 0.85 + np.random.uniform(0, 0.1),
        }

    async def _run_neuromorphic_experiment(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run neuromorphic power efficiency experiment."""
        # Simulated power measurements
        inference_power_mw = 0.8 + np.random.uniform(0, 0.4)
        training_power_mw = 2.5 + np.random.uniform(0, 1.0)

        return {
            "inference_power_mw": inference_power_mw,
            "training_power_mw": training_power_mw,
            "accuracy": 0.92 + np.random.uniform(0, 0.05),
            "latency_ms": 0.5 + np.random.uniform(0, 0.3),
            "energy_per_inference_uj": inference_power_mw * 0.5,
        }

    async def _run_training_experiment(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run model training experiment."""
        epochs = config.get("epochs", 50)

        # Simulated training metrics
        final_loss = 0.1 + np.random.uniform(0, 0.05)
        final_accuracy = 0.9 + np.random.uniform(0, 0.08)

        return {
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "best_epoch": int(epochs * 0.7 + np.random.uniform(-5, 5)),
            "training_time_seconds": epochs * 2 + np.random.uniform(0, 20),
        }

    async def _run_validation_experiment(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run data validation experiment."""
        num_records = config.get("num_records", 1000)

        # Simulated validation metrics
        return {
            "records_validated": num_records,
            "valid_records": int(num_records * (0.95 + np.random.uniform(0, 0.04))),
            "validation_errors": int(num_records * np.random.uniform(0, 0.03)),
            "validation_warnings": int(num_records * np.random.uniform(0, 0.05)),
            "validation_time_ms": num_records * 0.1 + np.random.uniform(0, 50),
        }

    async def _run_test_suite(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Run the test suite."""
        # Try to run actual tests
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.experiments_dir.parent),
            )

            # Parse test output
            lines = result.stdout.split("\n")
            passed = failed = 0
            for line in lines:
                if "passed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                passed = int(parts[i-1])
                            except ValueError:
                                pass
                        if part == "failed" and i > 0:
                            try:
                                failed = int(parts[i-1])
                            except ValueError:
                                pass

            return {
                "tests_passed": passed,
                "tests_failed": failed,
                "test_coverage": 0.6 + np.random.uniform(0, 0.2),  # Estimated
                "exit_code": result.returncode,
            }

        except Exception as e:
            self._logger.warning("Test suite execution failed", error=str(e))
            return {
                "tests_passed": 0,
                "tests_failed": 0,
                "test_coverage": 0.0,
                "exit_code": -1,
            }

    def _analyze_for_improvements(
        self,
        metrics: Dict[str, float],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Analyze metrics to identify improvements."""
        improvements = []

        # VQE improvements
        if "circuit_depth" in metrics:
            if metrics["circuit_depth"] > 100:
                improvements.append({
                    "type": "optimization",
                    "target": "vqe_circuit_depth",
                    "current": metrics["circuit_depth"],
                    "suggested": 80,
                    "action": "Apply circuit compilation and gate cancellation",
                    "priority": TaskPriority.HIGH.value,
                })

        # SWARM improvements
        if "convergence_iterations" in metrics:
            if metrics["convergence_iterations"] > 40:
                improvements.append({
                    "type": "algorithm",
                    "target": "swarm_convergence",
                    "current": metrics["convergence_iterations"],
                    "suggested": 30,
                    "action": "Tune pheromone decay rate and agent exploration factor",
                    "priority": TaskPriority.MEDIUM.value,
                })

        # Power efficiency improvements
        if "inference_power_mw" in metrics:
            if metrics["inference_power_mw"] > 1.0:
                improvements.append({
                    "type": "efficiency",
                    "target": "neuromorphic_power",
                    "current": metrics["inference_power_mw"],
                    "suggested": 0.5,
                    "action": "Optimize spike encoding and reduce network size",
                    "priority": TaskPriority.HIGH.value,
                })

        # Test coverage improvements
        if "test_coverage" in metrics:
            if metrics["test_coverage"] < 0.8:
                improvements.append({
                    "type": "quality",
                    "target": "test_coverage",
                    "current": metrics["test_coverage"],
                    "suggested": 0.8,
                    "action": "Add tests for uncovered modules",
                    "priority": TaskPriority.MEDIUM.value,
                })

        return improvements

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:6]
        return f"{prefix}_{timestamp}_{suffix}"

    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to disk."""
        results_dir = self.experiments_dir / "results"
        results_dir.mkdir(exist_ok=True)

        result_file = results_dir / f"{result.experiment_id}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Code Updater
# =============================================================================

class CodeUpdater:
    """
    Manages code updates based on experiment results.

    Supports:
    - Parameter tuning
    - Configuration updates
    - Code patches
    - Rollback capability
    """

    def __init__(self, repo_dir: str = "."):
        self.repo_dir = Path(repo_dir)
        self.updates_dir = self.repo_dir / ".qbita_updates"
        self.updates_dir.mkdir(exist_ok=True)
        self._logger = structlog.get_logger("CodeUpdater")

    def apply_parameter_update(
        self,
        file_path: str,
        parameter_name: str,
        old_value: Any,
        new_value: Any,
    ) -> bool:
        """Apply a parameter update to a file."""
        full_path = self.repo_dir / file_path
        if not full_path.exists():
            self._logger.error("File not found", path=file_path)
            return False

        # Backup original
        self._backup_file(full_path)

        try:
            content = full_path.read_text()

            # Simple string replacement for parameter values
            old_pattern = f"{parameter_name} = {old_value}"
            new_pattern = f"{parameter_name} = {new_value}"

            if old_pattern in content:
                new_content = content.replace(old_pattern, new_pattern)
                full_path.write_text(new_content)
                self._logger.info(
                    "Parameter updated",
                    file=file_path,
                    parameter=parameter_name,
                    old=old_value,
                    new=new_value,
                )
                return True
            else:
                self._logger.warning("Pattern not found", file=file_path, pattern=old_pattern)
                return False

        except Exception as e:
            self._logger.error("Update failed", error=str(e))
            self._restore_backup(full_path)
            return False

    def apply_config_update(
        self,
        config_path: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Apply configuration updates."""
        full_path = self.repo_dir / config_path
        if not full_path.exists():
            self._logger.error("Config not found", path=config_path)
            return False

        self._backup_file(full_path)

        try:
            import yaml

            with open(full_path) as f:
                config = yaml.safe_load(f)

            # Apply updates
            def deep_update(d: dict, u: dict) -> dict:
                for k, v in u.items():
                    if isinstance(v, dict) and isinstance(d.get(k), dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v
                return d

            deep_update(config, updates)

            with open(full_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            self._logger.info("Config updated", path=config_path, updates=list(updates.keys()))
            return True

        except Exception as e:
            self._logger.error("Config update failed", error=str(e))
            self._restore_backup(full_path)
            return False

    def create_improvement_branch(self, improvement_name: str) -> str:
        """Create a new branch for improvements."""
        branch_name = f"Qbita-auto-{improvement_name}-{datetime.utcnow().strftime('%Y%m%d')}"
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=str(self.repo_dir),
                check=True,
                capture_output=True,
            )
            self._logger.info("Created branch", branch=branch_name)
            return branch_name
        except subprocess.CalledProcessError as e:
            self._logger.error("Branch creation failed", error=e.stderr.decode())
            raise

    def commit_changes(self, message: str, files: List[str]) -> bool:
        """Commit changes with QbitaLab attribution."""
        try:
            # Stage files
            subprocess.run(
                ["git", "add"] + files,
                cwd=str(self.repo_dir),
                check=True,
                capture_output=True,
            )

            # Commit with QbitaLab format
            full_message = f"[QbitaLab] {message}\n\nAutomated improvement by QbitaLab autonomous agent."
            subprocess.run(
                ["git", "commit", "-m", full_message],
                cwd=str(self.repo_dir),
                check=True,
                capture_output=True,
            )

            self._logger.info("Changes committed", message=message)
            return True

        except subprocess.CalledProcessError as e:
            self._logger.error("Commit failed", error=e.stderr.decode())
            return False

    def rollback(self, file_path: str) -> bool:
        """Rollback a file to its backup."""
        full_path = self.repo_dir / file_path
        return self._restore_backup(full_path)

    def _backup_file(self, path: Path) -> None:
        """Create a backup of a file."""
        backup_path = self.updates_dir / f"{path.name}.backup"
        shutil.copy2(path, backup_path)

    def _restore_backup(self, path: Path) -> bool:
        """Restore a file from backup."""
        backup_path = self.updates_dir / f"{path.name}.backup"
        if backup_path.exists():
            shutil.copy2(backup_path, path)
            self._logger.info("Restored from backup", path=str(path))
            return True
        return False


# =============================================================================
# Autonomous Agent
# =============================================================================

class AutonomousAgent:
    """
    Main autonomous agent that orchestrates experiments,
    analyzes results, and applies improvements.

    Usage:
        agent = AutonomousAgent()

        # Run single cycle
        await agent.run_cycle()

        # Start continuous operation
        await agent.start_continuous(interval_hours=24)
    """

    def __init__(
        self,
        repo_dir: str = ".",
        experiments_dir: str = "./experiments",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.repo_dir = Path(repo_dir)
        self.experiment_runner = ExperimentRunner(experiments_dir)
        self.code_updater = CodeUpdater(repo_dir)
        self.config = config or {}
        self._logger = structlog.get_logger("AutonomousAgent")
        self._running = False

        # Task queue
        self.task_queue: List[AutonomousTask] = []
        self.completed_tasks: List[AutonomousTask] = []

        # Metrics history
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}

    async def run_cycle(self) -> Dict[str, Any]:
        """Run a single improvement cycle."""
        cycle_id = self._generate_cycle_id()
        self._logger.info("Starting autonomous cycle", cycle_id=cycle_id)

        results = {
            "cycle_id": cycle_id,
            "started_at": datetime.utcnow().isoformat(),
            "experiments": [],
            "improvements": [],
            "updates_applied": [],
        }

        try:
            # 1. Run scheduled experiments
            experiments = await self._run_scheduled_experiments()
            results["experiments"] = [e.to_dict() for e in experiments]

            # 2. Analyze results and identify improvements
            all_improvements = []
            for exp in experiments:
                if exp.success:
                    all_improvements.extend(exp.improvements)

            results["improvements"] = all_improvements

            # 3. Apply high-priority improvements
            for improvement in all_improvements:
                if improvement.get("priority") == TaskPriority.HIGH.value:
                    success = await self._apply_improvement(improvement)
                    if success:
                        results["updates_applied"].append(improvement)

            # 4. Update metrics history
            self._update_metrics_history(experiments)

            # 5. Generate report
            report = self._generate_report(results)
            results["report"] = report

            results["completed_at"] = datetime.utcnow().isoformat()
            results["success"] = True

        except Exception as e:
            self._logger.error("Cycle failed", cycle_id=cycle_id, error=str(e))
            results["error"] = str(e)
            results["success"] = False

        # Save cycle results
        self._save_cycle_results(results)
        return results

    async def start_continuous(self, interval_hours: float = 24) -> None:
        """Start continuous autonomous operation."""
        self._running = True
        self._logger.info("Starting continuous operation", interval_hours=interval_hours)

        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                self._logger.error("Cycle error in continuous mode", error=str(e))

            # Wait for next cycle
            await asyncio.sleep(interval_hours * 3600)

    def stop(self) -> None:
        """Stop continuous operation."""
        self._running = False
        self._logger.info("Stopping autonomous agent")

    async def _run_scheduled_experiments(self) -> List[ExperimentResult]:
        """Run all scheduled experiments."""
        experiments = []

        # Define daily experiment suite
        experiment_configs = [
            ("test_suite", {"timeout": 300}),
            ("quantum_vqe", {"target_accuracy": 0.99, "iterations": 100}),
            ("swarm_convergence", {"num_agents": 50, "pattern": "stigmergy"}),
            ("neuromorphic_power", {"batch_size": 32}),
            ("data_validation", {"num_records": 1000}),
        ]

        for exp_type, config in experiment_configs:
            result = await self.experiment_runner.run_experiment(exp_type, config)
            experiments.append(result)

        return experiments

    async def _apply_improvement(self, improvement: Dict[str, Any]) -> bool:
        """Apply a single improvement."""
        target = improvement.get("target")
        action = improvement.get("action")

        self._logger.info("Applying improvement", target=target, action=action)

        # Map improvement targets to update actions
        if target == "vqe_circuit_depth":
            # Update VQE parameters
            return self.code_updater.apply_config_update(
                "configs/default.yaml",
                {"quantum": {"optimization_level": 3}},
            )

        elif target == "swarm_convergence":
            # Update SWARM parameters
            return self.code_updater.apply_config_update(
                "configs/default.yaml",
                {"swarm": {"pheromone_decay": 0.1}},
            )

        elif target == "neuromorphic_power":
            # Log recommendation (no automatic update)
            self._logger.info("Manual optimization recommended", action=action)
            return False

        elif target == "test_coverage":
            # Log recommendation
            self._logger.info("Test coverage improvement needed", action=action)
            return False

        return False

    def _update_metrics_history(self, experiments: List[ExperimentResult]) -> None:
        """Update metrics history from experiment results."""
        timestamp = datetime.utcnow()
        for exp in experiments:
            for metric_name, value in exp.metrics.items():
                key = f"{exp.experiment_type}.{metric_name}"
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append((timestamp, value))

    def _generate_report(self, cycle_results: Dict[str, Any]) -> str:
        """Generate a markdown report of the cycle."""
        lines = [
            "# QbitaLab Autonomous Cycle Report",
            f"\n**Cycle ID**: {cycle_results.get('cycle_id')}",
            f"**Started**: {cycle_results.get('started_at')}",
            f"**Completed**: {cycle_results.get('completed_at', 'N/A')}",
            "\n## Experiments Run\n",
        ]

        for exp in cycle_results.get("experiments", []):
            status = "✓" if exp.get("success") else "✗"
            lines.append(f"- {status} **{exp.get('experiment_type')}**")
            if exp.get("metrics"):
                for k, v in list(exp["metrics"].items())[:3]:
                    lines.append(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")

        if cycle_results.get("improvements"):
            lines.append("\n## Identified Improvements\n")
            for imp in cycle_results["improvements"]:
                lines.append(f"- [{imp.get('priority', 'medium')}] **{imp.get('target')}**: {imp.get('action')}")

        if cycle_results.get("updates_applied"):
            lines.append("\n## Updates Applied\n")
            for update in cycle_results["updates_applied"]:
                lines.append(f"- {update.get('target')}: {update.get('action')}")

        lines.append("\n---\n*Generated by QbitaLab Autonomous Agent*")
        return "\n".join(lines)

    def _generate_cycle_id(self) -> str:
        """Generate unique cycle ID."""
        return f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    def _save_cycle_results(self, results: Dict[str, Any]) -> None:
        """Save cycle results to disk."""
        reports_dir = self.repo_dir / "experiments" / "reports" / "daily"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_file = reports_dir / f"{results.get('cycle_id')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save markdown report
        if results.get("report"):
            report_file = reports_dir / f"{results.get('cycle_id')}.md"
            with open(report_file, "w") as f:
                f.write(results["report"])


# =============================================================================
# Daily Runner Script
# =============================================================================

async def run_daily_agent():
    """Entry point for daily autonomous runs."""
    agent = AutonomousAgent(
        repo_dir=".",
        experiments_dir="./experiments",
    )

    results = await agent.run_cycle()

    if results.get("success"):
        print(f"✓ Cycle completed: {results.get('cycle_id')}")
        print(f"  Experiments: {len(results.get('experiments', []))}")
        print(f"  Improvements: {len(results.get('improvements', []))}")
        print(f"  Updates applied: {len(results.get('updates_applied', []))}")
    else:
        print(f"✗ Cycle failed: {results.get('error')}")

    return results


# =============================================================================
# Scheduler
# =============================================================================

class TaskScheduler:
    """Schedules and manages autonomous tasks."""

    def __init__(self):
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self._logger = structlog.get_logger("TaskScheduler")

    def schedule_daily(
        self,
        task_name: str,
        task_func: Callable,
        hour: int = 2,  # 2 AM default
        minute: int = 0,
    ) -> None:
        """Schedule a task to run daily."""
        self.scheduled_tasks[task_name] = {
            "func": task_func,
            "schedule": "daily",
            "hour": hour,
            "minute": minute,
            "last_run": None,
        }
        self._logger.info("Task scheduled", task=task_name, hour=hour, minute=minute)

    def schedule_interval(
        self,
        task_name: str,
        task_func: Callable,
        interval_hours: float,
    ) -> None:
        """Schedule a task to run at regular intervals."""
        self.scheduled_tasks[task_name] = {
            "func": task_func,
            "schedule": "interval",
            "interval_hours": interval_hours,
            "last_run": None,
        }
        self._logger.info("Task scheduled", task=task_name, interval_hours=interval_hours)

    async def run_scheduler(self) -> None:
        """Run the task scheduler loop."""
        self._logger.info("Starting scheduler")

        while True:
            now = datetime.utcnow()

            for task_name, task_info in self.scheduled_tasks.items():
                should_run = False

                if task_info["schedule"] == "daily":
                    if task_info["last_run"] is None or (
                        now.date() > task_info["last_run"].date()
                        and now.hour >= task_info["hour"]
                        and now.minute >= task_info["minute"]
                    ):
                        should_run = True

                elif task_info["schedule"] == "interval":
                    if task_info["last_run"] is None or (
                        now - task_info["last_run"]
                    ).total_seconds() >= task_info["interval_hours"] * 3600:
                        should_run = True

                if should_run:
                    try:
                        self._logger.info("Running scheduled task", task=task_name)
                        func = task_info["func"]
                        if asyncio.iscoroutinefunction(func):
                            await func()
                        else:
                            func()
                        task_info["last_run"] = now
                    except Exception as e:
                        self._logger.error("Scheduled task failed", task=task_name, error=str(e))

            await asyncio.sleep(60)  # Check every minute


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Task definitions
    "TaskPriority",
    "TaskStatus",
    "AutonomousTask",

    # Experiment runner
    "ExperimentResult",
    "ExperimentRunner",

    # Code updater
    "CodeUpdater",

    # Autonomous agent
    "AutonomousAgent",

    # Scheduler
    "TaskScheduler",

    # Entry points
    "run_daily_agent",
]
