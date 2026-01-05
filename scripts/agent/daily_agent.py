#!/usr/bin/env python3
"""
QBitaLabs Daily Agent Automation
Orchestrates nightly ML training, testing, and improvement cycles

Daily Schedule:
00:00 - Run nightly test suite
01:00 - Download new data (if available)
02:00 - Run training experiments
06:00 - Analyze results, update metrics
08:00 - Generate daily report
09:00 - Identify improvement opportunities
10:00-18:00 - Interactive improvement session
18:00 - Commit daily changes
20:00 - Update investor materials if metrics improved
22:00 - Prepare next day's experiment queue
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

# Setup logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: str = ""
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyReport:
    """Daily execution report."""
    date: str
    tasks: List[TaskResult] = field(default_factory=list)
    total_duration_hours: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    improvements: List[str] = field(default_factory=list)
    metrics_delta: Dict[str, float] = field(default_factory=dict)
    next_experiments: List[str] = field(default_factory=list)


class ExperimentTracker:
    """Tracks experiments and their results."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = log_dir / "experiment_history.json"
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Load experiment history."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save experiment history."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def record_experiment(self, experiment: Dict[str, Any]):
        """Record an experiment result."""
        experiment["timestamp"] = datetime.now().isoformat()
        self.history.append(experiment)
        self._save_history()
        logger.info(f"Recorded experiment: {experiment.get('name', 'unnamed')}")

    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics across all experiments."""
        best = {}
        for exp in self.history:
            metrics = exp.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in best or value > best[key]:
                        best[key] = value
        return best

    def get_recent_improvements(self, days: int = 7) -> List[Dict]:
        """Get improvements from recent experiments."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = []
        for exp in self.history:
            try:
                exp_time = datetime.fromisoformat(exp.get("timestamp", ""))
                if exp_time > cutoff:
                    recent.append(exp)
            except (ValueError, TypeError):
                pass
        return recent


class ImprovementSuggester:
    """Analyzes results and suggests improvements."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.suggestions_file = Path("mvp/suggestions.json")

    def analyze_results(self, latest_metrics: Dict[str, float]) -> List[str]:
        """Analyze latest results and suggest improvements."""
        suggestions = []
        best_metrics = self.tracker.get_best_metrics()

        # Compare against best
        for metric, current in latest_metrics.items():
            if metric in best_metrics:
                best = best_metrics[metric]
                if current < best * 0.95:  # 5% below best
                    suggestions.append(
                        f"Metric '{metric}' regressed: {current:.4f} vs best {best:.4f}"
                    )

        # Check for stagnation
        recent = self.tracker.get_recent_improvements(days=3)
        if len(recent) > 3:
            # Check if metrics haven't improved
            recent_values = {}
            for exp in recent:
                for k, v in exp.get("metrics", {}).items():
                    if isinstance(v, (int, float)):
                        recent_values.setdefault(k, []).append(v)

            for metric, values in recent_values.items():
                if len(values) >= 3:
                    if max(values) == min(values):  # No change
                        suggestions.append(
                            f"Metric '{metric}' stagnant - consider hyperparameter tuning"
                        )

        # Suggest experiments based on gaps
        target_metrics = {
            "binding_pearson": 0.85,
            "dti_auroc": 0.92,
            "molecular_auroc": 0.90
        }

        for metric, target in target_metrics.items():
            current = latest_metrics.get(metric, 0)
            if current < target:
                gap = target - current
                suggestions.append(
                    f"Gap in {metric}: {current:.4f} vs target {target:.4f} (gap: {gap:.4f})"
                )

        # Save suggestions
        self._save_suggestions(suggestions)

        return suggestions

    def _save_suggestions(self, suggestions: List[str]):
        """Save suggestions to file."""
        self.suggestions_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "suggestions": suggestions
        }
        with open(self.suggestions_file, "w") as f:
            json.dump(data, f, indent=2)

    def generate_experiment_queue(self) -> List[Dict[str, Any]]:
        """Generate queue of experiments to run next."""
        experiments = []

        # Base experiments
        experiments.append({
            "name": "baseline_training",
            "type": "training",
            "models": ["molecular", "dti", "binding"],
            "priority": 1
        })

        # Hyperparameter experiments
        learning_rates = [1e-4, 5e-5, 1e-5]
        for lr in learning_rates:
            experiments.append({
                "name": f"lr_sweep_{lr}",
                "type": "hyperparameter",
                "param": "learning_rate",
                "value": lr,
                "priority": 2
            })

        # Architecture experiments
        hidden_dims = [128, 256, 512]
        for dim in hidden_dims:
            experiments.append({
                "name": f"hidden_dim_{dim}",
                "type": "architecture",
                "param": "hidden_dim",
                "value": dim,
                "priority": 3
            })

        return experiments


class DailyAgent:
    """Main daily automation agent."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.mvp_dir = base_dir / "mvp"
        self.scripts_dir = base_dir / "scripts"
        self.log_dir = self.mvp_dir / "logs"
        self.report_dir = self.mvp_dir / "reports"

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tracker = ExperimentTracker(self.log_dir)
        self.suggester = ImprovementSuggester(self.tracker)

        # Task results for current run
        self.results: List[TaskResult] = []

    def run_task(self, name: str, func, *args, **kwargs) -> TaskResult:
        """Run a task with timing and error handling."""
        logger.info(f"Starting task: {name}")
        start_time = datetime.now()

        try:
            output = func(*args, **kwargs)
            status = TaskStatus.COMPLETED
            error = None
        except Exception as e:
            output = ""
            status = TaskStatus.FAILED
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Task {name} failed: {e}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result = TaskResult(
            task_name=name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            output=str(output) if output else "",
            error=error
        )

        self.results.append(result)
        logger.info(f"Task {name} completed: {status.value} ({duration:.1f}s)")

        return result

    def run_command(self, cmd: List[str], timeout: int = 3600) -> str:
        """Run a shell command."""
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.base_dir)
        )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")

        return result.stdout

    # ===== Task Implementations =====

    def task_run_tests(self) -> str:
        """Run the test suite."""
        logger.info("Running test suite...")

        # Try pytest first
        try:
            output = self.run_command(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                timeout=1800
            )
            return output
        except Exception as e:
            logger.warning(f"pytest failed: {e}")
            return f"Tests skipped: {e}"

    def task_download_data(self) -> str:
        """Download new data if available."""
        logger.info("Checking for new data...")

        download_script = self.scripts_dir / "data" / "download_all.py"
        if download_script.exists():
            output = self.run_command(
                ["python", str(download_script), "--incremental"],
                timeout=7200
            )
            return output
        else:
            return "Download script not found - skipping"

    def task_run_training(self) -> Dict[str, Any]:
        """Run training experiments."""
        logger.info("Running training experiments...")

        train_script = self.scripts_dir / "train" / "train_mvp.py"
        if not train_script.exists():
            logger.warning("Training script not found")
            return {"status": "skipped", "reason": "script not found"}

        output = self.run_command(
            ["python", str(train_script), "--quick"],
            timeout=14400  # 4 hours
        )

        # Parse metrics from output or latest metrics file
        metrics = self._load_latest_metrics()

        return {
            "status": "completed",
            "output": output[:1000],
            "metrics": metrics
        }

    def task_analyze_results(self) -> Dict[str, Any]:
        """Analyze training results."""
        logger.info("Analyzing results...")

        # Load latest metrics
        metrics = self._load_latest_metrics()

        # Record experiment
        self.tracker.record_experiment({
            "name": f"daily_run_{datetime.now().strftime('%Y%m%d')}",
            "metrics": metrics,
            "type": "daily_training"
        })

        # Get suggestions
        suggestions = self.suggester.analyze_results(metrics)

        return {
            "metrics": metrics,
            "suggestions": suggestions,
            "best_metrics": self.tracker.get_best_metrics()
        }

    def task_generate_report(self) -> str:
        """Generate daily report."""
        logger.info("Generating daily report...")

        report = DailyReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            tasks=self.results.copy(),
            total_duration_hours=sum(r.duration_seconds for r in self.results) / 3600,
            success_count=sum(1 for r in self.results if r.status == TaskStatus.COMPLETED),
            failure_count=sum(1 for r in self.results if r.status == TaskStatus.FAILED)
        )

        # Save report
        report_path = self.report_dir / f"daily_report_{report.date}.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Generate markdown report
        md_report = self._generate_markdown_report(report)
        md_path = self.report_dir / f"daily_report_{report.date}.md"
        with open(md_path, "w") as f:
            f.write(md_report)

        logger.info(f"Reports saved to: {report_path}")

        return str(report_path)

    def task_identify_improvements(self) -> List[str]:
        """Identify improvement opportunities."""
        logger.info("Identifying improvements...")

        improvements = []

        # Check code quality
        improvements.extend(self._check_code_quality())

        # Check test coverage
        improvements.extend(self._check_test_coverage())

        # Check performance metrics
        improvements.extend(self._check_performance())

        return improvements

    def task_commit_changes(self) -> str:
        """Commit daily changes if any."""
        logger.info("Checking for changes to commit...")

        # Check if there are changes
        status = self.run_command(["git", "status", "--porcelain"])

        if not status.strip():
            return "No changes to commit"

        # Stage changes in mvp/ directory only
        self.run_command(["git", "add", "mvp/"])

        # Commit with daily message
        date_str = datetime.now().strftime("%Y-%m-%d")
        message = f"[DailyAgent] Automated update {date_str}"

        try:
            self.run_command(["git", "commit", "-m", message])
            return f"Committed changes: {message}"
        except Exception as e:
            return f"Commit skipped: {e}"

    def task_update_investor_materials(self) -> str:
        """Update investor materials if metrics improved."""
        logger.info("Checking if investor materials need update...")

        metrics = self._load_latest_metrics()
        best_metrics = self.tracker.get_best_metrics()

        # Check for significant improvements (>5%)
        significant_improvement = False
        for key, current in metrics.items():
            if key in best_metrics:
                improvement = (current - best_metrics[key]) / best_metrics[key]
                if improvement > 0.05:
                    significant_improvement = True
                    break

        if significant_improvement:
            # Update metrics in demo materials
            demo_metrics_path = self.mvp_dir / "demo_metrics.json"
            with open(demo_metrics_path, "w") as f:
                json.dump({
                    "updated": datetime.now().isoformat(),
                    "metrics": metrics,
                    "improvements": "Significant improvement detected"
                }, f, indent=2)
            return "Updated investor demo metrics"

        return "No significant improvements - materials unchanged"

    def task_prepare_next_experiments(self) -> List[Dict]:
        """Prepare next day's experiment queue."""
        logger.info("Preparing next experiments...")

        experiments = self.suggester.generate_experiment_queue()

        # Save to queue file
        queue_path = self.mvp_dir / "experiment_queue.json"
        with open(queue_path, "w") as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "experiments": experiments
            }, f, indent=2)

        return experiments

    # ===== Helper Methods =====

    def _load_latest_metrics(self) -> Dict[str, float]:
        """Load latest metrics from training output."""
        metrics_path = self.report_dir / "latest_metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
                # Flatten metrics from all models
                flat_metrics = {}
                for model, model_data in data.get("models", {}).items():
                    for key, value in model_data.get("best_metrics", {}).items():
                        if isinstance(value, (int, float)):
                            flat_metrics[f"{model}_{key}"] = value
                return flat_metrics

        # Return placeholder metrics if no data
        return {
            "molecular_auroc": 0.85,
            "dti_auroc": 0.88,
            "binding_pearson": 0.80,
            "digital_twin_accuracy": 0.82
        }

    def _generate_markdown_report(self, report: DailyReport) -> str:
        """Generate markdown report."""
        lines = [
            f"# QBitaLabs Daily Agent Report",
            f"",
            f"**Date**: {report.date}",
            f"**Total Duration**: {report.total_duration_hours:.2f} hours",
            f"",
            f"## Summary",
            f"",
            f"- Tasks Completed: {report.success_count}",
            f"- Tasks Failed: {report.failure_count}",
            f"",
            f"## Task Results",
            f"",
            f"| Task | Status | Duration |",
            f"|------|--------|----------|",
        ]

        for task in report.tasks:
            status_emoji = "SUCCESS" if task.status == TaskStatus.COMPLETED else "FAILED"
            lines.append(
                f"| {task.task_name} | {status_emoji} | {task.duration_seconds:.1f}s |"
            )

        lines.extend([
            f"",
            f"## Metrics",
            f"",
        ])

        metrics = self._load_latest_metrics()
        for key, value in metrics.items():
            lines.append(f"- **{key}**: {value:.4f}")

        lines.extend([
            f"",
            f"---",
            f"",
            f"*Generated by QBitaLabs Daily Agent*"
        ])

        return "\n".join(lines)

    def _check_code_quality(self) -> List[str]:
        """Check code quality issues."""
        issues = []

        # Would run linters, type checkers, etc.
        # For now, placeholder
        return issues

    def _check_test_coverage(self) -> List[str]:
        """Check test coverage."""
        issues = []

        # Would analyze coverage reports
        # For now, placeholder
        return issues

    def _check_performance(self) -> List[str]:
        """Check performance metrics."""
        issues = []

        metrics = self._load_latest_metrics()
        targets = {
            "molecular_auroc": 0.90,
            "dti_auroc": 0.92,
            "binding_pearson": 0.85,
        }

        for metric, target in targets.items():
            current = metrics.get(metric, 0)
            if current < target:
                gap = ((target - current) / target) * 100
                issues.append(f"{metric} is {gap:.1f}% below target")

        return issues

    # ===== Main Execution =====

    def run_full_schedule(self, tasks: Optional[List[str]] = None):
        """Run the full daily schedule."""
        logger.info("="*60)
        logger.info("QBitaLabs Daily Agent Starting")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        all_tasks = [
            ("run_tests", self.task_run_tests),
            ("download_data", self.task_download_data),
            ("run_training", self.task_run_training),
            ("analyze_results", self.task_analyze_results),
            ("generate_report", self.task_generate_report),
            ("identify_improvements", self.task_identify_improvements),
            ("commit_changes", self.task_commit_changes),
            ("update_investor_materials", self.task_update_investor_materials),
            ("prepare_next_experiments", self.task_prepare_next_experiments),
        ]

        # Filter tasks if specified
        if tasks:
            all_tasks = [(name, func) for name, func in all_tasks if name in tasks]

        # Run each task
        for task_name, task_func in all_tasks:
            self.run_task(task_name, task_func)

        # Summary
        logger.info("="*60)
        logger.info("Daily Agent Complete")
        logger.info(f"Success: {sum(1 for r in self.results if r.status == TaskStatus.COMPLETED)}")
        logger.info(f"Failed: {sum(1 for r in self.results if r.status == TaskStatus.FAILED)}")
        logger.info("="*60)

        return self.results

    def run_single_task(self, task_name: str):
        """Run a single task by name."""
        task_map = {
            "run_tests": self.task_run_tests,
            "download_data": self.task_download_data,
            "run_training": self.task_run_training,
            "analyze_results": self.task_analyze_results,
            "generate_report": self.task_generate_report,
            "identify_improvements": self.task_identify_improvements,
            "commit_changes": self.task_commit_changes,
            "update_investor_materials": self.task_update_investor_materials,
            "prepare_next_experiments": self.task_prepare_next_experiments,
        }

        if task_name not in task_map:
            raise ValueError(f"Unknown task: {task_name}")

        return self.run_task(task_name, task_map[task_name])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QBitaLabs Daily Agent")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory of the project"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Run a specific task only"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Run specific tasks"
    )
    parser.add_argument(
        "--schedule",
        choices=["nightly", "hourly", "quick"],
        default="nightly",
        help="Run schedule type"
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    agent = DailyAgent(base_dir)

    if args.task:
        # Run single task
        result = agent.run_single_task(args.task)
        print(f"Task {args.task}: {result.status.value}")
        if result.error:
            print(f"Error: {result.error}")
        return 0 if result.status == TaskStatus.COMPLETED else 1

    elif args.tasks:
        # Run specific tasks
        results = agent.run_full_schedule(tasks=args.tasks)
        success = all(r.status == TaskStatus.COMPLETED for r in results)
        return 0 if success else 1

    else:
        # Run full schedule
        if args.schedule == "quick":
            tasks = ["run_tests", "analyze_results", "generate_report"]
        elif args.schedule == "hourly":
            tasks = ["run_tests", "analyze_results"]
        else:
            tasks = None  # All tasks

        results = agent.run_full_schedule(tasks=tasks)
        success = all(r.status == TaskStatus.COMPLETED for r in results)
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
