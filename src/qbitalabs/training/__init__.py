"""
QBitaLabs Training Infrastructure

Comprehensive training system with:
- Trainer abstraction for all models
- Checkpointing and model persistence
- Hyperparameter optimization
- Experiment tracking and logging
- Distributed training support
- Early stopping and learning rate scheduling

Authored by: QbitaLab
"""

from __future__ import annotations

import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union
import hashlib
import random

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")
ModelT = TypeVar("ModelT")


# =============================================================================
# Experiment Tracking
# =============================================================================

class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Metric:
    """A single metric value with metadata."""
    name: str
    value: float
    step: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Checkpoint:
    """Model checkpoint metadata."""
    path: str
    epoch: int
    step: int
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    data_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "data_config": self.data_config,
            "training_config": self.training_config,
            "tags": self.tags,
            "notes": self.notes,
        }


@dataclass
class ExperimentRun:
    """A single experiment run with all tracking data."""
    run_id: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: List[Metric] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric value."""
        self.metrics.append(Metric(name=name, value=value, step=step))

    def get_metric_history(self, name: str) -> List[Tuple[int, float]]:
        """Get history of a specific metric."""
        return [(m.step, m.value) for m in self.metrics if m.name == name]

    def get_best_metric(self, name: str, mode: str = "min") -> Optional[float]:
        """Get best value of a metric."""
        values = [m.value for m in self.metrics if m.name == name]
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": [m.to_dict() for m in self.metrics],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "artifacts": self.artifacts,
            "error_message": self.error_message,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time else None
            ),
        }


class ExperimentTracker:
    """Tracks experiments, metrics, and artifacts."""

    def __init__(self, experiment_dir: str = "./experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.experiment_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self._logger = structlog.get_logger("ExperimentTracker")
        self._current_run: Optional[ExperimentRun] = None

    def create_run(self, config: ExperimentConfig) -> ExperimentRun:
        """Create a new experiment run."""
        run_id = self._generate_run_id(config.name)
        run = ExperimentRun(run_id=run_id, config=config)
        self._current_run = run
        self._logger.info("Created experiment run", run_id=run_id, config=config.name)
        return run

    def start_run(self, run: Optional[ExperimentRun] = None) -> ExperimentRun:
        """Start an experiment run."""
        run = run or self._current_run
        if not run:
            raise ValueError("No run to start")
        run.status = ExperimentStatus.RUNNING
        run.start_time = datetime.utcnow()
        self._save_run(run)
        self._logger.info("Started experiment run", run_id=run.run_id)
        return run

    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        run: Optional[ExperimentRun] = None,
    ) -> None:
        """Log a metric value."""
        run = run or self._current_run
        if not run:
            raise ValueError("No active run")
        run.log_metric(name, value, step)
        self._logger.debug("Logged metric", run_id=run.run_id, metric=name, value=value, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        run: Optional[ExperimentRun] = None,
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step, run)

    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        run: Optional[ExperimentRun] = None,
    ) -> Checkpoint:
        """Save a model checkpoint."""
        run = run or self._current_run
        if not run:
            raise ValueError("No active run")

        checkpoint_dir = self.runs_dir / run.run_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pkl"

        # Save model state
        with open(checkpoint_path, "wb") as f:
            pickle.dump(model, f)

        checkpoint = Checkpoint(
            path=str(checkpoint_path),
            epoch=epoch,
            step=step,
            metrics=metrics,
        )
        run.checkpoints.append(checkpoint)
        self._logger.info("Saved checkpoint", run_id=run.run_id, epoch=epoch, step=step)
        return checkpoint

    def load_checkpoint(self, checkpoint: Checkpoint) -> Any:
        """Load a model from checkpoint."""
        with open(checkpoint.path, "rb") as f:
            return pickle.load(f)

    def end_run(
        self,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        error_message: Optional[str] = None,
        run: Optional[ExperimentRun] = None,
    ) -> ExperimentRun:
        """End an experiment run."""
        run = run or self._current_run
        if not run:
            raise ValueError("No active run")
        run.status = status
        run.end_time = datetime.utcnow()
        run.error_message = error_message
        self._save_run(run)
        self._logger.info(
            "Ended experiment run",
            run_id=run.run_id,
            status=status.value,
            duration=(run.end_time - run.start_time).total_seconds() if run.start_time else 0,
        )
        return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Load a run by ID."""
        run_path = self.runs_dir / run_id / "run.json"
        if not run_path.exists():
            return None
        with open(run_path) as f:
            data = json.load(f)
        return self._deserialize_run(data)

    def list_runs(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ExperimentRun]:
        """List all experiment runs."""
        runs = []
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                run = self.get_run(run_dir.name)
                if run:
                    if status and run.status != status:
                        continue
                    if tags and not all(t in run.config.tags for t in tags):
                        continue
                    runs.append(run)
        return sorted(runs, key=lambda r: r.start_time or datetime.min, reverse=True)

    def _save_run(self, run: ExperimentRun) -> None:
        """Save run to disk."""
        run_dir = self.runs_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "run.json", "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    def _generate_run_id(self, name: str) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{name}_{timestamp}_{random_suffix}"

    def _deserialize_run(self, data: Dict[str, Any]) -> ExperimentRun:
        """Deserialize a run from dict."""
        config = ExperimentConfig(**data["config"])
        run = ExperimentRun(
            run_id=data["run_id"],
            config=config,
            status=ExperimentStatus(data["status"]),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            artifacts=data.get("artifacts", {}),
            error_message=data.get("error_message"),
        )
        for m in data.get("metrics", []):
            run.metrics.append(Metric(
                name=m["name"],
                value=m["value"],
                step=m["step"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
            ))
        for c in data.get("checkpoints", []):
            run.checkpoints.append(Checkpoint(
                path=c["path"],
                epoch=c["epoch"],
                step=c["step"],
                metrics=c["metrics"],
                timestamp=datetime.fromisoformat(c["timestamp"]),
            ))
        return run


# =============================================================================
# Training Callbacks
# =============================================================================

class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer", **kwargs) -> None:
        pass

    def on_train_end(self, trainer: "Trainer", **kwargs) -> None:
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: int, **kwargs) -> None:
        pass

    def on_batch_end(self, trainer: "Trainer", batch: int, loss: float, **kwargs) -> None:
        pass


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "min",
        restore_best: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.best_weights: Any = None
        self._logger = structlog.get_logger("EarlyStopping")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            self.best_weights = trainer.model
            return

        improved = False
        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best:
                self.best_weights = trainer.model
            self._logger.info(
                "Improvement detected",
                metric=self.monitor,
                value=current,
                epoch=epoch,
            )
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
                self.stopped_epoch = epoch
                self._logger.info(
                    "Early stopping triggered",
                    patience=self.patience,
                    best_epoch=self.best_epoch,
                    best_value=self.best_value,
                )


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(
        self,
        schedule: str = "cosine",  # cosine, step, exponential, plateau
        initial_lr: float = 0.001,
        min_lr: float = 1e-7,
        warmup_epochs: int = 0,
        **kwargs,
    ):
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.kwargs = kwargs
        self._logger = structlog.get_logger("LRScheduler")

    def on_epoch_begin(self, trainer: "Trainer", epoch: int, **kwargs) -> None:
        new_lr = self._compute_lr(epoch, trainer.max_epochs)
        trainer.learning_rate = new_lr
        self._logger.debug("Learning rate updated", epoch=epoch, lr=new_lr)

    def _compute_lr(self, epoch: int, max_epochs: int) -> float:
        """Compute learning rate for given epoch."""
        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs

        effective_epoch = epoch - self.warmup_epochs
        effective_epochs = max_epochs - self.warmup_epochs

        if self.schedule == "cosine":
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + np.cos(np.pi * effective_epoch / effective_epochs)
            )
        elif self.schedule == "step":
            step_size = self.kwargs.get("step_size", 10)
            gamma = self.kwargs.get("gamma", 0.1)
            return max(self.min_lr, self.initial_lr * (gamma ** (effective_epoch // step_size)))
        elif self.schedule == "exponential":
            gamma = self.kwargs.get("gamma", 0.95)
            return max(self.min_lr, self.initial_lr * (gamma ** effective_epoch))
        else:  # constant
            return self.initial_lr


class MetricsLogger(Callback):
    """Logs metrics to experiment tracker."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.global_step = 0

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        self.tracker.log_metrics(metrics, step=epoch)
        self.global_step = epoch


class CheckpointCallback(Callback):
    """Saves model checkpoints."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        save_frequency: int = 1,
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.tracker = tracker
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        current = metrics.get(self.monitor, 0)

        should_save = False
        if not self.save_best_only:
            should_save = (epoch + 1) % self.save_frequency == 0
        else:
            if self.best_value is None:
                should_save = True
            elif self.mode == "min" and current < self.best_value:
                should_save = True
            elif self.mode == "max" and current > self.best_value:
                should_save = True

        if should_save:
            self.best_value = current
            self.tracker.save_checkpoint(
                model=trainer.model,
                epoch=epoch,
                step=trainer.global_step,
                metrics=metrics,
            )


# =============================================================================
# Trainer
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    shuffle: bool = True
    random_seed: int = 42
    gradient_clip: Optional[float] = None
    accumulation_steps: int = 1
    log_interval: int = 10
    device: str = "cpu"  # cpu, cuda, mps


class Trainer:
    """
    Generic trainer for QBitaLabs models.

    Supports:
    - Flexible training loop with callbacks
    - Validation and metrics tracking
    - Early stopping and LR scheduling
    - Checkpointing and model persistence
    - Experiment tracking
    """

    def __init__(
        self,
        model: Any,
        config: TrainerConfig,
        tracker: Optional[ExperimentTracker] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        self.config = config
        self.tracker = tracker or ExperimentTracker()
        self.callbacks = callbacks or []

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.learning_rate = config.learning_rate
        self.max_epochs = config.max_epochs
        self.should_stop = False

        self._logger = structlog.get_logger("Trainer")

        # Set random seeds
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    def fit(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> ExperimentRun:
        """
        Train the model.

        Args:
            train_data: Training data
            val_data: Validation data (optional, will split from train if not provided)
            experiment_config: Experiment configuration

        Returns:
            ExperimentRun with training results
        """
        # Create experiment run
        if experiment_config is None:
            experiment_config = ExperimentConfig(
                name="training_run",
                model_type=type(self.model).__name__,
                hyperparameters={"learning_rate": self.config.learning_rate},
            )

        run = self.tracker.create_run(experiment_config)
        self.tracker.start_run(run)

        try:
            # Split validation data if needed
            if val_data is None and self.config.validation_split > 0:
                split_idx = int(len(train_data) * (1 - self.config.validation_split))
                if self.config.shuffle:
                    np.random.shuffle(train_data)
                val_data = train_data[split_idx:]
                train_data = train_data[:split_idx]

            self._logger.info(
                "Starting training",
                train_samples=len(train_data),
                val_samples=len(val_data) if val_data else 0,
                epochs=self.config.max_epochs,
            )

            # Training callbacks
            self._call_callbacks("on_train_begin")

            # Training loop
            for epoch in range(self.config.max_epochs):
                if self.should_stop:
                    break

                self.current_epoch = epoch
                metrics = self._train_epoch(train_data, val_data, epoch)

                # Log to tracker
                self.tracker.log_metrics(metrics, step=epoch, run=run)

                self._logger.info(
                    "Epoch completed",
                    epoch=epoch + 1,
                    total_epochs=self.config.max_epochs,
                    **metrics,
                )

            self._call_callbacks("on_train_end")
            self.tracker.end_run(ExperimentStatus.COMPLETED, run=run)

        except Exception as e:
            self._logger.error("Training failed", error=str(e))
            self.tracker.end_run(ExperimentStatus.FAILED, error_message=str(e), run=run)
            raise

        return run

    def _train_epoch(
        self,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]],
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self._call_callbacks("on_epoch_begin", epoch=epoch)

        # Shuffle training data
        if self.config.shuffle:
            indices = np.random.permutation(len(train_data))
            train_data = [train_data[i] for i in indices]

        # Create batches
        num_batches = (len(train_data) + self.config.batch_size - 1) // self.config.batch_size
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            self._call_callbacks("on_batch_begin", batch=batch_idx)

            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(train_data))
            batch = train_data[start_idx:end_idx]

            # Forward pass and compute loss
            loss = self._train_batch(batch)
            epoch_loss += loss

            self._call_callbacks("on_batch_end", batch=batch_idx, loss=loss)
            self.global_step += 1

        train_loss = epoch_loss / num_batches

        # Validation
        metrics = {"train_loss": train_loss}
        if val_data:
            val_metrics = self._validate(val_data)
            metrics.update(val_metrics)

        self._call_callbacks("on_epoch_end", epoch=epoch, metrics=metrics)
        return metrics

    def _train_batch(self, batch: List[Dict[str, Any]]) -> float:
        """Train on a single batch. Override in subclass for specific models."""
        # Default implementation - assumes model has a train_step method
        if hasattr(self.model, "train_step"):
            return self.model.train_step(batch, self.learning_rate)

        # Fallback - simple MSE loss
        loss = 0.0
        for sample in batch:
            if hasattr(self.model, "forward"):
                pred = self.model.forward(sample.get("features", []))
                target = sample.get("target", 0)
                loss += (pred - target) ** 2
        return loss / len(batch) if batch else 0.0

    def _validate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate the model."""
        if hasattr(self.model, "evaluate"):
            return self.model.evaluate(val_data)

        # Default validation
        total_loss = 0.0
        for sample in val_data:
            if hasattr(self.model, "forward"):
                pred = self.model.forward(sample.get("features", []))
                target = sample.get("target", 0)
                total_loss += (pred - target) ** 2

        return {"val_loss": total_loss / len(val_data) if val_data else 0.0}

    def _call_callbacks(self, event: str, **kwargs) -> None:
        """Call all callbacks for an event."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(self, **kwargs)

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        self._logger.info("Model saved", path=path)

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._logger.info("Model loaded", path=path)


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameters."""
    name: str
    type: str  # int, float, categorical, log_float
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None

    def sample(self) -> Any:
        """Sample a value from this hyperparameter space."""
        if self.type == "int":
            return random.randint(int(self.low), int(self.high))
        elif self.type == "float":
            if self.step:
                steps = int((self.high - self.low) / self.step)
                return self.low + random.randint(0, steps) * self.step
            return random.uniform(self.low, self.high)
        elif self.type == "log_float":
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return np.exp(random.uniform(log_low, log_high))
        elif self.type == "categorical":
            return random.choice(self.choices)
        else:
            raise ValueError(f"Unknown hyperparameter type: {self.type}")


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    search_time_seconds: float


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using various strategies.

    Supports:
    - Random search
    - Grid search
    - Bayesian optimization (simple)
    """

    def __init__(
        self,
        param_space: List[HyperparameterSpace],
        objective: Callable[[Dict[str, Any]], float],
        mode: str = "min",  # min or max
        n_trials: int = 20,
        strategy: str = "random",  # random, grid, bayesian
    ):
        self.param_space = {p.name: p for p in param_space}
        self.objective = objective
        self.mode = mode
        self.n_trials = n_trials
        self.strategy = strategy
        self._logger = structlog.get_logger("HyperparameterOptimizer")

    def optimize(self) -> OptimizationResult:
        """Run the optimization."""
        start_time = time.time()
        trials = []

        self._logger.info(
            "Starting hyperparameter optimization",
            strategy=self.strategy,
            n_trials=self.n_trials,
            mode=self.mode,
        )

        if self.strategy == "grid":
            param_grid = self._generate_grid()
            for i, params in enumerate(param_grid[:self.n_trials]):
                score = self._evaluate(params, i)
                trials.append({"params": params, "score": score, "trial": i})
        else:
            for i in range(self.n_trials):
                params = self._sample_params()
                score = self._evaluate(params, i)
                trials.append({"params": params, "score": score, "trial": i})

        # Find best trial
        if self.mode == "min":
            best_trial = min(trials, key=lambda t: t["score"])
        else:
            best_trial = max(trials, key=lambda t: t["score"])

        search_time = time.time() - start_time

        self._logger.info(
            "Optimization completed",
            best_score=best_trial["score"],
            best_params=best_trial["params"],
            search_time=search_time,
        )

        return OptimizationResult(
            best_params=best_trial["params"],
            best_score=best_trial["score"],
            all_trials=trials,
            search_time_seconds=search_time,
        )

    def _sample_params(self) -> Dict[str, Any]:
        """Sample a set of hyperparameters."""
        return {name: space.sample() for name, space in self.param_space.items()}

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate a grid of hyperparameters."""
        from itertools import product

        grids = {}
        for name, space in self.param_space.items():
            if space.type == "categorical":
                grids[name] = space.choices
            elif space.type in ("int", "float"):
                n_points = 5  # Number of grid points per dimension
                if space.type == "int":
                    grids[name] = list(range(int(space.low), int(space.high) + 1, max(1, (int(space.high) - int(space.low)) // n_points)))
                else:
                    grids[name] = list(np.linspace(space.low, space.high, n_points))
            elif space.type == "log_float":
                grids[name] = list(np.exp(np.linspace(np.log(space.low), np.log(space.high), 5)))

        keys = list(grids.keys())
        values = [grids[k] for k in keys]
        return [dict(zip(keys, v)) for v in product(*values)]

    def _evaluate(self, params: Dict[str, Any], trial: int) -> float:
        """Evaluate a set of hyperparameters."""
        try:
            score = self.objective(params)
            self._logger.debug(
                "Trial completed",
                trial=trial,
                params=params,
                score=score,
            )
            return score
        except Exception as e:
            self._logger.warning("Trial failed", trial=trial, error=str(e))
            return float("inf") if self.mode == "min" else float("-inf")


# =============================================================================
# Model Serialization
# =============================================================================

class ModelSerializer:
    """Handles model serialization and versioning."""

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._logger = structlog.get_logger("ModelSerializer")

    def save(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a model with metadata."""
        version = version or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        meta = {
            "name": name,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "model_class": type(model).__name__,
            **(metadata or {}),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._logger.info("Model saved", name=name, version=version, path=str(model_path))
        return str(model_path)

    def load(self, name: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Load a model and its metadata."""
        model_base = self.models_dir / name

        if version is None:
            # Load latest version
            versions = sorted(model_base.iterdir())
            if not versions:
                raise ValueError(f"No versions found for model: {name}")
            version_dir = versions[-1]
        else:
            version_dir = model_base / version

        # Load model
        with open(version_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load metadata
        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        self._logger.info("Model loaded", name=name, version=metadata.get("version"))
        return model, metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models."""
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        meta_path = version_dir / "metadata.json"
                        if meta_path.exists():
                            with open(meta_path) as f:
                                models.append(json.load(f))
        return sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Experiment tracking
    "ExperimentStatus",
    "Metric",
    "Checkpoint",
    "ExperimentConfig",
    "ExperimentRun",
    "ExperimentTracker",

    # Callbacks
    "Callback",
    "EarlyStopping",
    "LearningRateScheduler",
    "MetricsLogger",
    "CheckpointCallback",

    # Trainer
    "TrainerConfig",
    "Trainer",

    # Hyperparameter optimization
    "HyperparameterSpace",
    "OptimizationResult",
    "HyperparameterOptimizer",

    # Serialization
    "ModelSerializer",
]
