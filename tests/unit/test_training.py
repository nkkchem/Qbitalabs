"""
QbitaLab: Unit tests for training infrastructure.

Tests:
- Experiment tracking
- Training callbacks
- Trainer functionality
- Hyperparameter optimization
- Model serialization
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class TestExperimentTracking:
    """Tests for experiment tracking."""

    def test_create_run(self):
        """Test creating an experiment run."""
        from qbitalabs.training import ExperimentTracker, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)

            config = ExperimentConfig(
                name="test_experiment",
                model_type="GNN",
                hyperparameters={"learning_rate": 0.001},
            )

            run = tracker.create_run(config)

            assert run.run_id.startswith("test_experiment_")
            assert run.config.model_type == "GNN"

    def test_log_metrics(self):
        """Test logging metrics."""
        from qbitalabs.training import ExperimentTracker, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            config = ExperimentConfig(name="test", model_type="test", hyperparameters={})
            run = tracker.create_run(config)
            tracker.start_run(run)

            tracker.log_metric("loss", 0.5, step=1)
            tracker.log_metric("loss", 0.3, step=2)
            tracker.log_metric("accuracy", 0.8, step=2)

            assert len(run.metrics) == 3
            assert run.get_best_metric("loss", mode="min") == 0.3
            assert run.get_best_metric("accuracy", mode="max") == 0.8

    def test_save_and_load_run(self):
        """Test saving and loading runs."""
        from qbitalabs.training import ExperimentTracker, ExperimentConfig, ExperimentStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)

            config = ExperimentConfig(
                name="persistence_test",
                model_type="test",
                hyperparameters={"lr": 0.01},
            )

            run = tracker.create_run(config)
            tracker.start_run(run)
            tracker.log_metric("loss", 0.5, step=1)
            tracker.end_run(ExperimentStatus.COMPLETED)

            # Load the run
            loaded_run = tracker.get_run(run.run_id)

            assert loaded_run is not None
            assert loaded_run.run_id == run.run_id
            assert loaded_run.status == ExperimentStatus.COMPLETED
            assert len(loaded_run.metrics) == 1

    def test_list_runs(self):
        """Test listing runs."""
        from qbitalabs.training import ExperimentTracker, ExperimentConfig, ExperimentStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)

            # Create multiple runs
            for i in range(3):
                config = ExperimentConfig(name=f"run_{i}", model_type="test", hyperparameters={})
                run = tracker.create_run(config)
                tracker.start_run(run)
                tracker.end_run(ExperimentStatus.COMPLETED)

            runs = tracker.list_runs()
            assert len(runs) == 3


class TestCallbacks:
    """Tests for training callbacks."""

    def test_early_stopping(self):
        """Test early stopping callback."""
        from qbitalabs.training import EarlyStopping, Trainer, TrainerConfig

        callback = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.01)

        # Simulated trainer
        class MockTrainer:
            should_stop = False
            model = None

        trainer = MockTrainer()

        # Improving metrics
        callback.on_epoch_end(trainer, epoch=0, metrics={"val_loss": 1.0})
        callback.on_epoch_end(trainer, epoch=1, metrics={"val_loss": 0.5})
        callback.on_epoch_end(trainer, epoch=2, metrics={"val_loss": 0.3})

        assert not trainer.should_stop

        # No improvement
        callback.on_epoch_end(trainer, epoch=3, metrics={"val_loss": 0.35})
        callback.on_epoch_end(trainer, epoch=4, metrics={"val_loss": 0.32})
        callback.on_epoch_end(trainer, epoch=5, metrics={"val_loss": 0.31})

        assert trainer.should_stop

    def test_learning_rate_scheduler_cosine(self):
        """Test cosine learning rate scheduler."""
        from qbitalabs.training import LearningRateScheduler

        scheduler = LearningRateScheduler(
            schedule="cosine",
            initial_lr=0.1,
            min_lr=0.001,
            warmup_epochs=0,
        )

        class MockTrainer:
            learning_rate = 0.1
            max_epochs = 100

        trainer = MockTrainer()

        # At epoch 0, LR should be near initial
        scheduler.on_epoch_begin(trainer, epoch=0)
        assert trainer.learning_rate == pytest.approx(0.1, rel=0.1)

        # At epoch 50 (middle), LR should be around mean
        scheduler.on_epoch_begin(trainer, epoch=50)
        expected_mid = 0.001 + 0.5 * (0.1 - 0.001)  # ~0.05
        assert trainer.learning_rate == pytest.approx(expected_mid, rel=0.1)

        # At epoch 100, LR should be near min
        scheduler.on_epoch_begin(trainer, epoch=99)
        assert trainer.learning_rate == pytest.approx(0.001, rel=0.1)

    def test_learning_rate_warmup(self):
        """Test learning rate warmup."""
        from qbitalabs.training import LearningRateScheduler

        scheduler = LearningRateScheduler(
            schedule="cosine",
            initial_lr=0.1,
            warmup_epochs=10,
        )

        class MockTrainer:
            learning_rate = 0.0
            max_epochs = 100

        trainer = MockTrainer()

        # During warmup, LR should increase linearly
        scheduler.on_epoch_begin(trainer, epoch=0)
        assert trainer.learning_rate == pytest.approx(0.01, rel=0.1)  # 1/10 of initial

        scheduler.on_epoch_begin(trainer, epoch=5)
        assert trainer.learning_rate == pytest.approx(0.06, rel=0.1)  # 6/10 of initial


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from qbitalabs.training import Trainer, TrainerConfig

        class SimpleModel:
            def forward(self, x):
                return sum(x) if x else 0

        config = TrainerConfig(
            max_epochs=10,
            batch_size=4,
            learning_rate=0.01,
        )

        trainer = Trainer(model=SimpleModel(), config=config)

        assert trainer.max_epochs == 10
        assert trainer.learning_rate == 0.01

    def test_trainer_fit(self):
        """Test trainer fit method."""
        from qbitalabs.training import Trainer, TrainerConfig, ExperimentConfig

        class SimpleModel:
            def forward(self, x):
                return sum(x.get("features", [])) if isinstance(x, dict) else 0

            def train_step(self, batch, lr):
                return 0.1  # Constant loss for testing

            def evaluate(self, data):
                return {"val_loss": 0.05}

        config = TrainerConfig(
            max_epochs=5,
            batch_size=2,
            learning_rate=0.01,
            validation_split=0.2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            from qbitalabs.training import ExperimentTracker
            tracker = ExperimentTracker(tmpdir)
            trainer = Trainer(model=SimpleModel(), config=config, tracker=tracker)

            # Training data
            train_data = [
                {"features": [1, 2, 3], "target": 6},
                {"features": [4, 5, 6], "target": 15},
                {"features": [7, 8, 9], "target": 24},
                {"features": [10, 11, 12], "target": 33},
                {"features": [13, 14, 15], "target": 42},
            ]

            experiment_config = ExperimentConfig(
                name="test_training",
                model_type="SimpleModel",
                hyperparameters={"lr": 0.01},
            )

            run = trainer.fit(train_data, experiment_config=experiment_config)

            assert run.status.value == "completed"
            assert len(run.metrics) > 0


class TestHyperparameterOptimization:
    """Tests for hyperparameter optimization."""

    def test_hyperparameter_space_sampling(self):
        """Test sampling from hyperparameter spaces."""
        from qbitalabs.training import HyperparameterSpace

        # Integer space
        int_space = HyperparameterSpace(name="n_layers", type="int", low=1, high=10)
        for _ in range(10):
            value = int_space.sample()
            assert 1 <= value <= 10
            assert isinstance(value, int)

        # Float space
        float_space = HyperparameterSpace(name="dropout", type="float", low=0.0, high=0.5)
        for _ in range(10):
            value = float_space.sample()
            assert 0.0 <= value <= 0.5

        # Log float space
        log_space = HyperparameterSpace(name="lr", type="log_float", low=1e-5, high=1e-1)
        for _ in range(10):
            value = log_space.sample()
            assert 1e-5 <= value <= 1e-1

        # Categorical space
        cat_space = HyperparameterSpace(name="activation", type="categorical", choices=["relu", "tanh", "sigmoid"])
        for _ in range(10):
            value = cat_space.sample()
            assert value in ["relu", "tanh", "sigmoid"]

    def test_random_search(self):
        """Test random search optimization."""
        from qbitalabs.training import HyperparameterOptimizer, HyperparameterSpace

        param_space = [
            HyperparameterSpace(name="x", type="float", low=-5, high=5),
            HyperparameterSpace(name="y", type="float", low=-5, high=5),
        ]

        # Objective: minimize (x-1)^2 + (y-2)^2
        def objective(params):
            return (params["x"] - 1) ** 2 + (params["y"] - 2) ** 2

        optimizer = HyperparameterOptimizer(
            param_space=param_space,
            objective=objective,
            mode="min",
            n_trials=50,
            strategy="random",
        )

        result = optimizer.optimize()

        # Should find params close to (1, 2)
        assert result.best_score < 2.0  # Reasonable threshold
        assert len(result.all_trials) == 50

    def test_grid_search(self):
        """Test grid search optimization."""
        from qbitalabs.training import HyperparameterOptimizer, HyperparameterSpace

        param_space = [
            HyperparameterSpace(name="activation", type="categorical", choices=["relu", "tanh"]),
            HyperparameterSpace(name="layers", type="int", low=1, high=3),
        ]

        results = {}

        def objective(params):
            key = f"{params['activation']}_{params['layers']}"
            score = hash(key) % 100 / 100  # Deterministic score
            results[key] = score
            return score

        optimizer = HyperparameterOptimizer(
            param_space=param_space,
            objective=objective,
            mode="min",
            n_trials=20,
            strategy="grid",
        )

        result = optimizer.optimize()

        assert result.best_params is not None
        assert "activation" in result.best_params
        assert "layers" in result.best_params


class TestModelSerializer:
    """Tests for model serialization."""

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        from qbitalabs.training import ModelSerializer

        class SimpleModel:
            def __init__(self, param):
                self.param = param

        with tempfile.TemporaryDirectory() as tmpdir:
            serializer = ModelSerializer(tmpdir)

            # Save model
            model = SimpleModel(param=42)
            path = serializer.save(
                model,
                name="test_model",
                metadata={"accuracy": 0.95},
            )

            assert Path(path).exists()

            # Load model
            loaded_model, metadata = serializer.load("test_model")

            assert loaded_model.param == 42
            assert metadata["accuracy"] == 0.95
            assert metadata["model_class"] == "SimpleModel"

    def test_model_versioning(self):
        """Test model versioning."""
        from qbitalabs.training import ModelSerializer

        class Model:
            def __init__(self, version):
                self.version = version

        with tempfile.TemporaryDirectory() as tmpdir:
            serializer = ModelSerializer(tmpdir)

            # Save multiple versions
            serializer.save(Model(1), name="versioned_model", version="v1")
            serializer.save(Model(2), name="versioned_model", version="v2")
            serializer.save(Model(3), name="versioned_model", version="v3")

            # Load specific version
            model_v1, _ = serializer.load("versioned_model", version="v1")
            assert model_v1.version == 1

            # Load latest (v3)
            model_latest, _ = serializer.load("versioned_model")
            assert model_latest.version == 3

    def test_list_models(self):
        """Test listing saved models."""
        from qbitalabs.training import ModelSerializer

        class Model:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            serializer = ModelSerializer(tmpdir)

            serializer.save(Model(), name="model_a")
            serializer.save(Model(), name="model_b")
            serializer.save(Model(), name="model_a", version="v2")

            models = serializer.list_models()

            assert len(models) == 3
            names = [m["name"] for m in models]
            assert "model_a" in names
            assert "model_b" in names


class TestMetricsTracking:
    """Tests for metrics tracking in experiment runs."""

    def test_metric_history(self):
        """Test getting metric history."""
        from qbitalabs.training import ExperimentRun, ExperimentConfig

        config = ExperimentConfig(name="test", model_type="test", hyperparameters={})
        run = ExperimentRun(run_id="test_run", config=config)

        run.log_metric("loss", 1.0, step=1)
        run.log_metric("loss", 0.5, step=2)
        run.log_metric("loss", 0.3, step=3)
        run.log_metric("accuracy", 0.8, step=3)

        history = run.get_metric_history("loss")
        assert history == [(1, 1.0), (2, 0.5), (3, 0.3)]

    def test_best_metric(self):
        """Test getting best metric value."""
        from qbitalabs.training import ExperimentRun, ExperimentConfig

        config = ExperimentConfig(name="test", model_type="test", hyperparameters={})
        run = ExperimentRun(run_id="test_run", config=config)

        run.log_metric("loss", 1.0, step=1)
        run.log_metric("loss", 0.5, step=2)
        run.log_metric("loss", 0.8, step=3)

        assert run.get_best_metric("loss", mode="min") == 0.5
        assert run.get_best_metric("loss", mode="max") == 1.0

    def test_run_serialization(self):
        """Test run serialization to dict."""
        from qbitalabs.training import ExperimentRun, ExperimentConfig, ExperimentStatus

        config = ExperimentConfig(
            name="test",
            model_type="GNN",
            hyperparameters={"lr": 0.01},
            tags=["test", "experiment"],
        )

        run = ExperimentRun(run_id="test_run", config=config)
        run.status = ExperimentStatus.COMPLETED
        run.log_metric("loss", 0.5, step=1)

        data = run.to_dict()

        assert data["run_id"] == "test_run"
        assert data["status"] == "completed"
        assert len(data["metrics"]) == 1
        assert data["config"]["tags"] == ["test", "experiment"]
