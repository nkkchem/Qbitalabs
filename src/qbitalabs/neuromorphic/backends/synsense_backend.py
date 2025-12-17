"""
SynSense Backend for QBitaLabs

Provides integration with SynSense (formerly aiCTX) neuromorphic chips.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import structlog

from qbitalabs.neuromorphic.backends.base_backend import (
    BackendType,
    BaseNeuromorphicBackend,
    InferenceResult,
    NeuronModel,
    SpikeData,
)

logger = structlog.get_logger(__name__)

# Lazy imports
sinabs = None
samna = None


def _import_synsense() -> None:
    """Lazily import SynSense SDK."""
    global sinabs, samna
    if sinabs is None:
        try:
            import sinabs as _sinabs
            import sinabs.layers as _layers

            sinabs = _sinabs
        except ImportError:
            pass

    if samna is None:
        try:
            import samna as _samna

            samna = _samna
        except ImportError:
            pass


class SynSenseBackend(BaseNeuromorphicBackend):
    """
    SynSense neuromorphic backend.

    Supports:
    - DYNAP-CNN (convolutional SNN)
    - Speck (ultra-low power)
    - Sinabs framework for simulation

    Features:
    - Event-driven vision processing
    - Ultra-low latency (<1ms)
    - Configurable neuron models
    - PyTorch-compatible training

    Example:
        >>> backend = SynSenseBackend()
        >>> await backend.initialize()
        >>> model = backend.convert_pytorch_model(torch_model)
        >>> result = await backend.infer(event_data)
    """

    def __init__(
        self,
        device_type: str = "dynapcnn",  # "dynapcnn", "speck"
        timestep: float = 1.0,
        neuron_model: NeuronModel = NeuronModel.LIF,
    ):
        """
        Initialize SynSense backend.

        Args:
            device_type: Type of SynSense device.
            timestep: Simulation timestep.
            neuron_model: Neuron model to use.
        """
        super().__init__(
            backend_type=BackendType.SYNSENSE,
            neuron_model=neuron_model,
            timestep=timestep,
            n_neurons_max=1000000,  # DYNAP-CNN capacity
        )

        self.device_type = device_type
        self._device = None
        self._model = None

    async def initialize(self) -> None:
        """Initialize SynSense device."""
        _import_synsense()

        if samna is not None:
            try:
                devices = samna.device.get_all_devices()
                if devices:
                    self._device = devices[0]
                    self._logger.info(
                        "SynSense device found",
                        device=str(self._device),
                    )
                else:
                    self._logger.info("No SynSense hardware, using simulation")
            except Exception as e:
                self._logger.warning(
                    "Failed to initialize SynSense device",
                    error=str(e),
                )

        self._initialized = True
        self._logger.info(
            "SynSense backend initialized",
            device_type=self.device_type,
        )

    async def shutdown(self) -> None:
        """Shutdown SynSense device."""
        self._device = None
        self._model = None
        self._initialized = False
        self._logger.info("SynSense backend shutdown")

    def load_model(self, model_path: str) -> Any:
        """Load a compiled SynSense model."""
        _import_synsense()

        import torch

        self._model = torch.load(model_path)
        self._logger.info("Model loaded", path=model_path)

        return self._model

    def compile_model(self, model: Any, **kwargs: Any) -> Any:
        """Compile a model for SynSense."""
        _import_synsense()

        return self.convert_pytorch_model(model, **kwargs)

    def convert_pytorch_model(
        self,
        torch_model: Any,
        input_shape: tuple[int, ...] | None = None,
        batch_size: int = 1,
    ) -> Any:
        """
        Convert PyTorch model to SynSense-compatible SNN.

        Args:
            torch_model: PyTorch model (ANN or SNN).
            input_shape: Input tensor shape.
            batch_size: Batch size.

        Returns:
            Sinabs SNN model.
        """
        _import_synsense()

        if sinabs is None:
            raise ImportError(
                "Sinabs not installed. Install with: pip install sinabs"
            )

        from sinabs.from_torch import from_model

        # Convert to SNN
        snn_model = from_model(
            torch_model,
            input_shape=input_shape,
            batch_size=batch_size,
            spike_threshold=1.0,
            spike_fn=sinabs.activation.SingleSpike,
        )

        self._model = snn_model
        self._logger.info("PyTorch model converted to SNN")

        return snn_model

    async def infer(
        self,
        input_data: SpikeData | np.ndarray,
        duration: float | None = None,
    ) -> InferenceResult:
        """Run inference."""
        if not self._initialized:
            await self.initialize()

        _import_synsense()

        # Convert input
        if isinstance(input_data, SpikeData):
            spike_matrix = input_data.to_dense(self.timestep)
        else:
            spike_matrix = input_data

        import time

        start_time = time.time()

        # Run inference
        if self._model is not None and sinabs is not None:
            import torch

            # Convert to tensor
            input_tensor = torch.tensor(
                spike_matrix, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                output = self._model(input_tensor)

            predictions = output.numpy().squeeze()
            output_spikes = (predictions > 0).astype(float)
        else:
            # Simulation fallback
            predictions = np.random.softmax(np.random.randn(10))
            output_spikes = np.zeros((10, spike_matrix.shape[1]))

        latency = (time.time() - start_time) * 1000

        # Estimate energy
        energy = self._estimate_energy(spike_matrix)
        self._energy_total += energy

        result = InferenceResult(
            output_spikes=SpikeData.from_dense(
                output_spikes if output_spikes.ndim == 2 else output_spikes.reshape(-1, 1),
                self.timestep,
            ),
            predictions=predictions.reshape(1, -1) if predictions.ndim == 1 else predictions,
            energy_consumption=energy,
            latency=latency,
            spike_count=int(np.sum(output_spikes)),
        )

        return result

    def _estimate_energy(self, input_data: np.ndarray) -> float:
        """Estimate energy consumption."""
        # DYNAP-CNN: ~1pJ per synaptic operation
        n_spikes = np.sum(input_data)
        energy_per_spike = 1e-12  # 1 pJ

        if self.device_type == "speck":
            energy_per_spike = 0.5e-12  # 0.5 pJ for Speck

        return float(n_spikes * energy_per_spike * 1000)  # Scale factor

    async def train_on_chip(
        self,
        input_data: SpikeData,
        target: np.ndarray,
        learning_rule: str = "stdp",
        epochs: int = 1,
    ) -> dict[str, Any]:
        """Train using backprop through time (simulation)."""
        if not self._initialized:
            await self.initialize()

        _import_synsense()

        if self._model is None or sinabs is None:
            return {"error": "No model loaded or Sinabs not available"}

        import torch
        import torch.nn as nn

        # Convert input
        if isinstance(input_data, SpikeData):
            spike_matrix = input_data.to_dense(self.timestep)
        else:
            spike_matrix = input_data

        input_tensor = torch.tensor(
            spike_matrix, dtype=torch.float32
        ).unsqueeze(0)
        target_tensor = torch.tensor(target, dtype=torch.long)

        # Training loop
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            output = self._model(input_tensor)
            loss = criterion(output, target_tensor)

            loss.backward()
            optimizer.step()

            history.append(float(loss.item()))

        return {
            "epochs": epochs,
            "final_loss": history[-1] if history else 0,
            "loss_history": history,
        }

    def create_snn_model(
        self,
        architecture: str = "lenet",
        n_classes: int = 10,
        input_shape: tuple[int, ...] = (1, 28, 28),
    ) -> Any:
        """
        Create a SynSense-optimized SNN.

        Args:
            architecture: Model architecture.
            n_classes: Number of output classes.
            input_shape: Input shape (C, H, W).

        Returns:
            Sinabs SNN model.
        """
        _import_synsense()

        if sinabs is None:
            raise ImportError("Sinabs required for model creation")

        import torch.nn as nn
        from sinabs.layers import LIF, IAF

        if architecture == "lenet":
            model = nn.Sequential(
                # Conv1
                nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=2),
                LIF(tau_mem=20.0),
                nn.AvgPool2d(2),

                # Conv2
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                LIF(tau_mem=20.0),
                nn.AvgPool2d(2),

                # Flatten
                nn.Flatten(),

                # FC
                nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 256),
                LIF(tau_mem=20.0),

                # Output
                nn.Linear(256, n_classes),
                IAF(),
            )

        elif architecture == "simple":
            flat_size = np.prod(input_shape)
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, 128),
                LIF(tau_mem=20.0),
                nn.Linear(128, n_classes),
                IAF(),
            )

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self._model = model
        return model

    def process_events(
        self,
        events: dict[str, np.ndarray],
        duration: float = 100.0,
    ) -> SpikeData:
        """
        Process event camera data.

        Args:
            events: Dictionary with 'x', 'y', 't', 'p' arrays.
            duration: Processing window duration.

        Returns:
            Converted spike data.
        """
        x = events["x"]
        y = events["y"]
        t = events["t"]
        p = events.get("p", np.ones_like(x))

        # Determine grid size
        width = int(np.max(x)) + 1
        height = int(np.max(y)) + 1

        # Convert to linear indices
        neurons = y * width + x

        # Normalize timestamps
        t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8) * duration

        return SpikeData(
            times=t_normalized,
            neurons=neurons.astype(int),
            n_neurons=width * height,
            duration=duration,
        )

    def _supports_on_chip_learning(self) -> bool:
        """Limited on-chip learning support."""
        return False

    def _get_supported_layers(self) -> list[str]:
        """Get SynSense-supported layers."""
        return [
            "Conv2d",
            "Linear",
            "AvgPool2d",
            "LIF",
            "IAF",
            "Flatten",
        ]


class EventCameraProcessor:
    """
    Process event camera (DVS) data with SynSense.

    Optimized for dynamic vision sensors.
    """

    def __init__(self, backend: SynSenseBackend):
        """Initialize event processor."""
        self.backend = backend

    def events_to_frames(
        self,
        events: dict[str, np.ndarray],
        shape: tuple[int, int],
        time_window: float = 10.0,  # ms
        n_frames: int = 10,
    ) -> np.ndarray:
        """
        Convert events to frame representation.

        Args:
            events: Event data.
            shape: Frame shape (H, W).
            time_window: Time window per frame.
            n_frames: Number of output frames.

        Returns:
            Frame tensor (n_frames, H, W).
        """
        x = events["x"]
        y = events["y"]
        t = events["t"]
        p = events.get("p", np.ones_like(x))

        # Normalize time
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8) * n_frames

        frames = np.zeros((n_frames, shape[0], shape[1]))

        for frame_idx in range(n_frames):
            mask = (t_norm >= frame_idx) & (t_norm < frame_idx + 1)
            frame_x = x[mask].astype(int)
            frame_y = y[mask].astype(int)
            frame_p = p[mask]

            # Accumulate events
            valid = (frame_x < shape[1]) & (frame_y < shape[0])
            np.add.at(frames[frame_idx], (frame_y[valid], frame_x[valid]), frame_p[valid])

        return frames

    def filter_noise(
        self,
        events: dict[str, np.ndarray],
        time_threshold: float = 1.0,  # ms
        spatial_threshold: int = 1,  # pixels
    ) -> dict[str, np.ndarray]:
        """
        Filter noise from event stream.

        Args:
            events: Raw event data.
            time_threshold: Temporal correlation window.
            spatial_threshold: Spatial correlation window.

        Returns:
            Filtered events.
        """
        x = events["x"]
        y = events["y"]
        t = events["t"]
        p = events.get("p", np.ones_like(x))

        # Simple temporal filtering
        valid = np.ones(len(x), dtype=bool)

        for i in range(1, len(x)):
            dt = t[i] - t[i - 1]
            dx = abs(x[i] - x[i - 1])
            dy = abs(y[i] - y[i - 1])

            if dt > time_threshold and (dx > spatial_threshold or dy > spatial_threshold):
                valid[i] = False

        return {
            "x": x[valid],
            "y": y[valid],
            "t": t[valid],
            "p": p[valid],
        }
