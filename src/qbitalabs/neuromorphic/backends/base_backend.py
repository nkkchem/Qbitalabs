"""
Base Neuromorphic Backend for QBitaLabs

Provides abstract interface for neuromorphic computing backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class BackendType(str, Enum):
    """Supported neuromorphic backend types."""

    AKIDA = "akida"
    LOIHI = "loihi"
    SYNSENSE = "synsense"
    SIMULATOR = "simulator"


class NeuronModel(str, Enum):
    """Supported neuron models."""

    LIF = "lif"  # Leaky Integrate-and-Fire
    ALIF = "alif"  # Adaptive LIF
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    RESONATE_FIRE = "resonate_fire"


@dataclass
class SpikeData:
    """Container for spike train data."""

    times: np.ndarray  # Spike times (ms)
    neurons: np.ndarray  # Neuron indices
    n_neurons: int = 0
    duration: float = 0.0  # Total duration (ms)

    @classmethod
    def from_dense(cls, spike_matrix: np.ndarray, dt: float = 1.0) -> "SpikeData":
        """Create from dense binary spike matrix (neurons x time)."""
        neurons, times = np.where(spike_matrix)
        return cls(
            times=times.astype(float) * dt,
            neurons=neurons,
            n_neurons=spike_matrix.shape[0],
            duration=spike_matrix.shape[1] * dt,
        )

    def to_dense(self, dt: float = 1.0) -> np.ndarray:
        """Convert to dense binary spike matrix."""
        n_timesteps = int(self.duration / dt) + 1
        matrix = np.zeros((self.n_neurons, n_timesteps), dtype=np.float32)
        time_indices = (self.times / dt).astype(int)
        valid = (time_indices >= 0) & (time_indices < n_timesteps)
        matrix[self.neurons[valid], time_indices[valid]] = 1.0
        return matrix


@dataclass
class InferenceResult:
    """Result from neuromorphic inference."""

    output_spikes: SpikeData
    predictions: np.ndarray | None = None
    energy_consumption: float = 0.0  # Joules
    latency: float = 0.0  # ms
    spike_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseNeuromorphicBackend(ABC):
    """
    Abstract base class for neuromorphic computing backends.

    Provides unified interface for:
    - SNN model loading and compilation
    - Spike-based inference
    - On-chip learning
    - Energy monitoring

    Example:
        >>> backend = AkidaBackend()
        >>> await backend.initialize()
        >>> model = backend.load_model("snn_classifier.h5")
        >>> result = await backend.infer(spike_data)
    """

    def __init__(
        self,
        backend_type: BackendType,
        neuron_model: NeuronModel = NeuronModel.LIF,
        timestep: float = 1.0,  # ms
        n_neurons_max: int = 10000,
    ):
        """
        Initialize the neuromorphic backend.

        Args:
            backend_type: Type of neuromorphic backend.
            neuron_model: Default neuron model to use.
            timestep: Simulation timestep in ms.
            n_neurons_max: Maximum number of neurons.
        """
        self.backend_type = backend_type
        self.neuron_model = neuron_model
        self.timestep = timestep
        self.n_neurons_max = n_neurons_max

        self._initialized = False
        self._model = None
        self._energy_total = 0.0

        self._logger = structlog.get_logger(f"neuromorphic.{backend_type.value}")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend connection."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend connection."""
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """
        Load a compiled SNN model.

        Args:
            model_path: Path to the model file.

        Returns:
            Loaded model object.
        """
        pass

    @abstractmethod
    def compile_model(self, model: Any, **kwargs: Any) -> Any:
        """
        Compile a model for the backend.

        Args:
            model: Model to compile (Keras, PyTorch, etc.).
            **kwargs: Backend-specific options.

        Returns:
            Compiled model.
        """
        pass

    @abstractmethod
    async def infer(
        self,
        input_data: SpikeData | np.ndarray,
        duration: float | None = None,
    ) -> InferenceResult:
        """
        Run inference on the neuromorphic hardware.

        Args:
            input_data: Input spike data or rate-coded input.
            duration: Inference duration in ms.

        Returns:
            Inference results.
        """
        pass

    @abstractmethod
    async def train_on_chip(
        self,
        input_data: SpikeData,
        target: np.ndarray,
        learning_rule: str = "stdp",
        epochs: int = 1,
    ) -> dict[str, Any]:
        """
        Run on-chip learning.

        Args:
            input_data: Training spike data.
            target: Target outputs.
            learning_rule: Learning rule (stdp, reward_modulated, etc.).
            epochs: Training epochs.

        Returns:
            Training metrics.
        """
        pass

    def encode_rate(
        self,
        data: np.ndarray,
        duration: float = 100.0,
        max_rate: float = 100.0,
    ) -> SpikeData:
        """
        Encode continuous data as spike rates.

        Args:
            data: Input data (samples x features).
            duration: Encoding duration per sample (ms).
            max_rate: Maximum firing rate (Hz).

        Returns:
            Rate-encoded spike data.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples, n_features = data.shape
        n_timesteps = int(duration / self.timestep)

        # Normalize to [0, 1]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

        all_times = []
        all_neurons = []

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                rate = data_norm[sample_idx, feature_idx] * max_rate
                if rate > 0:
                    # Poisson process
                    n_spikes = np.random.poisson(rate * duration / 1000)
                    spike_times = np.random.uniform(0, duration, n_spikes)
                    all_times.extend(spike_times + sample_idx * duration)
                    all_neurons.extend([feature_idx] * n_spikes)

        return SpikeData(
            times=np.array(all_times),
            neurons=np.array(all_neurons),
            n_neurons=n_features,
            duration=n_samples * duration,
        )

    def encode_temporal(
        self,
        data: np.ndarray,
        max_latency: float = 50.0,
    ) -> SpikeData:
        """
        Encode data using time-to-first-spike.

        Higher values = earlier spikes.

        Args:
            data: Input data.
            max_latency: Maximum spike latency (ms).

        Returns:
            Temporally encoded spike data.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples, n_features = data.shape

        # Normalize and invert (high value = low latency)
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        latencies = (1 - data_norm) * max_latency

        all_times = []
        all_neurons = []

        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                spike_time = latencies[sample_idx, feature_idx]
                all_times.append(spike_time + sample_idx * max_latency)
                all_neurons.append(feature_idx)

        return SpikeData(
            times=np.array(all_times),
            neurons=np.array(all_neurons),
            n_neurons=n_features,
            duration=n_samples * max_latency,
        )

    def decode_rate(
        self,
        spikes: SpikeData,
        window: float = 50.0,
    ) -> np.ndarray:
        """
        Decode spike rates to continuous values.

        Args:
            spikes: Output spike data.
            window: Time window for rate estimation (ms).

        Returns:
            Decoded values.
        """
        n_windows = int(spikes.duration / window)
        rates = np.zeros((n_windows, spikes.n_neurons))

        for w in range(n_windows):
            t_start = w * window
            t_end = (w + 1) * window

            mask = (spikes.times >= t_start) & (spikes.times < t_end)
            for n in range(spikes.n_neurons):
                neuron_mask = mask & (spikes.neurons == n)
                rates[w, n] = np.sum(neuron_mask) / (window / 1000)  # Hz

        return rates

    def get_energy_consumption(self) -> float:
        """Get total energy consumption in Joules."""
        return self._energy_total

    def reset_energy_counter(self) -> None:
        """Reset energy counter."""
        self._energy_total = 0.0

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    def get_capabilities(self) -> dict[str, Any]:
        """Get backend capabilities."""
        return {
            "backend_type": self.backend_type.value,
            "neuron_model": self.neuron_model.value,
            "max_neurons": self.n_neurons_max,
            "timestep_ms": self.timestep,
            "on_chip_learning": self._supports_on_chip_learning(),
            "supported_layers": self._get_supported_layers(),
        }

    def _supports_on_chip_learning(self) -> bool:
        """Check if backend supports on-chip learning."""
        return False

    def _get_supported_layers(self) -> list[str]:
        """Get supported layer types."""
        return ["dense", "conv2d", "pooling"]
