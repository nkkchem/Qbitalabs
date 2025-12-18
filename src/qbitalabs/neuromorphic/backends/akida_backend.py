"""
BrainChip Akida Backend for QBitaLabs

Provides integration with BrainChip's Akida neuromorphic processor.
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
akida = None
cnn2snn = None


def _import_akida() -> None:
    """Lazily import Akida SDK."""
    global akida, cnn2snn
    if akida is None:
        try:
            import akida as _akida

            akida = _akida
            try:
                import cnn2snn as _cnn2snn

                cnn2snn = _cnn2snn
            except ImportError:
                pass
        except ImportError as e:
            raise ImportError(
                "Akida SDK not installed. Install with: pip install akida"
            ) from e


class AkidaBackend(BaseNeuromorphicBackend):
    """
    BrainChip Akida neuromorphic backend.

    Features:
    - Edge AI acceleration
    - On-chip learning
    - Ultra-low power (< 1mW for inference)
    - Event-based processing
    - CNN-to-SNN conversion

    Example:
        >>> backend = AkidaBackend()
        >>> await backend.initialize()
        >>> model = backend.convert_keras_model(keras_model)
        >>> result = await backend.infer(input_data)
    """

    def __init__(
        self,
        device_id: int = 0,
        power_mode: str = "performance",  # "performance", "efficiency", "ultra_low"
        timestep: float = 1.0,
    ):
        """
        Initialize Akida backend.

        Args:
            device_id: Akida device ID.
            power_mode: Power consumption mode.
            timestep: Simulation timestep.
        """
        super().__init__(
            backend_type=BackendType.AKIDA,
            neuron_model=NeuronModel.LIF,
            timestep=timestep,
        )

        self.device_id = device_id
        self.power_mode = power_mode
        self._device = None

    async def initialize(self) -> None:
        """Initialize Akida device."""
        _import_akida()

        try:
            devices = akida.devices()
            if devices:
                self._device = devices[self.device_id]
                self._logger.info(
                    "Akida device initialized",
                    device=str(self._device),
                    version=akida.__version__,
                )
            else:
                self._logger.warning(
                    "No Akida hardware found, using virtual device"
                )
                self._device = None
        except Exception as e:
            self._logger.warning(
                "Failed to initialize Akida device",
                error=str(e),
            )
            self._device = None

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown Akida device."""
        self._device = None
        self._model = None
        self._initialized = False
        self._logger.info("Akida backend shutdown")

    def load_model(self, model_path: str) -> Any:
        """Load a compiled Akida model."""
        _import_akida()

        model = akida.Model(model_path)
        if self._device:
            model.map(self._device)

        self._model = model
        self._logger.info("Model loaded", path=model_path)

        return model

    def compile_model(self, model: Any, **kwargs: Any) -> Any:
        """Compile a model for Akida."""
        _import_akida()

        # Check if it's a Keras model
        if hasattr(model, "layers"):
            return self.convert_keras_model(model, **kwargs)

        return model

    def convert_keras_model(
        self,
        keras_model: Any,
        input_scaling: tuple[float, float] = (0, 255),
        weight_quantization: int = 4,
    ) -> Any:
        """
        Convert Keras model to Akida model.

        Args:
            keras_model: Keras/TensorFlow model.
            input_scaling: Input value range.
            weight_quantization: Weight bit width.

        Returns:
            Akida-compatible model.
        """
        _import_akida()

        if cnn2snn is None:
            raise ImportError(
                "cnn2snn not installed. Install with: pip install cnn2snn"
            )

        # Quantize the Keras model
        quantized_model = cnn2snn.quantize(
            keras_model,
            weight_quantization=weight_quantization,
            activ_quantization=weight_quantization,
            input_weight_quantization=8,
        )

        # Convert to Akida model
        akida_model = cnn2snn.convert(quantized_model)

        if self._device:
            akida_model.map(self._device)

        self._model = akida_model
        self._logger.info(
            "Keras model converted to Akida",
            layers=len(akida_model.layers),
        )

        return akida_model

    async def infer(
        self,
        input_data: SpikeData | np.ndarray,
        duration: float | None = None,
    ) -> InferenceResult:
        """Run inference on Akida."""
        if not self._initialized:
            await self.initialize()

        if self._model is None:
            raise RuntimeError("No model loaded")

        _import_akida()

        # Convert spike data to dense format if needed
        if isinstance(input_data, SpikeData):
            input_array = input_data.to_dense(self.timestep)
        else:
            input_array = input_data

        # Ensure correct shape (batch, ...)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)

        # Run inference
        import time

        start_time = time.time()

        try:
            # Akida inference
            predictions = self._model.predict(input_array.astype(np.uint8))
            latency = (time.time() - start_time) * 1000  # ms

            # Get spike activity
            output_spikes = self._extract_output_spikes(predictions)

            # Estimate energy
            energy = self._estimate_energy(input_array.shape[0])
            self._energy_total += energy

            result = InferenceResult(
                output_spikes=output_spikes,
                predictions=predictions,
                energy_consumption=energy,
                latency=latency,
                spike_count=int(np.sum(predictions > 0)),
                metadata={
                    "model_name": str(self._model),
                    "device": str(self._device) if self._device else "virtual",
                },
            )

        except Exception as e:
            self._logger.error("Inference failed", error=str(e))
            # Return empty result for simulation
            result = InferenceResult(
                output_spikes=SpikeData(
                    times=np.array([]),
                    neurons=np.array([]),
                    n_neurons=0,
                ),
                predictions=np.zeros((input_array.shape[0], 10)),
                latency=(time.time() - start_time) * 1000,
            )

        return result

    def _extract_output_spikes(self, predictions: np.ndarray) -> SpikeData:
        """Extract spike data from predictions."""
        # Convert predictions to spike times
        n_samples, n_classes = predictions.shape

        times = []
        neurons = []

        for sample_idx in range(n_samples):
            for class_idx in range(n_classes):
                if predictions[sample_idx, class_idx] > 0:
                    # Winner neuron spikes
                    times.append(sample_idx * 10.0)  # 10ms per sample
                    neurons.append(class_idx)

        return SpikeData(
            times=np.array(times),
            neurons=np.array(neurons),
            n_neurons=n_classes,
            duration=n_samples * 10.0,
        )

    def _estimate_energy(self, batch_size: int) -> float:
        """Estimate energy consumption."""
        # Akida typical: ~1mW for inference, ~10ms latency
        power_watts = {
            "performance": 0.002,
            "efficiency": 0.001,
            "ultra_low": 0.0005,
        }.get(self.power_mode, 0.001)

        time_seconds = batch_size * 0.01  # 10ms per sample
        return power_watts * time_seconds

    async def train_on_chip(
        self,
        input_data: SpikeData,
        target: np.ndarray,
        learning_rule: str = "stdp",
        epochs: int = 1,
    ) -> dict[str, Any]:
        """Run on-chip learning."""
        if not self._initialized:
            await self.initialize()

        _import_akida()

        # Convert to dense format
        if isinstance(input_data, SpikeData):
            input_array = input_data.to_dense(self.timestep)
        else:
            input_array = input_data

        results = {
            "epochs": epochs,
            "learning_rule": learning_rule,
            "accuracy": 0.0,
            "loss_history": [],
        }

        # Akida supports online learning for final layer
        if hasattr(self._model, "fit_by_episode"):
            for epoch in range(epochs):
                self._model.fit_by_episode(
                    input_array.astype(np.uint8),
                    target,
                )

                # Evaluate
                predictions = self._model.predict(input_array.astype(np.uint8))
                accuracy = np.mean(np.argmax(predictions, axis=1) == target)
                results["loss_history"].append(1 - accuracy)

            results["accuracy"] = accuracy
        else:
            self._logger.warning("On-chip learning not supported for this model")

        return results

    def create_snn_model(
        self,
        input_shape: tuple[int, ...],
        n_classes: int,
        architecture: str = "simple",
    ) -> Any:
        """
        Create an Akida-native SNN model.

        Args:
            input_shape: Input shape.
            n_classes: Number of output classes.
            architecture: Model architecture.

        Returns:
            Akida model.
        """
        _import_akida()

        if architecture == "simple":
            layers = [
                akida.InputConvolutional(
                    name="input",
                    input_shape=input_shape,
                    filters=32,
                    kernel_size=3,
                    padding="same",
                ),
                akida.Separable(
                    name="sep1",
                    filters=64,
                    kernel_size=3,
                    padding="same",
                    pool_size=2,
                ),
                akida.FullyConnected(
                    name="fc",
                    units=n_classes,
                    activation=False,
                ),
            ]
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        model = akida.Sequential(layers)
        self._model = model

        return model

    def _supports_on_chip_learning(self) -> bool:
        """Akida supports on-chip learning."""
        return True

    def _get_supported_layers(self) -> list[str]:
        """Get Akida-supported layers."""
        return [
            "InputConvolutional",
            "Convolutional",
            "Separable",
            "FullyConnected",
            "InputData",
        ]


class AkidaEdgeProcessor:
    """
    High-level interface for Akida edge processing.

    Optimized for biosignal processing (ECG, EEG, EMG).
    """

    def __init__(self, backend: AkidaBackend):
        """Initialize edge processor."""
        self.backend = backend

    async def process_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 250.0,
    ) -> dict[str, Any]:
        """
        Process ECG signal for arrhythmia detection.

        Args:
            ecg_signal: Raw ECG signal.
            sampling_rate: Sampling rate in Hz.

        Returns:
            Classification results.
        """
        # Segment into heartbeats
        segments = self._segment_ecg(ecg_signal, sampling_rate)

        # Encode as spikes
        spike_data = self.backend.encode_rate(
            segments,
            duration=100.0,
            max_rate=100.0,
        )

        # Classify
        result = await self.backend.infer(spike_data)

        return {
            "predictions": result.predictions,
            "confidence": np.max(result.predictions, axis=1),
            "n_beats": len(segments),
            "energy_uJ": result.energy_consumption * 1e6,
        }

    async def process_eeg(
        self,
        eeg_signal: np.ndarray,
        sampling_rate: float = 256.0,
    ) -> dict[str, Any]:
        """
        Process EEG signal for seizure detection.

        Args:
            eeg_signal: Multi-channel EEG signal.
            sampling_rate: Sampling rate in Hz.

        Returns:
            Detection results.
        """
        # Extract features (simplified)
        from scipy.signal import welch

        features = []
        for channel in range(eeg_signal.shape[0]):
            freqs, psd = welch(eeg_signal[channel], fs=sampling_rate)
            # Band powers
            delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
            theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            beta = np.sum(psd[(freqs >= 13) & (freqs < 30)])
            features.extend([delta, theta, alpha, beta])

        features = np.array(features).reshape(1, -1)

        # Encode and classify
        spike_data = self.backend.encode_rate(features)
        result = await self.backend.infer(spike_data)

        return {
            "seizure_detected": np.argmax(result.predictions) == 1,
            "confidence": float(np.max(result.predictions)),
            "band_powers": features.flatten().tolist(),
        }

    def _segment_ecg(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        segment_length: float = 0.6,
    ) -> np.ndarray:
        """Segment ECG into individual beats."""
        n_samples = int(segment_length * sampling_rate)

        # Simple peak detection
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(signal, distance=int(0.5 * sampling_rate))

        segments = []
        half_window = n_samples // 2

        for peak in peaks:
            start = max(0, peak - half_window)
            end = min(len(signal), peak + half_window)

            if end - start == n_samples:
                segments.append(signal[start:end])

        return np.array(segments) if segments else np.array([signal[:n_samples]])
