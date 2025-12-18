"""
Software Simulator Backend for QBitaLabs

Provides a pure NumPy-based neuromorphic simulator for testing.
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


class SimulatorBackend(BaseNeuromorphicBackend):
    """
    Software neuromorphic simulator.

    Features:
    - Multiple neuron models (LIF, ALIF, Izhikevich)
    - Flexible network architecture
    - STDP learning
    - No hardware dependencies

    Example:
        >>> backend = SimulatorBackend()
        >>> await backend.initialize()
        >>> network = backend.create_network([784, 256, 10])
        >>> result = await backend.infer(spike_data)
    """

    def __init__(
        self,
        neuron_model: NeuronModel = NeuronModel.LIF,
        timestep: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize simulator.

        Args:
            neuron_model: Neuron model to use.
            timestep: Simulation timestep in ms.
            seed: Random seed.
        """
        super().__init__(
            backend_type=BackendType.SIMULATOR,
            neuron_model=neuron_model,
            timestep=timestep,
        )

        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._network: dict[str, Any] | None = None

    async def initialize(self) -> None:
        """Initialize simulator."""
        self._initialized = True
        self._logger.info(
            "Simulator backend initialized",
            neuron_model=self.neuron_model.value,
        )

    async def shutdown(self) -> None:
        """Shutdown simulator."""
        self._network = None
        self._initialized = False
        self._logger.info("Simulator backend shutdown")

    def load_model(self, model_path: str) -> Any:
        """Load a network from file."""
        import pickle

        with open(model_path, "rb") as f:
            self._network = pickle.load(f)

        self._logger.info("Model loaded", path=model_path)
        return self._network

    def save_model(self, model_path: str) -> None:
        """Save network to file."""
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(self._network, f)

        self._logger.info("Model saved", path=model_path)

    def compile_model(self, model: Any, **kwargs: Any) -> Any:
        """Compile model (no-op for simulator)."""
        self._network = model
        return model

    def create_network(
        self,
        layer_sizes: list[int],
        connection_prob: float = 0.5,
        weight_scale: float = 0.1,
        excitatory_ratio: float = 0.8,
    ) -> dict[str, Any]:
        """
        Create a spiking neural network.

        Args:
            layer_sizes: Neurons per layer.
            connection_prob: Connection probability.
            weight_scale: Weight initialization scale.
            excitatory_ratio: Ratio of excitatory neurons.

        Returns:
            Network dictionary.
        """
        network = {
            "layers": [],
            "connections": [],
            "neuron_model": self.neuron_model.value,
            "timestep": self.timestep,
        }

        for i, size in enumerate(layer_sizes):
            # Determine excitatory/inhibitory
            n_excitatory = int(size * excitatory_ratio)

            layer = {
                "name": f"layer_{i}",
                "n_neurons": size,
                "n_excitatory": n_excitatory,
                "n_inhibitory": size - n_excitatory,
                "parameters": self._get_layer_params(),
            }
            network["layers"].append(layer)

            if i > 0:
                # Create connection matrix
                pre_size = layer_sizes[i - 1]
                weights = self._rng.randn(pre_size, size) * weight_scale
                mask = self._rng.random((pre_size, size)) < connection_prob
                weights *= mask

                # Separate excitatory/inhibitory
                weights[:, n_excitatory:] *= -1  # Inhibitory weights negative

                connection = {
                    "source": f"layer_{i - 1}",
                    "target": f"layer_{i}",
                    "weights": weights,
                    "delays": self._rng.integers(1, 5, size=(pre_size, size)),
                }
                network["connections"].append(connection)

        self._network = network
        self._logger.info(
            "Network created",
            layers=len(layer_sizes),
            total_neurons=sum(layer_sizes),
        )

        return network

    def _get_layer_params(self) -> dict[str, float]:
        """Get default layer parameters."""
        if self.neuron_model == NeuronModel.LIF:
            return {
                "tau_mem": 20.0,  # Membrane time constant (ms)
                "tau_syn": 5.0,  # Synaptic time constant (ms)
                "v_rest": -65.0,  # Resting potential (mV)
                "v_thresh": -50.0,  # Threshold potential (mV)
                "v_reset": -65.0,  # Reset potential (mV)
                "refrac": 2.0,  # Refractory period (ms)
            }

        elif self.neuron_model == NeuronModel.ALIF:
            return {
                "tau_mem": 20.0,
                "tau_syn": 5.0,
                "tau_adapt": 100.0,
                "v_rest": -65.0,
                "v_thresh": -50.0,
                "v_reset": -65.0,
                "refrac": 2.0,
                "adapt_increment": 0.1,
            }

        elif self.neuron_model == NeuronModel.IZHIKEVICH:
            return {
                "a": 0.02,
                "b": 0.2,
                "c": -65.0,
                "d": 8.0,
            }

        return {}

    async def infer(
        self,
        input_data: SpikeData | np.ndarray,
        duration: float | None = None,
    ) -> InferenceResult:
        """Run inference simulation."""
        if not self._initialized:
            await self.initialize()

        if self._network is None:
            raise RuntimeError("No network created")

        # Convert input
        if isinstance(input_data, SpikeData):
            spike_matrix = input_data.to_dense(self.timestep)
        else:
            spike_matrix = input_data

        duration = duration or spike_matrix.shape[1] * self.timestep
        n_timesteps = int(duration / self.timestep)

        import time

        start_time = time.time()

        # Simulate
        if self.neuron_model == NeuronModel.LIF:
            output_spikes, spike_count = self._simulate_lif(spike_matrix, n_timesteps)
        elif self.neuron_model == NeuronModel.ALIF:
            output_spikes, spike_count = self._simulate_alif(spike_matrix, n_timesteps)
        elif self.neuron_model == NeuronModel.IZHIKEVICH:
            output_spikes, spike_count = self._simulate_izhikevich(spike_matrix, n_timesteps)
        else:
            output_spikes, spike_count = self._simulate_lif(spike_matrix, n_timesteps)

        latency = (time.time() - start_time) * 1000

        # Decode predictions
        predictions = self._decode_output(output_spikes)

        # Estimate energy (software overhead only)
        energy = self._estimate_energy(spike_count)
        self._energy_total += energy

        result = InferenceResult(
            output_spikes=SpikeData.from_dense(output_spikes, self.timestep),
            predictions=predictions,
            energy_consumption=energy,
            latency=latency,
            spike_count=spike_count,
        )

        return result

    def _simulate_lif(
        self,
        input_spikes: np.ndarray,
        n_timesteps: int,
    ) -> tuple[np.ndarray, int]:
        """Simulate LIF network."""
        layers = self._network["layers"]
        connections = self._network["connections"]

        # Initialize state
        state = {}
        for layer in layers:
            params = layer["parameters"]
            n = layer["n_neurons"]
            state[layer["name"]] = {
                "v": np.ones(n) * params["v_rest"],
                "i_syn": np.zeros(n),
                "refrac_counter": np.zeros(n),
            }

        # Output storage
        output_layer = layers[-1]
        n_output = output_layer["n_neurons"]
        output_spikes = np.zeros((n_output, n_timesteps))
        total_spike_count = 0

        for t in range(n_timesteps):
            # Input layer
            if t < input_spikes.shape[1]:
                input_current = input_spikes[:, t]
            else:
                input_current = np.zeros(input_spikes.shape[0])

            prev_spikes = input_current

            for i, layer in enumerate(layers):
                name = layer["name"]
                params = layer["parameters"]
                s = state[name]

                # Compute incoming current
                incoming = np.zeros(layer["n_neurons"])
                for conn in connections:
                    if conn["target"] == name:
                        incoming += conn["weights"].T @ prev_spikes

                # Add input current for first layer
                if i == 0 and len(input_current) > 0:
                    n_input = min(len(input_current), layer["n_neurons"])
                    incoming[:n_input] += input_current[:n_input]

                # Update synaptic current
                s["i_syn"] *= np.exp(-self.timestep / params["tau_syn"])
                s["i_syn"] += incoming

                # Update membrane potential (only non-refractory neurons)
                not_refrac = s["refrac_counter"] <= 0
                dv = (
                    -(s["v"] - params["v_rest"]) / params["tau_mem"]
                    + s["i_syn"]
                ) * self.timestep
                s["v"][not_refrac] += dv[not_refrac]

                # Decrease refractory counter
                s["refrac_counter"] = np.maximum(0, s["refrac_counter"] - self.timestep)

                # Check for spikes
                spikes = s["v"] >= params["v_thresh"]
                s["v"][spikes] = params["v_reset"]
                s["refrac_counter"][spikes] = params["refrac"]

                prev_spikes = spikes.astype(float)
                total_spike_count += np.sum(spikes)

            # Record output
            output_spikes[:, t] = prev_spikes[:n_output]

        return output_spikes, total_spike_count

    def _simulate_alif(
        self,
        input_spikes: np.ndarray,
        n_timesteps: int,
    ) -> tuple[np.ndarray, int]:
        """Simulate ALIF network."""
        # Similar to LIF but with adaptive threshold
        layers = self._network["layers"]
        connections = self._network["connections"]

        state = {}
        for layer in layers:
            params = layer["parameters"]
            n = layer["n_neurons"]
            state[layer["name"]] = {
                "v": np.ones(n) * params["v_rest"],
                "i_syn": np.zeros(n),
                "thresh": np.ones(n) * params["v_thresh"],
                "refrac_counter": np.zeros(n),
            }

        output_layer = layers[-1]
        n_output = output_layer["n_neurons"]
        output_spikes = np.zeros((n_output, n_timesteps))
        total_spike_count = 0

        for t in range(n_timesteps):
            if t < input_spikes.shape[1]:
                input_current = input_spikes[:, t]
            else:
                input_current = np.zeros(input_spikes.shape[0])

            prev_spikes = input_current

            for i, layer in enumerate(layers):
                name = layer["name"]
                params = layer["parameters"]
                s = state[name]

                incoming = np.zeros(layer["n_neurons"])
                for conn in connections:
                    if conn["target"] == name:
                        incoming += conn["weights"].T @ prev_spikes

                if i == 0 and len(input_current) > 0:
                    n_input = min(len(input_current), layer["n_neurons"])
                    incoming[:n_input] += input_current[:n_input]

                # Threshold adaptation
                s["thresh"] += (-s["thresh"] + params["v_thresh"]) / params["tau_adapt"] * self.timestep

                # Synaptic current
                s["i_syn"] *= np.exp(-self.timestep / params["tau_syn"])
                s["i_syn"] += incoming

                # Membrane potential
                not_refrac = s["refrac_counter"] <= 0
                dv = (
                    -(s["v"] - params["v_rest"]) / params["tau_mem"]
                    + s["i_syn"]
                ) * self.timestep
                s["v"][not_refrac] += dv[not_refrac]

                s["refrac_counter"] = np.maximum(0, s["refrac_counter"] - self.timestep)

                # Spikes
                spikes = s["v"] >= s["thresh"]
                s["v"][spikes] = params["v_reset"]
                s["thresh"][spikes] += params["adapt_increment"]
                s["refrac_counter"][spikes] = params["refrac"]

                prev_spikes = spikes.astype(float)
                total_spike_count += np.sum(spikes)

            output_spikes[:, t] = prev_spikes[:n_output]

        return output_spikes, total_spike_count

    def _simulate_izhikevich(
        self,
        input_spikes: np.ndarray,
        n_timesteps: int,
    ) -> tuple[np.ndarray, int]:
        """Simulate Izhikevich network."""
        layers = self._network["layers"]
        connections = self._network["connections"]

        state = {}
        for layer in layers:
            params = layer["parameters"]
            n = layer["n_neurons"]
            state[layer["name"]] = {
                "v": np.ones(n) * params["c"],
                "u": np.ones(n) * params["b"] * params["c"],
                "i": np.zeros(n),
            }

        output_layer = layers[-1]
        n_output = output_layer["n_neurons"]
        output_spikes = np.zeros((n_output, n_timesteps))
        total_spike_count = 0

        for t in range(n_timesteps):
            if t < input_spikes.shape[1]:
                input_current = input_spikes[:, t] * 10
            else:
                input_current = np.zeros(input_spikes.shape[0])

            prev_spikes = input_current / 10

            for i, layer in enumerate(layers):
                name = layer["name"]
                params = layer["parameters"]
                s = state[name]

                incoming = np.zeros(layer["n_neurons"])
                for conn in connections:
                    if conn["target"] == name:
                        incoming += conn["weights"].T @ prev_spikes * 10

                if i == 0 and len(input_current) > 0:
                    n_input = min(len(input_current), layer["n_neurons"])
                    incoming[:n_input] += input_current[:n_input]

                s["i"] = incoming

                # Izhikevich equations
                dv = 0.04 * s["v"] ** 2 + 5 * s["v"] + 140 - s["u"] + s["i"]
                du = params["a"] * (params["b"] * s["v"] - s["u"])

                s["v"] += dv * self.timestep
                s["u"] += du * self.timestep

                # Spikes
                spikes = s["v"] >= 30
                s["v"][spikes] = params["c"]
                s["u"][spikes] += params["d"]

                prev_spikes = spikes.astype(float)
                total_spike_count += np.sum(spikes)

            output_spikes[:, t] = prev_spikes[:n_output]

        return output_spikes, total_spike_count

    def _decode_output(self, output_spikes: np.ndarray) -> np.ndarray:
        """Decode output spikes."""
        spike_counts = np.sum(output_spikes, axis=1)
        total = np.sum(spike_counts)
        if total > 0:
            return (spike_counts / total).reshape(1, -1)
        return spike_counts.reshape(1, -1)

    def _estimate_energy(self, spike_count: int) -> float:
        """Estimate computational energy."""
        # Software simulation: ~1nJ per spike operation
        return spike_count * 1e-9

    async def train_on_chip(
        self,
        input_data: SpikeData,
        target: np.ndarray,
        learning_rule: str = "stdp",
        epochs: int = 1,
    ) -> dict[str, Any]:
        """Train using STDP."""
        if self._network is None:
            raise RuntimeError("No network created")

        results = {
            "epochs": epochs,
            "learning_rule": learning_rule,
            "accuracy_history": [],
        }

        if isinstance(input_data, SpikeData):
            spike_matrix = input_data.to_dense(self.timestep)
        else:
            spike_matrix = input_data

        for epoch in range(epochs):
            # Run forward pass
            output_spikes, _ = self._simulate_lif(spike_matrix, spike_matrix.shape[1])

            # Apply STDP to connections
            self._apply_stdp(spike_matrix, output_spikes)

            # Evaluate
            predictions = self._decode_output(output_spikes)
            accuracy = float(np.argmax(predictions) == target[0])
            results["accuracy_history"].append(accuracy)

        return results

    def _apply_stdp(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ) -> None:
        """Apply STDP learning rule."""
        for conn in self._network["connections"]:
            weights = conn["weights"]

            # Compute spike times
            pre_times = np.where(pre_spikes)[1] if pre_spikes.ndim > 1 else np.where(pre_spikes)[0]
            post_times = np.where(post_spikes)[1] if post_spikes.ndim > 1 else np.where(post_spikes)[0]

            # STDP weight update (simplified)
            if len(pre_times) > 0 and len(post_times) > 0:
                dt = np.mean(post_times) - np.mean(pre_times)
                if dt > 0:
                    dw = a_plus * np.exp(-dt / tau_plus)
                else:
                    dw = -a_minus * np.exp(dt / tau_minus)

                conn["weights"] += dw * 0.01  # Small learning rate

    def _supports_on_chip_learning(self) -> bool:
        """Simulator supports STDP."""
        return True
