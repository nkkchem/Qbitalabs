"""
Intel Loihi Backend for QBitaLabs

Provides integration with Intel's Loihi neuromorphic research chip.
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
nxsdk = None
lava = None


def _import_loihi() -> None:
    """Lazily import Loihi SDK."""
    global nxsdk, lava
    if nxsdk is None and lava is None:
        try:
            import lava.lib.dl.slayer as _lava

            lava = _lava
        except ImportError:
            pass

        try:
            import nxsdk as _nxsdk

            nxsdk = _nxsdk
        except ImportError:
            pass

        if nxsdk is None and lava is None:
            logger.warning(
                "Neither NxSDK nor Lava installed. Using simulation mode."
            )


class LoihiBackend(BaseNeuromorphicBackend):
    """
    Intel Loihi neuromorphic backend.

    Features:
    - Programmable neuron models
    - On-chip learning with STDP
    - Multi-compartment neurons
    - Graded spikes
    - Lava framework support

    Example:
        >>> backend = LoihiBackend()
        >>> await backend.initialize()
        >>> model = backend.create_network(topology)
        >>> result = await backend.infer(spike_data)
    """

    def __init__(
        self,
        chip_id: int = 0,
        neuron_model: NeuronModel = NeuronModel.LIF,
        timestep: float = 1.0,
        use_lava: bool = True,
    ):
        """
        Initialize Loihi backend.

        Args:
            chip_id: Loihi chip ID.
            neuron_model: Neuron model to use.
            timestep: Simulation timestep.
            use_lava: Use Lava framework (vs NxSDK).
        """
        super().__init__(
            backend_type=BackendType.LOIHI,
            neuron_model=neuron_model,
            timestep=timestep,
            n_neurons_max=130000,  # Loihi 2 has 1M neurons
        )

        self.chip_id = chip_id
        self.use_lava = use_lava
        self._network = None
        self._probes: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize Loihi connection."""
        _import_loihi()

        if self.use_lava and lava is not None:
            self._logger.info("Using Lava framework")
        elif nxsdk is not None:
            try:
                # Connect to Loihi board
                self._board = nxsdk.N2Board(
                    self.chip_id,
                    numCores=128,
                    numSynapsesPerCore=1024,
                )
                self._logger.info("Connected to Loihi hardware")
            except Exception as e:
                self._logger.warning(
                    "Failed to connect to Loihi, using simulation",
                    error=str(e),
                )
        else:
            self._logger.info("Running in simulation mode")

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown Loihi connection."""
        if hasattr(self, "_board") and self._board is not None:
            self._board.disconnect()

        self._network = None
        self._probes.clear()
        self._initialized = False
        self._logger.info("Loihi backend shutdown")

    def load_model(self, model_path: str) -> Any:
        """Load a compiled Loihi model."""
        _import_loihi()

        if self.use_lava and lava is not None:
            # Load Lava model
            import torch

            self._model = torch.load(model_path)
        else:
            # Load NxSDK network
            import pickle

            with open(model_path, "rb") as f:
                self._network = pickle.load(f)

        self._logger.info("Model loaded", path=model_path)
        return self._model if hasattr(self, "_model") else self._network

    def compile_model(self, model: Any, **kwargs: Any) -> Any:
        """Compile a model for Loihi."""
        _import_loihi()

        if self.use_lava and lava is not None:
            return self._compile_lava_model(model, **kwargs)
        else:
            return self._compile_nxsdk_model(model, **kwargs)

    def _compile_lava_model(self, model: Any, **kwargs: Any) -> Any:
        """Compile model using Lava."""
        # Lava compilation
        self._model = model
        return model

    def _compile_nxsdk_model(self, model: Any, **kwargs: Any) -> Any:
        """Compile model using NxSDK."""
        # NxSDK compilation
        self._network = model
        return model

    def create_network(
        self,
        layer_sizes: list[int],
        connection_prob: float = 0.1,
        weight_init: str = "random",
    ) -> Any:
        """
        Create a Loihi SNN network.

        Args:
            layer_sizes: Number of neurons per layer.
            connection_prob: Connection probability between layers.
            weight_init: Weight initialization method.

        Returns:
            Network object.
        """
        _import_loihi()

        if self.use_lava and lava is not None:
            return self._create_lava_network(
                layer_sizes, connection_prob, weight_init
            )
        else:
            return self._create_nxsdk_network(
                layer_sizes, connection_prob, weight_init
            )

    def _create_lava_network(
        self,
        layer_sizes: list[int],
        connection_prob: float,
        weight_init: str,
    ) -> Any:
        """Create network using Lava."""
        # Simplified network structure
        network = {
            "layers": [],
            "connections": [],
        }

        for i, size in enumerate(layer_sizes):
            layer = {
                "name": f"layer_{i}",
                "n_neurons": size,
                "neuron_model": self.neuron_model.value,
                "parameters": self._get_neuron_params(),
            }
            network["layers"].append(layer)

            if i > 0:
                # Connect to previous layer
                weights = np.random.randn(layer_sizes[i - 1], size) * 0.1
                mask = np.random.random((layer_sizes[i - 1], size)) < connection_prob
                weights *= mask

                connection = {
                    "source": f"layer_{i - 1}",
                    "target": f"layer_{i}",
                    "weights": weights,
                }
                network["connections"].append(connection)

        self._network = network
        return network

    def _create_nxsdk_network(
        self,
        layer_sizes: list[int],
        connection_prob: float,
        weight_init: str,
    ) -> Any:
        """Create network using NxSDK."""
        if nxsdk is None:
            return self._create_lava_network(
                layer_sizes, connection_prob, weight_init
            )

        # NxSDK network creation
        net = nxsdk.NxNet()

        prototype = nxsdk.NxProbe()
        compartment_prototype = nxsdk.CompartmentPrototype(
            vThMant=100,
            compartmentCurrentDecay=4096,
            compartmentVoltageDecay=256,
        )

        layers = []
        for i, size in enumerate(layer_sizes):
            cg = net.createCompartmentGroup(
                size=size,
                prototype=compartment_prototype,
            )
            layers.append(cg)

            if i > 0:
                # Create connections
                weights = np.random.randint(0, 255, (layer_sizes[i - 1], size))
                mask = np.random.random((layer_sizes[i - 1], size)) < connection_prob
                weights *= mask.astype(int)

                layers[i - 1].connect(
                    cg,
                    prototype=nxsdk.ConnectionPrototype(weight=weights),
                )

        self._network = {"net": net, "layers": layers}
        return self._network

    def _get_neuron_params(self) -> dict[str, Any]:
        """Get neuron parameters for current model."""
        if self.neuron_model == NeuronModel.LIF:
            return {
                "threshold": 1.0,
                "tau_mem": 20.0,  # ms
                "tau_syn": 5.0,  # ms
                "reset_voltage": 0.0,
            }
        elif self.neuron_model == NeuronModel.ALIF:
            return {
                "threshold": 1.0,
                "tau_mem": 20.0,
                "tau_syn": 5.0,
                "tau_adapt": 100.0,
                "adapt_increment": 0.1,
            }
        else:
            return {"threshold": 1.0}

    async def infer(
        self,
        input_data: SpikeData | np.ndarray,
        duration: float | None = None,
    ) -> InferenceResult:
        """Run inference on Loihi."""
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

        import time

        start_time = time.time()

        # Simulate network
        output_spikes = self._simulate_network(spike_matrix, int(duration / self.timestep))

        latency = (time.time() - start_time) * 1000

        # Estimate energy
        energy = self._estimate_energy(spike_matrix, output_spikes)
        self._energy_total += energy

        # Decode output
        predictions = self._decode_output(output_spikes)

        result = InferenceResult(
            output_spikes=SpikeData.from_dense(output_spikes, self.timestep),
            predictions=predictions,
            energy_consumption=energy,
            latency=latency,
            spike_count=int(np.sum(output_spikes)),
        )

        return result

    def _simulate_network(
        self,
        input_spikes: np.ndarray,
        n_timesteps: int,
    ) -> np.ndarray:
        """Simulate network (software fallback)."""
        if self._network is None:
            return np.zeros((10, n_timesteps))

        layers = self._network.get("layers", [])
        connections = self._network.get("connections", [])

        if not layers:
            return np.zeros((10, n_timesteps))

        # Initialize state
        n_output = layers[-1]["n_neurons"]
        output_spikes = np.zeros((n_output, n_timesteps))

        # LIF simulation
        states = {}
        for layer in layers:
            states[layer["name"]] = {
                "voltage": np.zeros(layer["n_neurons"]),
                "current": np.zeros(layer["n_neurons"]),
            }

        # Get neuron parameters
        params = self._get_neuron_params()
        threshold = params["threshold"]
        tau_mem = params["tau_mem"]
        tau_syn = params["tau_syn"]

        for t in range(n_timesteps):
            # Input layer
            if t < input_spikes.shape[1]:
                input_current = input_spikes[:, t]
            else:
                input_current = np.zeros(input_spikes.shape[0])

            # Process each layer
            prev_spikes = input_current
            for i, layer in enumerate(layers):
                name = layer["name"]
                state = states[name]

                # Find incoming connection
                incoming_current = np.zeros(layer["n_neurons"])
                for conn in connections:
                    if conn["target"] == name:
                        incoming_current += conn["weights"].T @ prev_spikes

                # Add input for first layer
                if i == 0:
                    incoming_current += input_current[:layer["n_neurons"]] if len(input_current) >= layer["n_neurons"] else np.pad(input_current, (0, layer["n_neurons"] - len(input_current)))

                # Update synaptic current
                state["current"] += (-state["current"] / tau_syn + incoming_current) * self.timestep

                # Update membrane voltage
                state["voltage"] += (-state["voltage"] / tau_mem + state["current"]) * self.timestep

                # Check for spikes
                spikes = (state["voltage"] >= threshold).astype(float)
                state["voltage"][spikes > 0] = 0  # Reset

                prev_spikes = spikes

            # Record output
            output_spikes[:, t] = prev_spikes[:n_output]

        return output_spikes

    def _decode_output(self, output_spikes: np.ndarray) -> np.ndarray:
        """Decode output spikes to predictions."""
        # Rate-based decoding
        spike_counts = np.sum(output_spikes, axis=1)
        if np.sum(spike_counts) > 0:
            return spike_counts / np.sum(spike_counts)
        return spike_counts

    def _estimate_energy(
        self,
        input_spikes: np.ndarray,
        output_spikes: np.ndarray,
    ) -> float:
        """Estimate energy consumption."""
        # Loihi: ~25pJ per synaptic operation
        total_spikes = np.sum(input_spikes) + np.sum(output_spikes)

        if self._network:
            n_synapses = sum(
                conn["weights"].size for conn in self._network.get("connections", [])
            )
        else:
            n_synapses = 1000

        # Energy model
        energy_per_spike = 25e-12  # 25 pJ
        return float(total_spikes * energy_per_spike * n_synapses / 1000)

    async def train_on_chip(
        self,
        input_data: SpikeData,
        target: np.ndarray,
        learning_rule: str = "stdp",
        epochs: int = 1,
    ) -> dict[str, Any]:
        """Run on-chip learning with STDP."""
        if not self._initialized:
            await self.initialize()

        # STDP parameters
        stdp_params = {
            "a_plus": 0.1,
            "a_minus": 0.12,
            "tau_plus": 20.0,  # ms
            "tau_minus": 20.0,  # ms
        }

        results = {
            "epochs": epochs,
            "learning_rule": learning_rule,
            "weight_changes": [],
        }

        # Convert input
        if isinstance(input_data, SpikeData):
            spike_matrix = input_data.to_dense(self.timestep)
        else:
            spike_matrix = input_data

        for epoch in range(epochs):
            total_weight_change = 0.0

            # Apply STDP
            if self._network and "connections" in self._network:
                for conn in self._network["connections"]:
                    delta_w = self._apply_stdp(
                        conn["weights"],
                        spike_matrix,
                        stdp_params,
                    )
                    conn["weights"] += delta_w
                    total_weight_change += np.sum(np.abs(delta_w))

            results["weight_changes"].append(total_weight_change)

        return results

    def _apply_stdp(
        self,
        weights: np.ndarray,
        spikes: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        """Apply STDP learning rule."""
        # Simplified STDP
        pre_rates = np.mean(spikes, axis=1)
        post_rates = np.ones(weights.shape[1]) * 0.1  # Placeholder

        # Potentiation and depression
        delta_w = np.outer(pre_rates, post_rates) * params["a_plus"]
        delta_w -= np.outer(pre_rates, np.ones_like(post_rates)) * params["a_minus"]

        return delta_w * 0.01  # Small learning rate

    def _supports_on_chip_learning(self) -> bool:
        """Loihi supports on-chip STDP."""
        return True

    def _get_supported_layers(self) -> list[str]:
        """Get Loihi-supported operations."""
        return [
            "dense",
            "conv2d",
            "recurrent",
            "spike_input",
            "spike_output",
            "stdp_learning",
        ]
