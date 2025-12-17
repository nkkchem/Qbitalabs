"""
Spiking Neural Network Components for QBitaLabs

Provides building blocks for SNN construction:
- Neuron models (LIF, ALIF, Izhikevich)
- Synaptic connections
- Learning rules
- Network architectures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class NeuronType(str, Enum):
    """Types of spiking neurons."""

    LIF = "lif"
    ALIF = "alif"
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hh"


class SynapseType(str, Enum):
    """Types of synaptic connections."""

    STATIC = "static"
    STDP = "stdp"
    SHORT_TERM_PLASTICITY = "stp"


@dataclass
class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.

    The LIF model is one of the simplest spiking neuron models:
    τ_m dV/dt = -(V - V_rest) + R*I

    When V >= V_thresh: emit spike, V -> V_reset
    """

    neuron_id: int = 0
    tau_mem: float = 20.0  # Membrane time constant (ms)
    tau_syn: float = 5.0  # Synaptic time constant (ms)
    v_rest: float = -65.0  # Resting potential (mV)
    v_thresh: float = -50.0  # Threshold potential (mV)
    v_reset: float = -65.0  # Reset potential (mV)
    refrac_period: float = 2.0  # Refractory period (ms)

    # State
    v: float = field(default=-65.0, repr=False)
    i_syn: float = field(default=0.0, repr=False)
    refrac_counter: float = field(default=0.0, repr=False)
    spike: bool = field(default=False, repr=False)

    def reset(self) -> None:
        """Reset neuron state."""
        self.v = self.v_rest
        self.i_syn = 0.0
        self.refrac_counter = 0.0
        self.spike = False

    def step(self, i_input: float, dt: float = 1.0) -> bool:
        """
        Advance neuron by one timestep.

        Args:
            i_input: Input current.
            dt: Timestep in ms.

        Returns:
            True if neuron spiked.
        """
        self.spike = False

        # Update synaptic current
        self.i_syn = self.i_syn * np.exp(-dt / self.tau_syn) + i_input

        # Update refractory counter
        if self.refrac_counter > 0:
            self.refrac_counter -= dt
            return False

        # Update membrane potential
        dv = (-(self.v - self.v_rest) + self.i_syn) / self.tau_mem * dt
        self.v += dv

        # Check for spike
        if self.v >= self.v_thresh:
            self.spike = True
            self.v = self.v_reset
            self.refrac_counter = self.refrac_period
            return True

        return False


@dataclass
class ALIFNeuron(LIFNeuron):
    """
    Adaptive Leaky Integrate-and-Fire neuron.

    Extends LIF with an adaptive threshold that increases
    after each spike, modeling spike-frequency adaptation.
    """

    tau_adapt: float = 100.0  # Adaptation time constant (ms)
    adapt_increment: float = 0.5  # Threshold increase per spike

    # Adaptive state
    thresh_adapt: float = field(default=0.0, repr=False)

    def reset(self) -> None:
        """Reset neuron state."""
        super().reset()
        self.thresh_adapt = 0.0

    def step(self, i_input: float, dt: float = 1.0) -> bool:
        """Advance neuron with adaptive threshold."""
        self.spike = False

        # Update adaptive threshold
        self.thresh_adapt -= self.thresh_adapt / self.tau_adapt * dt

        # Update synaptic current
        self.i_syn = self.i_syn * np.exp(-dt / self.tau_syn) + i_input

        if self.refrac_counter > 0:
            self.refrac_counter -= dt
            return False

        # Effective threshold
        effective_thresh = self.v_thresh + self.thresh_adapt

        # Update membrane
        dv = (-(self.v - self.v_rest) + self.i_syn) / self.tau_mem * dt
        self.v += dv

        # Check for spike
        if self.v >= effective_thresh:
            self.spike = True
            self.v = self.v_reset
            self.refrac_counter = self.refrac_period
            self.thresh_adapt += self.adapt_increment
            return True

        return False


@dataclass
class SynapticConnection:
    """
    Connection between neurons with optional plasticity.

    Supports:
    - Static weights
    - STDP (Spike-Timing Dependent Plasticity)
    - Short-term plasticity
    """

    pre_id: int
    post_id: int
    weight: float = 0.1
    delay: int = 1  # Timesteps

    synapse_type: SynapseType = SynapseType.STATIC

    # STDP parameters
    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0

    # Short-term plasticity parameters
    u_se: float = 0.5  # Release probability
    tau_rec: float = 100.0  # Recovery time constant
    tau_fac: float = 50.0  # Facilitation time constant

    # State
    _pre_trace: float = field(default=0.0, repr=False)
    _post_trace: float = field(default=0.0, repr=False)
    _stp_x: float = field(default=1.0, repr=False)
    _stp_u: float = field(default=0.5, repr=False)

    def reset(self) -> None:
        """Reset synapse state."""
        self._pre_trace = 0.0
        self._post_trace = 0.0
        self._stp_x = 1.0
        self._stp_u = self.u_se

    def get_current(self, pre_spike: bool, dt: float = 1.0) -> float:
        """
        Get synaptic current for this timestep.

        Args:
            pre_spike: Whether presynaptic neuron spiked.
            dt: Timestep.

        Returns:
            Synaptic current.
        """
        if self.synapse_type == SynapseType.SHORT_TERM_PLASTICITY:
            return self._get_stp_current(pre_spike, dt)

        if pre_spike:
            return self.weight
        return 0.0

    def _get_stp_current(self, pre_spike: bool, dt: float) -> float:
        """Get current with short-term plasticity."""
        # Update STP variables
        self._stp_x += (1.0 - self._stp_x) / self.tau_rec * dt
        self._stp_u += (self.u_se - self._stp_u) / self.tau_fac * dt

        if pre_spike:
            current = self.weight * self._stp_u * self._stp_x
            self._stp_x -= self._stp_u * self._stp_x
            self._stp_u += self.u_se * (1 - self._stp_u)
            return current

        return 0.0

    def apply_stdp(
        self,
        pre_spike: bool,
        post_spike: bool,
        dt: float = 1.0,
    ) -> float:
        """
        Apply STDP learning rule.

        Args:
            pre_spike: Presynaptic spike occurred.
            post_spike: Postsynaptic spike occurred.
            dt: Timestep.

        Returns:
            Weight change.
        """
        if self.synapse_type != SynapseType.STDP:
            return 0.0

        # Update traces
        self._pre_trace *= np.exp(-dt / self.tau_plus)
        self._post_trace *= np.exp(-dt / self.tau_minus)

        dw = 0.0

        if pre_spike:
            # LTD: pre before post
            dw -= self.a_minus * self._post_trace
            self._pre_trace += 1.0

        if post_spike:
            # LTP: post after pre
            dw += self.a_plus * self._pre_trace
            self._post_trace += 1.0

        self.weight = np.clip(self.weight + dw, 0, 1)
        return dw


class SpikingNeuralNetwork:
    """
    Complete Spiking Neural Network implementation.

    Features:
    - Multiple neuron types
    - Configurable topology
    - Learning rules (STDP)
    - Spike recording

    Example:
        >>> snn = SpikingNeuralNetwork()
        >>> snn.add_layer("input", 784, NeuronType.LIF)
        >>> snn.add_layer("hidden", 256, NeuronType.ALIF)
        >>> snn.add_layer("output", 10, NeuronType.LIF)
        >>> snn.connect("input", "hidden", prob=0.3)
        >>> snn.connect("hidden", "output", prob=0.5)
        >>> output = snn.forward(input_spikes, duration=100)
    """

    def __init__(self, timestep: float = 1.0, seed: int | None = None):
        """
        Initialize SNN.

        Args:
            timestep: Simulation timestep in ms.
            seed: Random seed.
        """
        self.timestep = timestep
        self._rng = np.random.default_rng(seed)

        self._layers: dict[str, dict[str, Any]] = {}
        self._connections: list[dict[str, Any]] = []
        self._layer_order: list[str] = []

        self._spike_history: dict[str, list[np.ndarray]] = {}

        self._logger = structlog.get_logger("snn")

    def add_layer(
        self,
        name: str,
        n_neurons: int,
        neuron_type: NeuronType = NeuronType.LIF,
        **neuron_params: Any,
    ) -> None:
        """
        Add a layer to the network.

        Args:
            name: Layer name.
            n_neurons: Number of neurons.
            neuron_type: Type of neurons.
            **neuron_params: Neuron parameters.
        """
        neurons = []
        for i in range(n_neurons):
            if neuron_type == NeuronType.LIF:
                neuron = LIFNeuron(neuron_id=i, **neuron_params)
            elif neuron_type == NeuronType.ALIF:
                neuron = ALIFNeuron(neuron_id=i, **neuron_params)
            else:
                neuron = LIFNeuron(neuron_id=i, **neuron_params)
            neurons.append(neuron)

        self._layers[name] = {
            "neurons": neurons,
            "type": neuron_type,
            "n_neurons": n_neurons,
        }
        self._layer_order.append(name)
        self._spike_history[name] = []

        self._logger.debug("Layer added", name=name, neurons=n_neurons)

    def connect(
        self,
        source: str,
        target: str,
        prob: float = 0.5,
        weight_mean: float = 0.1,
        weight_std: float = 0.05,
        synapse_type: SynapseType = SynapseType.STATIC,
        delays: tuple[int, int] = (1, 5),
    ) -> None:
        """
        Connect two layers.

        Args:
            source: Source layer name.
            target: Target layer name.
            prob: Connection probability.
            weight_mean: Mean initial weight.
            weight_std: Weight standard deviation.
            synapse_type: Type of synapse.
            delays: (min, max) delays in timesteps.
        """
        if source not in self._layers or target not in self._layers:
            raise ValueError(f"Layer not found: {source} or {target}")

        src_neurons = self._layers[source]["n_neurons"]
        tgt_neurons = self._layers[target]["n_neurons"]

        synapses = []
        for i in range(src_neurons):
            for j in range(tgt_neurons):
                if self._rng.random() < prob:
                    weight = self._rng.normal(weight_mean, weight_std)
                    weight = max(0, weight)
                    delay = self._rng.integers(delays[0], delays[1] + 1)

                    synapse = SynapticConnection(
                        pre_id=i,
                        post_id=j,
                        weight=weight,
                        delay=delay,
                        synapse_type=synapse_type,
                    )
                    synapses.append(synapse)

        self._connections.append({
            "source": source,
            "target": target,
            "synapses": synapses,
        })

        self._logger.debug(
            "Layers connected",
            source=source,
            target=target,
            n_synapses=len(synapses),
        )

    def forward(
        self,
        input_spikes: np.ndarray,
        duration: float | None = None,
        record: bool = True,
    ) -> np.ndarray:
        """
        Run forward pass.

        Args:
            input_spikes: Input spike matrix (n_input x n_timesteps).
            duration: Override duration.
            record: Record spike history.

        Returns:
            Output spike matrix.
        """
        if len(self._layer_order) == 0:
            raise ValueError("No layers in network")

        input_layer = self._layer_order[0]
        output_layer = self._layer_order[-1]

        n_timesteps = input_spikes.shape[1] if duration is None else int(duration / self.timestep)
        n_output = self._layers[output_layer]["n_neurons"]

        output_spikes = np.zeros((n_output, n_timesteps))

        # Reset all neurons
        for layer_data in self._layers.values():
            for neuron in layer_data["neurons"]:
                neuron.reset()

        # Reset synapses
        for conn in self._connections:
            for synapse in conn["synapses"]:
                synapse.reset()

        # Clear history
        if record:
            for name in self._spike_history:
                self._spike_history[name] = []

        # Spike delay buffer
        delay_buffer: dict[str, dict[int, list[bool]]] = {}
        for conn in self._connections:
            key = f"{conn['source']}_{conn['target']}"
            max_delay = max(s.delay for s in conn["synapses"]) if conn["synapses"] else 1
            delay_buffer[key] = {d: [] for d in range(max_delay + 1)}

        # Simulation loop
        for t in range(n_timesteps):
            # Get input for this timestep
            if t < input_spikes.shape[1]:
                input_current = input_spikes[:, t]
            else:
                input_current = np.zeros(self._layers[input_layer]["n_neurons"])

            # Process each layer
            layer_spikes: dict[str, np.ndarray] = {}

            for layer_name in self._layer_order:
                layer = self._layers[layer_name]
                n = layer["n_neurons"]
                neurons = layer["neurons"]
                spikes = np.zeros(n, dtype=bool)

                # Compute input current
                currents = np.zeros(n)

                # External input for first layer
                if layer_name == input_layer:
                    currents[:len(input_current)] = input_current[:n]

                # Synaptic input from connections
                for conn in self._connections:
                    if conn["target"] != layer_name:
                        continue

                    src_name = conn["source"]
                    key = f"{src_name}_{layer_name}"

                    # Get delayed spikes
                    for synapse in conn["synapses"]:
                        if synapse.delay <= len(delay_buffer[key].get(synapse.delay, [])):
                            delayed = delay_buffer[key][synapse.delay]
                            if delayed and synapse.pre_id < len(delayed):
                                pre_spike = delayed[synapse.pre_id]
                                currents[synapse.post_id] += synapse.get_current(
                                    pre_spike, self.timestep
                                )

                # Update neurons
                for i, neuron in enumerate(neurons):
                    spikes[i] = neuron.step(currents[i], self.timestep)

                layer_spikes[layer_name] = spikes

                # Record
                if record:
                    self._spike_history[layer_name].append(spikes.copy())

            # Update delay buffers
            for conn in self._connections:
                key = f"{conn['source']}_{conn['target']}"
                src_spikes = layer_spikes[conn["source"]]

                # Shift buffer
                for d in sorted(delay_buffer[key].keys(), reverse=True):
                    if d > 0:
                        delay_buffer[key][d] = delay_buffer[key].get(d - 1, [])
                delay_buffer[key][0] = src_spikes.tolist()

            # STDP updates
            for conn in self._connections:
                src_spikes = layer_spikes[conn["source"]]
                tgt_spikes = layer_spikes[conn["target"]]

                for synapse in conn["synapses"]:
                    if synapse.synapse_type == SynapseType.STDP:
                        synapse.apply_stdp(
                            src_spikes[synapse.pre_id],
                            tgt_spikes[synapse.post_id],
                            self.timestep,
                        )

            # Record output
            output_spikes[:, t] = layer_spikes[output_layer]

        return output_spikes

    def get_spike_history(self, layer: str) -> np.ndarray:
        """Get recorded spike history for a layer."""
        if layer not in self._spike_history:
            return np.array([])
        return np.array(self._spike_history[layer]).T

    def get_weight_matrix(self, source: str, target: str) -> np.ndarray:
        """Get weight matrix between layers."""
        for conn in self._connections:
            if conn["source"] == source and conn["target"] == target:
                src_n = self._layers[source]["n_neurons"]
                tgt_n = self._layers[target]["n_neurons"]
                matrix = np.zeros((src_n, tgt_n))

                for synapse in conn["synapses"]:
                    matrix[synapse.pre_id, synapse.post_id] = synapse.weight

                return matrix

        return np.array([])

    def summary(self) -> str:
        """Get network summary."""
        lines = ["Spiking Neural Network Summary", "=" * 40]

        total_neurons = 0
        total_synapses = 0

        for name, layer in self._layers.items():
            lines.append(f"Layer '{name}': {layer['n_neurons']} {layer['type'].value} neurons")
            total_neurons += layer["n_neurons"]

        for conn in self._connections:
            n_syn = len(conn["synapses"])
            lines.append(f"Connection {conn['source']} -> {conn['target']}: {n_syn} synapses")
            total_synapses += n_syn

        lines.append("-" * 40)
        lines.append(f"Total neurons: {total_neurons}")
        lines.append(f"Total synapses: {total_synapses}")

        return "\n".join(lines)
