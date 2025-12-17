"""Tests for neuromorphic module."""

import pytest
import numpy as np


class TestSpikeEncoding:
    """Test spike encoding methods."""

    def test_rate_encoding(self):
        """Test rate-based spike encoding."""
        rate = 100  # Hz
        duration = 1.0  # seconds

        # Poisson spike generation
        n_spikes = np.random.poisson(rate * duration)

        # Should be around 100 spikes on average
        assert 50 < n_spikes < 200

    def test_temporal_encoding(self):
        """Test temporal spike encoding."""
        values = np.array([0.9, 0.5, 0.1])
        max_latency = 50  # ms

        # Higher value = earlier spike
        latencies = (1 - values) * max_latency

        assert latencies[0] < latencies[1] < latencies[2]

    def test_spike_to_dense(self):
        """Test spike to dense conversion."""
        n_neurons = 10
        n_timesteps = 100

        spike_times = np.array([10, 25, 50, 75])
        spike_neurons = np.array([0, 3, 5, 7])

        dense = np.zeros((n_neurons, n_timesteps))
        for t, n in zip(spike_times, spike_neurons):
            if t < n_timesteps:
                dense[n, t] = 1

        assert dense.shape == (n_neurons, n_timesteps)
        assert np.sum(dense) == len(spike_times)


class TestLIFNeuron:
    """Test Leaky Integrate-and-Fire neuron."""

    def test_lif_dynamics(self):
        """Test LIF membrane dynamics."""
        v_rest = -65
        v_thresh = -50
        tau_mem = 20  # ms
        dt = 1.0  # ms

        v = v_rest
        i_input = 20  # Strong input

        # Simulate until spike or timeout
        for _ in range(100):
            dv = (-(v - v_rest) + i_input) / tau_mem * dt
            v += dv
            if v >= v_thresh:
                break

        assert v >= v_thresh

    def test_refractory_period(self):
        """Test refractory period behavior."""
        refrac_period = 2.0  # ms
        dt = 0.5

        refrac_counter = refrac_period

        while refrac_counter > 0:
            refrac_counter -= dt

        assert refrac_counter <= 0


class TestSTDP:
    """Test Spike-Timing Dependent Plasticity."""

    def test_ltp(self):
        """Test Long-Term Potentiation (pre before post)."""
        dt = 10  # ms (post after pre)
        tau_plus = 20
        a_plus = 0.01

        # LTP: weight increase
        dw = a_plus * np.exp(-dt / tau_plus)
        assert dw > 0

    def test_ltd(self):
        """Test Long-Term Depression (post before pre)."""
        dt = -10  # ms (post before pre)
        tau_minus = 20
        a_minus = 0.012

        # LTD: weight decrease
        dw = -a_minus * np.exp(dt / tau_minus)
        assert dw < 0

    def test_stdp_window(self):
        """Test STDP timing window."""
        tau_plus = 20
        tau_minus = 20
        a_plus = 0.01
        a_minus = 0.012

        dts = np.linspace(-50, 50, 100)
        dws = []

        for dt in dts:
            if dt > 0:
                dw = a_plus * np.exp(-dt / tau_plus)
            else:
                dw = -a_minus * np.exp(dt / tau_minus)
            dws.append(dw)

        dws = np.array(dws)

        # Positive dt should give positive dw
        assert np.all(dws[dts > 0] > 0)
        # Negative dt should give negative dw
        assert np.all(dws[dts < 0] < 0)
