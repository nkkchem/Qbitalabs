"""
QBitaLabs Quickstart Example

Demonstrates basic usage of the QBita platform.
"""

import asyncio
import numpy as np

from qbitalabs.core.config import QBitaLabsConfig
from qbitalabs.core.types import AgentRole


async def main():
    """Run quickstart examples."""
    print("=" * 60)
    print("QBitaLabs - Quantum-Bio Swarm Intelligence Platform")
    print("=" * 60)

    # 1. Configuration
    print("\n1. Loading Configuration...")
    config = QBitaLabsConfig()
    print(f"   Environment: {config.environment}")
    print(f"   Log Level: {config.log_level}")

    # 2. Agent Roles
    print("\n2. Available Agent Roles:")
    for role in AgentRole:
        print(f"   - {role.value}")

    # 3. Quantum Simulation (simplified)
    print("\n3. Quantum Circuit Simulation:")
    print("   Creating Bell state circuit...")

    # Simulate Bell state
    zero_state = np.array([1, 0, 0, 0])
    H_I = np.kron(
        np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        np.eye(2)
    )
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    bell_state = CNOT @ H_I @ zero_state
    print(f"   Bell state amplitudes: {bell_state}")
    print(f"   |00⟩ probability: {abs(bell_state[0])**2:.3f}")
    print(f"   |11⟩ probability: {abs(bell_state[3])**2:.3f}")

    # 4. Digital Twin Preview
    print("\n4. Digital Twin Preview:")
    patient_data = {
        "age": 55,
        "conditions": ["hypertension", "prediabetes"],
        "biomarkers": {"glucose": 110, "hba1c": 6.0, "crp": 2.5}
    }
    print(f"   Patient age: {patient_data['age']}")
    print(f"   Conditions: {', '.join(patient_data['conditions'])}")
    print(f"   Glucose: {patient_data['biomarkers']['glucose']} mg/dL")

    # Risk calculation
    cv_risk = 0.1 + patient_data['age'] * 0.005
    if "hypertension" in patient_data['conditions']:
        cv_risk += 0.15
    if patient_data['biomarkers']['glucose'] > 100:
        cv_risk += 0.1

    print(f"   Estimated CV Risk: {cv_risk:.1%}")

    # 5. Neuromorphic Spike Train
    print("\n5. Neuromorphic Computing Preview:")
    print("   Generating spike train...")

    rate = 50  # Hz
    duration = 0.1  # seconds
    n_spikes = np.random.poisson(rate * duration)
    spike_times = np.sort(np.random.uniform(0, duration * 1000, n_spikes))

    print(f"   Firing rate: {rate} Hz")
    print(f"   Generated {n_spikes} spikes")
    print(f"   Spike times (ms): {spike_times[:5]}...")

    print("\n" + "=" * 60)
    print("QBitaLabs quickstart complete!")
    print("Visit https://qbitalabs.com for more information.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
