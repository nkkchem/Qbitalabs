# Getting Started with QBitaLabs

This guide will help you install QBitaLabs and run your first quantum-bio simulation in under 5 minutes.

## Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- (Optional) CUDA for GPU acceleration
- (Optional) Quantum hardware API keys

## Installation

### Option 1: pip (Recommended)

```bash
# Install from PyPI
pip install qbitalabs

# Install with all optional dependencies
pip install qbitalabs[all]
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/qbitalabs/qbitalabs.git
cd qbitalabs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Option 3: Docker

```bash
# Pull the official image
docker pull qbitalabs/qbita-fabric:latest

# Run with docker-compose
docker-compose up -d
```

## Quick Start

### 1. Your First Molecular Simulation

```python
from qbitalabs.quantum import MolecularHamiltonian, VQESolver
from qbitalabs.quantum.backends import SimulatorBackend

# Create a simple molecule (H2)
molecule = MolecularHamiltonian.from_formula("H2", bond_length=0.74)

# Build the Hamiltonian
hamiltonian = molecule.build_hamiltonian()
print(f"Hamiltonian terms: {len(hamiltonian.terms)}")

# Solve with VQE
backend = SimulatorBackend(num_qubits=4)
solver = VQESolver(backend=backend)
result = solver.solve(hamiltonian)

print(f"Ground state energy: {result.energy:.6f} Hartree")
print(f"Optimization converged: {result.converged}")
```

### 2. Your First SWARM Analysis

```python
import asyncio
from qbitalabs.swarm import SwarmFabric
from qbitalabs.swarm.agents import MolecularAgent, PathwayAgent

async def run_swarm_analysis():
    # Create the swarm fabric
    fabric = SwarmFabric()

    # Add specialized agents
    mol_agent = MolecularAgent(
        agent_id="mol-1",
        specialization="drug_binding"
    )
    pathway_agent = PathwayAgent(
        agent_id="path-1",
        specialization="metabolic"
    )

    fabric.add_agents([mol_agent, pathway_agent])

    # Define the task
    task = {
        "type": "drug_analysis",
        "compound": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "target": "COX-2",
    }

    # Execute with swarm coordination
    results = await fabric.execute(task)
    return results

# Run the analysis
results = asyncio.run(run_swarm_analysis())
print(f"Analysis complete: {len(results)} findings")
```

### 3. Your First Digital Twin

```python
from qbitalabs.digital_twin import DigitalTwinEngine, PatientProfile

# Create a patient profile
profile = PatientProfile(
    patient_id="demo-001",
    age=45,
    sex="male",
    genomics={
        "APOE": "e3/e4",  # Alzheimer's risk variant
        "CYP2D6": "*1/*1",  # Normal drug metabolism
    },
    biomarkers={
        "glucose_fasting": 95,  # mg/dL
        "hba1c": 5.4,  # %
        "ldl_cholesterol": 130,  # mg/dL
    }
)

# Create the digital twin
engine = DigitalTwinEngine()
twin = engine.create_twin(profile)

# Simulate disease progression
trajectory = twin.simulate_progression(
    years=10,
    conditions=["type2_diabetes", "cardiovascular"]
)

print(f"10-year diabetes risk: {trajectory['diabetes_risk']:.1%}")
print(f"10-year CVD risk: {trajectory['cvd_risk']:.1%}")
```

## Configuration

Create a configuration file at `~/.qbitalabs/config.yaml`:

```yaml
# QBitaLabs Configuration
project:
  name: my-research-project
  version: 0.1.0

quantum:
  default_backend: simulator
  backends:
    simulator:
      num_qubits: 20
    ibm:
      api_key: ${IBM_QUANTUM_API_KEY}
    ionq:
      api_key: ${IONQ_API_KEY}

swarm:
  max_agents: 100
  coordination_pattern: hierarchical
  message_bus:
    type: memory  # or 'redis' for distributed

digital_twin:
  default_timestep: 1.0  # hours
  models:
    - metabolism
    - cardiovascular
    - immune

api:
  host: 0.0.0.0
  port: 8000
  debug: false
```

## Example Notebooks

Explore our Jupyter notebooks for detailed tutorials:

| Notebook | Description |
|----------|-------------|
| [01_molecular_simulation.ipynb](../notebooks/01_molecular_simulation.ipynb) | Quantum molecular simulation basics |
| [02_drug_discovery.ipynb](../notebooks/02_drug_discovery.ipynb) | End-to-end drug discovery pipeline |
| [03_digital_twin_demo.ipynb](../notebooks/03_digital_twin_demo.ipynb) | Patient digital twin creation |
| [04_swarm_optimization.ipynb](../notebooks/04_swarm_optimization.ipynb) | SWARM-based optimization |
| [05_aging_analysis.ipynb](../notebooks/05_aging_analysis.ipynb) | Biological age analysis |

## API Usage

Start the API server:

```bash
# Development mode
uvicorn qbitalabs.api:app --reload

# Production mode
uvicorn qbitalabs.api:app --host 0.0.0.0 --port 8000 --workers 4
```

Make API requests:

```bash
# Health check
curl http://localhost:8000/health

# Predict drug binding
curl -X POST http://localhost:8000/predict/binding \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": "COX-2"}'

# Run quantum simulation
curl -X POST http://localhost:8000/simulate/molecule \
  -H "Content-Type: application/json" \
  -d '{"smiles": "O", "method": "vqe"}'
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'qbitalabs'`
```bash
# Solution: Install the package
pip install qbitalabs
```

**Issue**: Quantum backend not available
```bash
# Solution: Install optional dependencies
pip install qbitalabs[quantum]  # For Qiskit, PennyLane
```

**Issue**: Out of memory during simulation
```python
# Solution: Reduce system size or use distributed backend
from qbitalabs.quantum import set_memory_limit
set_memory_limit(8 * 1024**3)  # 8 GB
```

### Getting Help

- **Documentation**: [docs.qbitalabs.com](https://docs.qbitalabs.com)
- **GitHub Issues**: [github.com/qbitalabs/qbitalabs/issues](https://github.com/qbitalabs/qbitalabs/issues)
- **Discord**: [discord.gg/qbitalabs](https://discord.gg/qbitalabs)
- **Email**: hello@qbitalabs.com

## Next Steps

1. Explore the [Architecture](architecture.md) documentation
2. Read the [API Reference](api-reference.md)
3. Check out example [Notebooks](../notebooks/)
4. Join our [Community](contributing.md)

---

*QBitaLabs, Inc. — Swarm intelligence for quantum biology and human health*
