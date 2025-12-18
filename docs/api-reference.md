# API Reference

Complete reference documentation for the QBitaLabs REST API and Python SDK.

## REST API

Base URL: `https://api.qbitalabs.com/v1` (or `http://localhost:8000` for local development)

### Authentication

```bash
# Header-based authentication
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.qbitalabs.com/v1/health
```

### Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/swarm/create` | POST | Create SWARM task |
| `/swarm/{task_id}/status` | GET | Get task status |
| `/quantum/simulate` | POST | Run quantum simulation |
| `/quantum/vqe` | POST | Run VQE calculation |
| `/twin/create` | POST | Create digital twin |
| `/twin/{twin_id}/simulate` | POST | Run twin simulation |
| `/predict/binding` | POST | Drug-target binding prediction |
| `/analyze/pathway` | POST | Pathway analysis |

---

## SWARM Endpoints

### POST `/swarm/create`

Create a new SWARM analysis task.

**Request Body:**
```json
{
  "task_type": "drug_discovery",
  "parameters": {
    "target": "EGFR",
    "compound_library": ["SMILES1", "SMILES2"],
    "optimization_goal": "binding_affinity"
  },
  "agents": {
    "molecular": 5,
    "pathway": 3,
    "literature": 2
  },
  "coordination_pattern": "protein_swarm"
}
```

**Response:**
```json
{
  "task_id": "swarm-abc123",
  "status": "running",
  "agents_deployed": 10,
  "estimated_completion": "2024-01-15T10:30:00Z"
}
```

### GET `/swarm/{task_id}/status`

Get status of a running SWARM task.

**Response:**
```json
{
  "task_id": "swarm-abc123",
  "status": "completed",
  "progress": 100,
  "results": {
    "top_compounds": [...],
    "pathway_insights": [...],
    "literature_support": [...]
  },
  "metrics": {
    "agents_used": 10,
    "iterations": 150,
    "wall_time_seconds": 342
  }
}
```

---

## Quantum Endpoints

### POST `/quantum/simulate`

Run a quantum molecular simulation.

**Request Body:**
```json
{
  "molecule": {
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "name": "aspirin"
  },
  "method": "vqe",
  "backend": "simulator",
  "options": {
    "ansatz": "uccsd",
    "optimizer": "cobyla",
    "max_iterations": 100
  }
}
```

**Response:**
```json
{
  "job_id": "quantum-xyz789",
  "status": "completed",
  "results": {
    "ground_state_energy": -232.456789,
    "energy_unit": "hartree",
    "orbital_energies": [-10.23, -5.67, ...],
    "dipole_moment": [0.12, 0.34, 0.56],
    "convergence": true
  },
  "execution_time_ms": 1234
}
```

### POST `/quantum/vqe`

Run VQE calculation with custom Hamiltonian.

**Request Body:**
```json
{
  "hamiltonian": {
    "terms": [
      {"coefficient": 0.5, "pauli_string": "ZZII"},
      {"coefficient": 0.3, "pauli_string": "XXII"}
    ]
  },
  "num_qubits": 4,
  "ansatz": {
    "type": "hardware_efficient",
    "depth": 3
  },
  "optimizer": "adam",
  "shots": 1000
}
```

---

## Digital Twin Endpoints

### POST `/twin/create`

Create a patient digital twin.

**Request Body:**
```json
{
  "patient_id": "patient-001",
  "profile": {
    "age": 45,
    "sex": "male",
    "height_cm": 175,
    "weight_kg": 80,
    "genomics": {
      "APOE": "e3/e4",
      "CYP2D6": "*1/*1"
    },
    "biomarkers": {
      "glucose_fasting": 95,
      "hba1c": 5.4,
      "ldl_cholesterol": 130
    }
  },
  "models": ["metabolism", "cardiovascular", "immune"]
}
```

**Response:**
```json
{
  "twin_id": "twin-abc123",
  "status": "initialized",
  "models_loaded": ["metabolism", "cardiovascular", "immune"],
  "baseline_state": {
    "biological_age": 47.2,
    "health_score": 78.5,
    "risk_factors": ["elevated_ldl", "apoe4_carrier"]
  }
}
```

### POST `/twin/{twin_id}/simulate`

Run simulation on a digital twin.

**Request Body:**
```json
{
  "simulation_type": "intervention",
  "intervention": {
    "type": "drug",
    "name": "metformin",
    "dose_mg": 500,
    "frequency": "twice_daily"
  },
  "duration_days": 90,
  "timestep_hours": 24
}
```

**Response:**
```json
{
  "simulation_id": "sim-xyz789",
  "twin_id": "twin-abc123",
  "results": {
    "trajectory": [
      {"day": 0, "glucose": 95, "hba1c": 5.4},
      {"day": 30, "glucose": 92, "hba1c": 5.3},
      {"day": 90, "glucose": 88, "hba1c": 5.1}
    ],
    "predicted_outcomes": {
      "glucose_reduction": 7.4,
      "diabetes_risk_reduction": 0.23
    },
    "side_effects_risk": {
      "gi_upset": 0.15,
      "lactic_acidosis": 0.001
    }
  }
}
```

---

## Prediction Endpoints

### POST `/predict/binding`

Predict drug-target binding affinity.

**Request Body:**
```json
{
  "compound": {
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
  },
  "target": {
    "gene_name": "PTGS2",
    "uniprot_id": "P35354"
  },
  "method": "ensemble"
}
```

**Response:**
```json
{
  "prediction_id": "pred-123",
  "binding_affinity": {
    "pIC50": 6.8,
    "Ki_nM": 158.5,
    "confidence": 0.92
  },
  "interaction_details": {
    "binding_site": "active_site",
    "key_residues": ["Arg120", "Tyr355", "Glu524"],
    "interaction_types": ["hydrogen_bond", "hydrophobic"]
  },
  "selectivity": {
    "vs_PTGS1": 0.15
  }
}
```

---

## Python SDK Reference

### Core Classes

#### `qbitalabs.core.QBitaConfig`

```python
from qbitalabs.core import QBitaConfig

# Load configuration
config = QBitaConfig.from_yaml("config.yaml")

# Access settings
print(config.quantum.default_backend)
print(config.swarm.max_agents)
```

### SWARM Module

#### `qbitalabs.swarm.SwarmFabric`

```python
from qbitalabs.swarm import SwarmFabric

class SwarmFabric:
    """Main orchestrator for SWARM agent systems."""

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        coordination_pattern: str = "hierarchical"
    ):
        """
        Initialize the SWARM fabric.

        Args:
            config: SWARM configuration object
            coordination_pattern: One of "hierarchical", "stigmergy",
                                 "protein_swarm", "ant_colony", "particle_swarm"
        """

    def add_agents(self, agents: List[BaseAgent]) -> None:
        """Add agents to the swarm."""

    async def execute(
        self,
        task: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> SwarmResult:
        """
        Execute a task using the swarm.

        Args:
            task: Task specification dictionary
            timeout: Maximum execution time in seconds

        Returns:
            SwarmResult with findings and metrics
        """
```

### Quantum Module

#### `qbitalabs.quantum.VQESolver`

```python
from qbitalabs.quantum import VQESolver

class VQESolver:
    """Variational Quantum Eigensolver for molecular simulation."""

    def __init__(
        self,
        backend: BaseQuantumBackend,
        ansatz: str = "uccsd",
        optimizer: str = "cobyla"
    ):
        """
        Initialize VQE solver.

        Args:
            backend: Quantum backend to use
            ansatz: Ansatz type ("uccsd", "hardware_efficient")
            optimizer: Classical optimizer ("cobyla", "adam", "l-bfgs-b")
        """

    def solve(
        self,
        hamiltonian: QubitOperator,
        initial_params: Optional[np.ndarray] = None
    ) -> VQEResult:
        """
        Solve for ground state energy.

        Args:
            hamiltonian: Molecular Hamiltonian in qubit representation
            initial_params: Initial variational parameters

        Returns:
            VQEResult with energy, parameters, and convergence info
        """
```

### Digital Twin Module

#### `qbitalabs.digital_twin.DigitalTwinEngine`

```python
from qbitalabs.digital_twin import DigitalTwinEngine, PatientProfile

class DigitalTwinEngine:
    """Engine for creating and simulating patient digital twins."""

    def create_twin(self, profile: PatientProfile) -> PatientTwin:
        """
        Create a digital twin from patient profile.

        Args:
            profile: Patient demographic and omics data

        Returns:
            PatientTwin instance ready for simulation
        """

class PatientTwin:
    """Represents a patient's digital twin."""

    def simulate_progression(
        self,
        years: float,
        conditions: List[str],
        interventions: Optional[List[Intervention]] = None
    ) -> TrajectoryResult:
        """
        Simulate disease progression over time.

        Args:
            years: Simulation duration in years
            conditions: Conditions to model
            interventions: Optional interventions to apply

        Returns:
            TrajectoryResult with time series predictions
        """
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `400` | Bad Request - Invalid input parameters |
| `401` | Unauthorized - Invalid or missing API key |
| `403` | Forbidden - Insufficient permissions |
| `404` | Not Found - Resource does not exist |
| `422` | Unprocessable Entity - Validation error |
| `429` | Too Many Requests - Rate limit exceeded |
| `500` | Internal Server Error |
| `503` | Service Unavailable - Quantum backend offline |

## Rate Limits

| Tier | Requests/min | Quantum jobs/day |
|------|--------------|------------------|
| Free | 10 | 5 |
| Developer | 100 | 50 |
| Professional | 1000 | 500 |
| Enterprise | Unlimited | Unlimited |

---

*For more examples, see our [Getting Started Guide](getting-started.md) and [Notebooks](../notebooks/).*
