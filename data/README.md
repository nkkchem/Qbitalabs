# QBitaLabs Test Data Catalog

> **Authored by QbitaLab** - Autonomous AI Agent for QBitaLabs Platform Development

## Directory Structure

```
data/
├── raw/                    # Raw, unprocessed data
├── processed/              # Cleaned and featurized data
├── models/                 # Trained model checkpoints
├── sample/                 # Quick-start sample data
│   ├── molecules/          # Sample molecular data
│   ├── patients/           # Synthetic patient profiles
│   └── biomarkers/         # Sample biomarker panels
├── benchmarks/             # Benchmark datasets
│   ├── quantum/            # Quantum simulation benchmarks
│   └── swarm/              # SWARM convergence tests
└── tests/                  # Test fixtures
```

## Quick Start Data

### Molecules (`data/sample/molecules/`)

| File | Description | Size | Format |
|------|-------------|------|--------|
| `egfr_ligands.sdf` | EGFR inhibitor candidates | 100 molecules | SDF |
| `kinase_library.csv` | Kinase inhibitor SMILES | 1,000 molecules | CSV |
| `drug_candidates.json` | Drug-like molecules | 500 molecules | JSON |

### Patients (`data/sample/patients/`)

| File | Description | Size | Format |
|------|-------------|------|--------|
| `synthetic_cohort.json` | Synthetic patient profiles | 1,000 patients | JSON |
| `diabetes_cohort.csv` | Diabetes risk cohort | 500 patients | CSV |
| `longevity_cohort.json` | Longevity study profiles | 200 patients | JSON |

### Biomarkers (`data/sample/biomarkers/`)

| File | Description | Markers | Format |
|------|-------------|---------|--------|
| `diabetes_panel.csv` | Diabetes biomarker panel | 20 markers | CSV |
| `cardiac_panel.csv` | Cardiovascular markers | 15 markers | CSV |
| `aging_panel.json` | Biological age markers | 50 markers | JSON |

## Benchmark Data

### Quantum Benchmarks (`data/benchmarks/quantum/`)

| File | Molecule | Atoms | Qubits | Expected Energy |
|------|----------|-------|--------|-----------------|
| `h2_hamiltonian.json` | H₂ | 2 | 4 | -1.137 Ha |
| `h2o_hamiltonian.json` | H₂O | 3 | 14 | -76.4 Ha |
| `lih_hamiltonian.json` | LiH | 2 | 12 | -8.07 Ha |
| `caffeine_hamiltonian.json` | Caffeine | 24 | 100+ | TBD |

### SWARM Benchmarks (`data/benchmarks/swarm/`)

| File | Pattern | Agents | Expected Convergence |
|------|---------|--------|---------------------|
| `protein_swarm_test.json` | Protein Swarm | 100 | <50 iterations |
| `stigmergy_test.json` | Stigmergy | 50 | <100 iterations |
| `ant_colony_test.json` | Ant Colony | 100 | <200 iterations |

## Data Sources

### Molecular/Drug Discovery

| Source | URL | License |
|--------|-----|---------|
| ChEMBL | https://www.ebi.ac.uk/chembl/ | CC BY-SA 3.0 |
| PubChem | https://pubchem.ncbi.nlm.nih.gov/ | Public Domain |
| ZINC | https://zinc.docking.org/ | Free for research |
| DrugBank | https://go.drugbank.com/ | CC BY-NC 4.0 |

### Genomics/Biomarkers

| Source | URL | License |
|--------|-----|---------|
| 1000 Genomes | https://www.internationalgenome.org/ | Public Domain |
| GTEx | https://gtexportal.org/ | dbGaP |
| KEGG | https://www.genome.jp/kegg/ | Academic license |

### Clinical Data (Synthetic)

| Source | URL | License |
|--------|-----|---------|
| Synthea | https://synthetichealth.github.io/synthea/ | Apache 2.0 |

## Usage Examples

### Loading Molecular Data

```python
from qbitalabs.data import MolecularDataLoader

loader = MolecularDataLoader()
molecules = loader.load("data/sample/molecules/egfr_ligands.sdf")
print(f"Loaded {len(molecules)} molecules")
```

### Loading Patient Profiles

```python
from qbitalabs.data import ClinicalDataLoader

loader = ClinicalDataLoader()
patients = loader.load("data/sample/patients/synthetic_cohort.json")
print(f"Loaded {len(patients)} patient profiles")
```

### Loading Quantum Benchmarks

```python
import json

with open("data/benchmarks/quantum/h2_hamiltonian.json") as f:
    benchmark = json.load(f)

print(f"Molecule: {benchmark['name']}")
print(f"Expected energy: {benchmark['expected_energy']} Ha")
```

## Data Generation Scripts

```bash
# Generate synthetic patient cohort
python scripts/generate_synthetic_patients.py --size 1000

# Generate molecular benchmark data
python scripts/generate_molecular_benchmarks.py

# Validate data integrity
python scripts/validate_data.py
```

## Data Quality Standards

All data in this repository meets:
- ✅ Schema validation
- ✅ Missing value handling
- ✅ Duplicate detection
- ✅ Range validation
- ✅ Format consistency

---

*QBitaLabs, Inc. — Quality data for quantum-accurate healthcare*
