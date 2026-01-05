#!/usr/bin/env python3
"""
QbitaLab: Data Download Pipeline

Downloads and processes molecular and health datasets:
- ChEMBL: Drug-target interactions
- PDBbind: Binding affinity data
- ZINC250K: Drug-like compounds
- Synthea: Synthetic patient data
- KEGG: Biological pathways

Usage:
    python download_all.py --datasets chembl,zinc,synthea
    python download_all.py --all
"""

import os
import sys
import json
import gzip
import shutil
import hashlib
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import subprocess

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "mvp" / "data"


class DatasetDownloader:
    """Base class for dataset downloaders."""

    name: str = "base"
    description: str = ""
    license: str = ""
    size_estimate: str = ""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.data_dir / "download_log.json"

    def download(self) -> bool:
        """Download the dataset. Returns True on success."""
        raise NotImplementedError

    def verify(self) -> bool:
        """Verify the downloaded data."""
        raise NotImplementedError

    def process(self) -> bool:
        """Process raw data into usable format."""
        raise NotImplementedError

    def log_download(self, success: bool, details: Dict):
        """Log download attempt."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "dataset": self.name,
            **details,
        }

        logs = []
        if self.log_file.exists():
            with open(self.log_file) as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)

    def download_file(self, url: str, dest: Path, chunk_size: int = 8192) -> bool:
        """Download a file with progress."""
        try:
            print(f"  Downloading {url}...")
            with urllib.request.urlopen(url, timeout=300) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            print(f"\r  Progress: {progress:.1f}%", end="", flush=True)

                print()  # New line after progress
            return True
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            return False


class ChEMBLDownloader(DatasetDownloader):
    """Download ChEMBL drug-target interaction data."""

    name = "chembl"
    description = "Drug-target interactions database"
    license = "CC BY-SA 4.0"
    size_estimate = "~500MB compressed"

    # ChEMBL SQLite database URL (smaller subset for MVP)
    CHEMBL_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_sqlite.tar.gz"

    # Alternative: Use ChEMBL web services for smaller downloads
    CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

    def download(self) -> bool:
        """Download ChEMBL data via API for MVP (smaller download)."""
        print(f"\n[ChEMBL] Downloading drug-target interactions...")

        try:
            # Download sample data via API for MVP
            # Full database is ~2GB - we'll use API for key data
            targets = self._download_targets()
            compounds = self._download_compounds()
            activities = self._download_activities()

            success = targets and compounds and activities
            self.log_download(success, {
                "targets": len(targets) if targets else 0,
                "compounds": len(compounds) if compounds else 0,
                "activities": len(activities) if activities else 0,
            })
            return success

        except Exception as e:
            print(f"  Error: {e}")
            self.log_download(False, {"error": str(e)})
            return False

    def _download_targets(self, limit: int = 1000) -> Optional[List]:
        """Download target data."""
        url = f"{self.CHEMBL_API}/target.json?limit={limit}"
        dest = self.data_dir / "targets.json"

        if self.download_file(url, dest):
            with open(dest) as f:
                data = json.load(f)
            return data.get("targets", [])
        return None

    def _download_compounds(self, limit: int = 10000) -> Optional[List]:
        """Download compound data."""
        url = f"{self.CHEMBL_API}/molecule.json?limit={limit}"
        dest = self.data_dir / "molecules.json"

        if self.download_file(url, dest):
            with open(dest) as f:
                data = json.load(f)
            return data.get("molecules", [])
        return None

    def _download_activities(self, limit: int = 50000) -> Optional[List]:
        """Download activity data."""
        url = f"{self.CHEMBL_API}/activity.json?limit={limit}"
        dest = self.data_dir / "activities.json"

        if self.download_file(url, dest):
            with open(dest) as f:
                data = json.load(f)
            return data.get("activities", [])
        return None

    def verify(self) -> bool:
        """Verify downloaded data."""
        required_files = ["targets.json", "molecules.json", "activities.json"]
        return all((self.data_dir / f).exists() for f in required_files)

    def process(self) -> bool:
        """Process raw data into training format."""
        print("  Processing ChEMBL data...")

        try:
            # Load raw data
            with open(self.data_dir / "molecules.json") as f:
                molecules = json.load(f).get("molecules", [])
            with open(self.data_dir / "activities.json") as f:
                activities = json.load(f).get("activities", [])

            # Create training dataset
            training_data = []
            for activity in activities:
                if activity.get("standard_value") and activity.get("canonical_smiles"):
                    training_data.append({
                        "smiles": activity.get("canonical_smiles"),
                        "target_chembl_id": activity.get("target_chembl_id"),
                        "activity_type": activity.get("standard_type"),
                        "activity_value": float(activity.get("standard_value", 0)),
                        "activity_units": activity.get("standard_units"),
                    })

            # Save processed data
            with open(self.data_dir / "training_data.json", "w") as f:
                json.dump(training_data, f)

            print(f"  Processed {len(training_data)} activity records")
            return True

        except Exception as e:
            print(f"  Processing error: {e}")
            return False


class ZINC250KDownloader(DatasetDownloader):
    """Download ZINC250K drug-like molecules."""

    name = "zinc250k"
    description = "Drug-like compound dataset"
    license = "Open"
    size_estimate = "~50MB"

    # ZINC250K from MoleculeNet
    ZINC_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/zinc250k.csv"

    def download(self) -> bool:
        """Download ZINC250K dataset."""
        print(f"\n[ZINC250K] Downloading drug-like molecules...")

        dest = self.data_dir / "zinc250k.csv"
        success = self.download_file(self.ZINC_URL, dest)
        self.log_download(success, {"file": str(dest)})
        return success

    def verify(self) -> bool:
        """Verify downloaded data."""
        return (self.data_dir / "zinc250k.csv").exists()

    def process(self) -> bool:
        """Process ZINC data."""
        print("  Processing ZINC250K data...")

        try:
            # Read and process CSV
            molecules = []
            with open(self.data_dir / "zinc250k.csv") as f:
                header = f.readline().strip().split(",")
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        molecules.append({
                            "smiles": parts[0],
                            "logp": float(parts[1]) if len(parts) > 1 and parts[1] else None,
                        })

            with open(self.data_dir / "molecules_processed.json", "w") as f:
                json.dump(molecules, f)

            print(f"  Processed {len(molecules)} molecules")
            return True

        except Exception as e:
            print(f"  Processing error: {e}")
            return False


class SyntheaDownloader(DatasetDownloader):
    """Generate synthetic patient data using Synthea."""

    name = "synthea"
    description = "Synthetic patient data"
    license = "Apache 2.0"
    size_estimate = "Configurable (100MB-10GB)"

    def download(self) -> bool:
        """Generate synthetic patient data."""
        print(f"\n[Synthea] Generating synthetic patient data...")

        # For MVP, we'll create synthetic data directly
        # Full Synthea requires Java and is ~1GB
        return self._generate_synthetic_patients(1000)

    def _generate_synthetic_patients(self, n_patients: int) -> bool:
        """Generate synthetic patient records."""
        import random

        patients = []
        conditions = [
            "diabetes", "hypertension", "heart_disease",
            "copd", "asthma", "cancer", "obesity"
        ]
        medications = [
            "metformin", "lisinopril", "atorvastatin",
            "omeprazole", "amlodipine", "albuterol", "insulin"
        ]

        for i in range(n_patients):
            age = random.randint(18, 90)
            patient = {
                "patient_id": f"P{i:06d}",
                "age": age,
                "sex": random.choice(["M", "F"]),
                "height_cm": random.gauss(170, 10),
                "weight_kg": random.gauss(75, 15),
                "conditions": random.sample(conditions, random.randint(0, 3)),
                "medications": random.sample(medications, random.randint(0, 4)),
                "lab_results": {
                    "glucose": random.gauss(100, 25),
                    "hemoglobin": random.gauss(14, 2),
                    "creatinine": random.gauss(1.0, 0.3),
                    "cholesterol_total": random.gauss(200, 40),
                    "ldl": random.gauss(100, 30),
                    "hdl": random.gauss(50, 15),
                },
                "vitals": {
                    "blood_pressure_systolic": random.gauss(120, 15),
                    "blood_pressure_diastolic": random.gauss(80, 10),
                    "heart_rate": random.gauss(72, 12),
                },
            }
            patients.append(patient)

        with open(self.data_dir / "patients.json", "w") as f:
            json.dump(patients, f, indent=2)

        self.log_download(True, {"n_patients": n_patients})
        print(f"  Generated {n_patients} synthetic patients")
        return True

    def verify(self) -> bool:
        """Verify generated data."""
        return (self.data_dir / "patients.json").exists()

    def process(self) -> bool:
        """Process patient data for training."""
        print("  Processing Synthea data...")

        try:
            with open(self.data_dir / "patients.json") as f:
                patients = json.load(f)

            # Create features for risk prediction
            training_data = []
            for patient in patients:
                features = {
                    "age": patient["age"],
                    "sex": 1 if patient["sex"] == "M" else 0,
                    "bmi": patient["weight_kg"] / ((patient["height_cm"] / 100) ** 2),
                    "glucose": patient["lab_results"]["glucose"],
                    "cholesterol": patient["lab_results"]["cholesterol_total"],
                    "ldl": patient["lab_results"]["ldl"],
                    "hdl": patient["lab_results"]["hdl"],
                    "systolic_bp": patient["vitals"]["blood_pressure_systolic"],
                    "has_diabetes": 1 if "diabetes" in patient["conditions"] else 0,
                    "has_hypertension": 1 if "hypertension" in patient["conditions"] else 0,
                }
                training_data.append(features)

            with open(self.data_dir / "training_features.json", "w") as f:
                json.dump(training_data, f)

            print(f"  Processed {len(training_data)} patient records")
            return True

        except Exception as e:
            print(f"  Processing error: {e}")
            return False


class KEGGDownloader(DatasetDownloader):
    """Download KEGG pathway data."""

    name = "kegg"
    description = "Biological pathways database"
    license = "Academic (non-commercial)"
    size_estimate = "~100MB"

    KEGG_API = "https://rest.kegg.jp"

    def download(self) -> bool:
        """Download KEGG pathway data."""
        print(f"\n[KEGG] Downloading biological pathways...")

        try:
            # Get pathway list
            pathways = self._download_pathway_list()
            if not pathways:
                return False

            # Download pathway details (limited for MVP)
            pathway_details = []
            for pathway_id in pathways[:100]:  # Limit for MVP
                details = self._download_pathway(pathway_id)
                if details:
                    pathway_details.append(details)

            with open(self.data_dir / "pathways.json", "w") as f:
                json.dump(pathway_details, f, indent=2)

            self.log_download(True, {"n_pathways": len(pathway_details)})
            print(f"  Downloaded {len(pathway_details)} pathways")
            return True

        except Exception as e:
            print(f"  Error: {e}")
            self.log_download(False, {"error": str(e)})
            return False

    def _download_pathway_list(self) -> List[str]:
        """Get list of human pathways."""
        url = f"{self.KEGG_API}/list/pathway/hsa"
        dest = self.data_dir / "pathway_list.txt"

        if self.download_file(url, dest):
            pathways = []
            with open(dest) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        pathways.append(parts[0])
            return pathways
        return []

    def _download_pathway(self, pathway_id: str) -> Optional[Dict]:
        """Download pathway details."""
        try:
            url = f"{self.KEGG_API}/get/{pathway_id}"
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8")
                return {"id": pathway_id, "content": content}
        except Exception:
            return None

    def verify(self) -> bool:
        """Verify downloaded data."""
        return (self.data_dir / "pathways.json").exists()

    def process(self) -> bool:
        """Process pathway data."""
        print("  Processing KEGG data...")
        # Pathway data is already structured
        return True


class MoleculeNetDownloader(DatasetDownloader):
    """Download MoleculeNet benchmark datasets."""

    name = "moleculenet"
    description = "Molecular property prediction benchmarks"
    license = "MIT"
    size_estimate = "~200MB"

    # MoleculeNet datasets
    DATASETS = {
        "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "bbbp": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    }

    def download(self) -> bool:
        """Download MoleculeNet datasets."""
        print(f"\n[MoleculeNet] Downloading benchmark datasets...")

        success_count = 0
        for name, url in self.DATASETS.items():
            dest = self.data_dir / f"{name}.csv"
            if self.download_file(url, dest):
                success_count += 1

        success = success_count == len(self.DATASETS)
        self.log_download(success, {"datasets_downloaded": success_count})
        return success

    def verify(self) -> bool:
        """Verify downloaded data."""
        return all(
            (self.data_dir / f"{name}.csv").exists()
            for name in self.DATASETS
        )

    def process(self) -> bool:
        """Process MoleculeNet data."""
        print("  Processing MoleculeNet data...")

        try:
            processed = {}
            for name in self.DATASETS:
                filepath = self.data_dir / f"{name}.csv"
                if filepath.exists():
                    data = []
                    with open(filepath) as f:
                        header = f.readline().strip().split(",")
                        for line in f:
                            parts = line.strip().split(",")
                            if len(parts) >= 2:
                                data.append(dict(zip(header, parts)))
                    processed[name] = data
                    print(f"    {name}: {len(data)} samples")

            with open(self.data_dir / "processed_benchmarks.json", "w") as f:
                json.dump(processed, f)

            return True

        except Exception as e:
            print(f"  Processing error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Download QbitaLab datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets: chembl,zinc,synthea,kegg,moleculenet",
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing downloads")
    parser.add_argument("--process-only", action="store_true", help="Only process existing data")

    args = parser.parse_args()

    # Initialize downloaders
    downloaders = {
        "chembl": ChEMBLDownloader(DATA_DIR),
        "zinc": ZINC250KDownloader(DATA_DIR),
        "synthea": SyntheaDownloader(DATA_DIR),
        "kegg": KEGGDownloader(DATA_DIR),
        "moleculenet": MoleculeNetDownloader(DATA_DIR),
    }

    # Determine which datasets to process
    if args.all:
        selected = list(downloaders.keys())
    elif args.datasets:
        selected = [d.strip() for d in args.datasets.split(",")]
    else:
        # Default: minimal set for MVP
        selected = ["zinc", "synthea", "moleculenet"]

    print("=" * 60)
    print("QbitaLab Data Download Pipeline")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Selected datasets: {', '.join(selected)}")
    print()

    results = {}

    for name in selected:
        if name not in downloaders:
            print(f"Unknown dataset: {name}")
            continue

        downloader = downloaders[name]
        print(f"\n{'=' * 40}")
        print(f"Dataset: {downloader.name}")
        print(f"Description: {downloader.description}")
        print(f"License: {downloader.license}")
        print(f"Size: {downloader.size_estimate}")
        print("=" * 40)

        if args.verify_only:
            results[name] = {"verified": downloader.verify()}
        elif args.process_only:
            results[name] = {"processed": downloader.process()}
        else:
            # Full pipeline
            download_ok = downloader.download()
            verify_ok = downloader.verify() if download_ok else False
            process_ok = downloader.process() if verify_ok else False
            results[name] = {
                "download": download_ok,
                "verify": verify_ok,
                "process": process_ok,
            }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if all(result.values()) else "✗"
        print(f"  {status} {name}: {result}")

    # Save summary
    summary_file = DATA_DIR / "download_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
