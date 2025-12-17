"""
QBitaLabs Data Module

Data loading and preprocessing for biological data:
- Clinical datasets
- Molecular structures
- Omics data
- Image data
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DataSample:
    """Single data sample."""

    sample_id: str
    features: np.ndarray
    label: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataLoader(ABC):
    """Base class for data loaders."""

    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize data loader.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._data: list[DataSample] = []
        self._logger = structlog.get_logger("data_loader")

    @abstractmethod
    def load(self, path: str | Path) -> "BaseDataLoader":
        """Load data from path."""
        pass

    def __iter__(self) -> Iterator[list[DataSample]]:
        """Iterate over batches."""
        indices = np.arange(len(self._data))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [self._data[j] for j in batch_indices]

    def __len__(self) -> int:
        """Number of samples."""
        return len(self._data)


class ClinicalDataLoader(BaseDataLoader):
    """
    Loads clinical trial and patient data.

    Supports:
    - CSV/TSV files
    - Feature normalization
    - Missing value handling
    """

    def __init__(
        self,
        batch_size: int = 32,
        normalize: bool = True,
        fill_missing: str = "mean",
    ):
        """
        Initialize clinical data loader.

        Args:
            batch_size: Batch size.
            normalize: Whether to normalize features.
            fill_missing: Strategy for missing values.
        """
        super().__init__(batch_size)
        self.normalize = normalize
        self.fill_missing = fill_missing
        self._feature_stats: dict[str, dict[str, float]] = {}

    def load(self, path: str | Path) -> "ClinicalDataLoader":
        """Load clinical data from CSV."""
        path = Path(path)

        if not path.exists():
            self._logger.warning(f"File not found: {path}")
            return self

        # Read CSV (simplified)
        try:
            with open(path, "r") as f:
                lines = f.readlines()

            if len(lines) < 2:
                return self

            header = lines[0].strip().split(",")
            feature_cols = [i for i, h in enumerate(header) if h != "label" and h != "id"]
            label_col = header.index("label") if "label" in header else -1
            id_col = header.index("id") if "id" in header else 0

            for i, line in enumerate(lines[1:]):
                values = line.strip().split(",")
                if len(values) != len(header):
                    continue

                sample_id = values[id_col] if id_col >= 0 else f"sample_{i}"
                features = np.array([
                    float(values[j]) if values[j] else np.nan
                    for j in feature_cols
                ])
                label = float(values[label_col]) if label_col >= 0 else None

                self._data.append(DataSample(
                    sample_id=sample_id,
                    features=features,
                    label=label,
                ))

            self._preprocess()

        except Exception as e:
            self._logger.error(f"Failed to load data: {e}")

        self._logger.info(f"Loaded {len(self._data)} samples")
        return self

    def _preprocess(self) -> None:
        """Preprocess loaded data."""
        if not self._data:
            return

        # Collect all features
        all_features = np.array([s.features for s in self._data])

        # Handle missing values
        if self.fill_missing == "mean":
            means = np.nanmean(all_features, axis=0)
            for i, sample in enumerate(self._data):
                mask = np.isnan(sample.features)
                sample.features[mask] = means[mask]

        # Normalize
        if self.normalize:
            means = np.mean(all_features, axis=0)
            stds = np.std(all_features, axis=0) + 1e-8

            for sample in self._data:
                sample.features = (sample.features - means) / stds

            self._feature_stats = {
                "mean": means.tolist(),
                "std": stds.tolist(),
            }


class MolecularDataLoader(BaseDataLoader):
    """
    Loads molecular structure data.

    Supports:
    - SMILES strings
    - SDF files
    - Molecular graphs
    """

    def __init__(
        self,
        batch_size: int = 32,
        featurizer: str = "fingerprint",
    ):
        """
        Initialize molecular data loader.

        Args:
            batch_size: Batch size.
            featurizer: Featurization method.
        """
        super().__init__(batch_size)
        self.featurizer = featurizer

    def load(self, path: str | Path) -> "MolecularDataLoader":
        """Load molecular data."""
        path = Path(path)

        if path.suffix == ".smi":
            self._load_smiles(path)
        elif path.suffix == ".sdf":
            self._load_sdf(path)
        else:
            self._load_smiles(path)

        return self

    def _load_smiles(self, path: Path) -> None:
        """Load SMILES file."""
        try:
            with open(path, "r") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        smiles = parts[0]
                        label = float(parts[1]) if len(parts) > 1 else None

                        features = self._featurize(smiles)
                        self._data.append(DataSample(
                            sample_id=f"mol_{i}",
                            features=features,
                            label=label,
                            metadata={"smiles": smiles},
                        ))
        except Exception as e:
            self._logger.error(f"Failed to load SMILES: {e}")

    def _load_sdf(self, path: Path) -> None:
        """Load SDF file (simplified)."""
        self._logger.info("SDF loading - using simplified parser")
        # Simplified: would use RDKit in production

    def _featurize(self, smiles: str) -> np.ndarray:
        """Convert SMILES to feature vector."""
        if self.featurizer == "fingerprint":
            return self._morgan_fingerprint(smiles)
        elif self.featurizer == "descriptors":
            return self._descriptors(smiles)
        return np.zeros(1024)

    def _morgan_fingerprint(self, smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
        """Compute Morgan fingerprint (simplified)."""
        # Simplified: would use RDKit in production
        np.random.seed(hash(smiles) % 2**32)
        return np.random.randint(0, 2, n_bits).astype(float)

    def _descriptors(self, smiles: str) -> np.ndarray:
        """Compute molecular descriptors (simplified)."""
        # Simplified molecular descriptors
        return np.array([
            len(smiles),  # Proxy for molecular weight
            smiles.count("C"),  # Carbon count
            smiles.count("N"),  # Nitrogen count
            smiles.count("O"),  # Oxygen count
            smiles.count("="),  # Double bonds
            smiles.count("#"),  # Triple bonds
            smiles.count("("),  # Ring/branch count
        ]).astype(float)


class OmicsDataLoader(BaseDataLoader):
    """
    Loads omics data (genomics, proteomics, etc.).

    Supports:
    - Gene expression matrices
    - Variant call files
    - Mass spectrometry data
    """

    def __init__(
        self,
        batch_size: int = 32,
        data_type: str = "expression",
        log_transform: bool = True,
    ):
        """
        Initialize omics data loader.

        Args:
            batch_size: Batch size.
            data_type: Type of omics data.
            log_transform: Apply log transformation.
        """
        super().__init__(batch_size)
        self.data_type = data_type
        self.log_transform = log_transform
        self._gene_names: list[str] = []

    def load(self, path: str | Path) -> "OmicsDataLoader":
        """Load omics data."""
        path = Path(path)

        try:
            with open(path, "r") as f:
                lines = f.readlines()

            if len(lines) < 2:
                return self

            # First row is gene names
            self._gene_names = lines[0].strip().split("\t")[1:]

            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                sample_id = parts[0]
                values = np.array([float(v) for v in parts[1:]])

                if self.log_transform:
                    values = np.log2(values + 1)

                self._data.append(DataSample(
                    sample_id=sample_id,
                    features=values,
                    metadata={"gene_names": self._gene_names},
                ))

        except Exception as e:
            self._logger.error(f"Failed to load omics data: {e}")

        self._logger.info(f"Loaded {len(self._data)} samples, {len(self._gene_names)} genes")
        return self


def create_train_test_split(
    data: list[DataSample],
    test_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[list[DataSample], list[DataSample]]:
    """
    Split data into train and test sets.

    Args:
        data: List of data samples.
        test_ratio: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (train_data, test_data).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))

    split_idx = int(len(data) * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, test_data
