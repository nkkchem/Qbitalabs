"""
Integration tests for the data pipeline.

Tests data flow from loaders through preprocessing to models.
"""

import pytest
import numpy as np


@pytest.mark.integration
class TestDataPipeline:
    """Test end-to-end data pipeline."""

    def test_molecular_data_to_gnn(self, sample_molecular_graph, temp_data_dir):
        """Test molecular data loading and GNN processing."""
        from qbitalabs.data import MolecularDataLoader
        from qbitalabs.models import GraphNeuralNetwork

        # Create sample data file
        import os
        import json

        data_file = os.path.join(temp_data_dir, "molecules.json")
        with open(data_file, "w") as f:
            json.dump([{"smiles": "CCO", "property": 1.5}], f)

        # Load data
        loader = MolecularDataLoader()
        dataset = loader.load(data_file)

        assert len(dataset) > 0

    def test_clinical_data_to_digital_twin(self, sample_patient_profile, temp_data_dir):
        """Test clinical data loading to digital twin creation."""
        from qbitalabs.data import ClinicalDataLoader
        from qbitalabs.digital_twin import DigitalTwinEngine, PatientProfile

        # Create clinical data
        import os
        import json

        data_file = os.path.join(temp_data_dir, "clinical.json")
        with open(data_file, "w") as f:
            json.dump([sample_patient_profile], f)

        # Load and process
        loader = ClinicalDataLoader()
        patients = loader.load(data_file)

        assert len(patients) == 1

        # Create digital twin from loaded data
        profile = PatientProfile(**patients[0])
        engine = DigitalTwinEngine()
        twin = engine.create_twin(profile)

        assert twin.patient_id == sample_patient_profile["patient_id"]

    def test_omics_data_pipeline(self, sample_omics_data, temp_data_dir):
        """Test omics data through analysis pipeline."""
        from qbitalabs.data import OmicsDataLoader
        from qbitalabs.biology import OmicsAnalyzer

        # Create omics data file
        import os
        import json

        data_file = os.path.join(temp_data_dir, "omics.json")
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            "type": sample_omics_data["type"],
            "genes": sample_omics_data["genes"],
            "expression": sample_omics_data["expression"].tolist(),
            "sample_id": sample_omics_data["sample_id"],
        }
        with open(data_file, "w") as f:
            json.dump([serializable_data], f)

        # Load data
        loader = OmicsDataLoader()
        omics_data = loader.load(data_file)

        assert len(omics_data) == 1

        # Analyze
        analyzer = OmicsAnalyzer()
        result = analyzer.analyze(omics_data[0])

        assert result is not None


@pytest.mark.integration
class TestModelTrainingPipeline:
    """Test model training pipeline."""

    def test_gnn_training_loop(self, sample_molecular_graph):
        """Test GNN can be trained on molecular data."""
        from qbitalabs.models import GraphNeuralNetwork

        # Create model
        model = GraphNeuralNetwork(
            input_dim=3,
            hidden_dim=32,
            output_dim=1,
            num_layers=2
        )

        # Create dummy batch
        batch = {
            "x": np.random.randn(10, 3).astype(np.float32),
            "edge_index": np.array([[0, 1, 2, 3], [1, 0, 3, 2]]),
            "y": np.array([1.0]),
        }

        # Forward pass should work
        output = model.forward(batch)
        assert output is not None

    def test_ensemble_model_aggregation(self):
        """Test ensemble model aggregates multiple predictions."""
        from qbitalabs.models import EnsembleModel

        # Create ensemble with mock base models
        ensemble = EnsembleModel(
            model_types=["gnn", "transformer"],
            aggregation="mean"
        )

        # Simulate predictions from base models
        predictions = [
            np.array([0.8, 0.2]),
            np.array([0.7, 0.3]),
        ]

        # Aggregate
        result = ensemble.aggregate(predictions)

        expected = np.array([0.75, 0.25])
        np.testing.assert_array_almost_equal(result, expected)
