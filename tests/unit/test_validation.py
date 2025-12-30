"""
QbitaLab: Unit tests for data validation framework.

Tests:
- Schema validation
- Domain validators
- Validation pipelines
- Data quality profiling
"""

import pytest
from datetime import datetime
from typing import Dict, Any


class TestMoleculeSchema:
    """Tests for MoleculeSchema validation."""

    def test_valid_smiles(self):
        """Test valid SMILES validation."""
        from qbitalabs.data.validation import MoleculeSchema

        # Valid molecules
        valid_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CC(C)C",  # Isobutane
            "C1CCCCC1",  # Cyclohexane
        ]

        for smiles in valid_smiles:
            molecule = MoleculeSchema(smiles=smiles)
            assert molecule.smiles == smiles

    def test_invalid_smiles_brackets(self):
        """Test SMILES with unbalanced brackets."""
        from qbitalabs.data.validation import MoleculeSchema
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            MoleculeSchema(smiles="C[C")

    def test_invalid_smiles_parentheses(self):
        """Test SMILES with unbalanced parentheses."""
        from qbitalabs.data.validation import MoleculeSchema
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            MoleculeSchema(smiles="CC(=O")

    def test_molecule_properties(self):
        """Test molecule with additional properties."""
        from qbitalabs.data.validation import MoleculeSchema

        molecule = MoleculeSchema(
            smiles="CCO",
            name="Ethanol",
            molecular_weight=46.07,
            logp=-0.31,
            num_atoms=9,
            properties={"pka": 15.9},
        )

        assert molecule.name == "Ethanol"
        assert molecule.molecular_weight == 46.07
        assert molecule.logp == -0.31


class TestPatientSchema:
    """Tests for PatientSchema validation."""

    def test_valid_patient(self):
        """Test valid patient data."""
        from qbitalabs.data.validation import PatientSchema

        patient = PatientSchema(
            patient_id="P001",
            age=45,
            sex="M",
            height_cm=175.0,
            weight_kg=70.0,
            conditions=["hypertension"],
            medications=["lisinopril"],
        )

        assert patient.patient_id == "P001"
        assert patient.age == 45

    def test_invalid_age(self):
        """Test patient with invalid age."""
        from qbitalabs.data.validation import PatientSchema
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            PatientSchema(patient_id="P001", age=-5, sex="M")

        with pytest.raises(pydantic.ValidationError):
            PatientSchema(patient_id="P001", age=200, sex="M")

    def test_invalid_sex(self):
        """Test patient with invalid sex code."""
        from qbitalabs.data.validation import PatientSchema
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            PatientSchema(patient_id="P001", age=45, sex="X")

    def test_bmi_validation(self):
        """Test BMI range validation."""
        from qbitalabs.data.validation import PatientSchema
        import pydantic

        # Valid BMI
        patient = PatientSchema(
            patient_id="P001",
            age=45,
            sex="M",
            height_cm=175.0,
            weight_kg=70.0,
        )
        bmi = 70.0 / (1.75 ** 2)
        assert 18 < bmi < 30  # Normal range

        # Invalid BMI (too low)
        with pytest.raises(pydantic.ValidationError):
            PatientSchema(
                patient_id="P001",
                age=45,
                sex="M",
                height_cm=175.0,
                weight_kg=10.0,  # Too light
            )


class TestValidators:
    """Tests for individual validators."""

    def test_range_validator(self):
        """Test RangeValidator."""
        from qbitalabs.data.validation import RangeValidator, ValidationSeverity

        validator = RangeValidator({
            "temperature": (35.0, 42.0),
            "heart_rate": (40, 200),
        })

        # Valid data
        result = validator.validate({"temperature": 37.0, "heart_rate": 72})
        assert result.valid
        assert len(result.issues) == 0

        # Invalid temperature
        result = validator.validate({"temperature": 45.0, "heart_rate": 72})
        assert not result.valid
        assert len(result.issues) == 1
        assert result.issues[0].severity == ValidationSeverity.ERROR

    def test_null_check_validator(self):
        """Test NullCheckValidator."""
        from qbitalabs.data.validation import NullCheckValidator

        validator = NullCheckValidator(["id", "name", "value"])

        # Valid data
        result = validator.validate({"id": 1, "name": "test", "value": 100})
        assert result.valid

        # Missing field
        result = validator.validate({"id": 1, "value": 100})
        assert not result.valid
        assert any("name" in i.field for i in result.issues)

        # Null field
        result = validator.validate({"id": 1, "name": None, "value": 100})
        assert not result.valid

    def test_unique_validator(self):
        """Test UniqueValidator."""
        from qbitalabs.data.validation import UniqueValidator

        validator = UniqueValidator(["id"])

        # Unique values
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = validator.validate(data)
        assert result.valid

        # Duplicate values
        data = [{"id": 1}, {"id": 2}, {"id": 1}]
        result = validator.validate(data)
        assert not result.valid
        assert "1" in str(result.issues[0].value)

    def test_statistical_validator(self):
        """Test StatisticalValidator."""
        from qbitalabs.data.validation import StatisticalValidator
        import numpy as np

        # Normal data
        values = list(np.random.normal(100, 10, 1000))
        validator = StatisticalValidator(
            field="values",
            expected_mean=100,
            expected_std=10,
            max_outlier_ratio=0.1,
        )

        result = validator.validate({"values": values})
        assert result.valid
        assert "values_mean" in result.metrics
        assert "values_std" in result.metrics


class TestMolecularValidator:
    """Tests for MolecularValidator."""

    def test_valid_molecule(self):
        """Test valid molecule validation."""
        from qbitalabs.data.validation import MolecularValidator

        validator = MolecularValidator()

        result = validator.validate({
            "smiles": "CCO",
            "molecular_weight": 46.07,
            "logp": -0.31,
        })
        assert result.valid

    def test_invalid_smiles_syntax(self):
        """Test invalid SMILES syntax detection."""
        from qbitalabs.data.validation import MolecularValidator

        validator = MolecularValidator()

        result = validator.validate({"smiles": "C[C"})
        assert not result.valid
        assert any("bracket" in i.message.lower() for i in result.issues)

    def test_unusual_molecular_weight(self):
        """Test unusual molecular weight warning."""
        from qbitalabs.data.validation import MolecularValidator, ValidationSeverity

        validator = MolecularValidator()

        result = validator.validate({
            "smiles": "CCO",
            "molecular_weight": 10000.0,  # Very high
        })

        warnings = [i for i in result.issues if i.severity == ValidationSeverity.WARNING]
        assert len(warnings) > 0

    def test_lipinski_violation(self):
        """Test Lipinski rule violation detection."""
        from qbitalabs.data.validation import MolecularValidator

        validator = MolecularValidator()

        result = validator.validate({
            "smiles": "CCO",
            "logp": 15.0,  # Way too high
        })

        assert any("lipinski" in i.code.lower() for i in result.issues)


class TestClinicalValidator:
    """Tests for ClinicalValidator."""

    def test_normal_lab_values(self):
        """Test normal lab value validation."""
        from qbitalabs.data.validation import ClinicalValidator

        validator = ClinicalValidator()

        result = validator.validate({
            "lab_results": {
                "hemoglobin": 14.0,
                "glucose": 90,
                "creatinine": 1.0,
            }
        })
        assert result.valid

    def test_abnormal_lab_values(self):
        """Test abnormal lab value detection."""
        from qbitalabs.data.validation import ClinicalValidator, ValidationSeverity

        validator = ClinicalValidator()

        result = validator.validate({
            "lab_results": {
                "glucose": 250,  # High
                "potassium": 6.0,  # High
            }
        })

        assert len(result.issues) >= 2

    def test_critical_lab_values(self):
        """Test critical lab value detection."""
        from qbitalabs.data.validation import ClinicalValidator, ValidationSeverity

        validator = ClinicalValidator()

        result = validator.validate({
            "lab_results": {
                "potassium": 7.0,  # Critical!
            }
        })

        errors = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(errors) > 0


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_pipeline_execution(self):
        """Test validation pipeline execution."""
        from qbitalabs.data.validation import (
            ValidationPipeline,
            SchemaValidator,
            MolecularValidator,
            MoleculeSchema,
        )

        pipeline = ValidationPipeline("molecule_pipeline")
        pipeline.add_validator("schema", SchemaValidator(MoleculeSchema))
        pipeline.add_validator("molecular", MolecularValidator())

        # Valid molecule
        result = pipeline.validate({
            "smiles": "CCO",
            "molecular_weight": 46.07,
        })
        assert result.valid

        # Invalid molecule
        result = pipeline.validate({
            "smiles": "C[C",  # Invalid
            "molecular_weight": 46.07,
        })
        assert not result.valid

    def test_pipeline_required_validator(self):
        """Test that required validators stop pipeline on failure."""
        from qbitalabs.data.validation import (
            ValidationPipeline,
            NullCheckValidator,
            RangeValidator,
        )

        pipeline = ValidationPipeline("test_pipeline")
        pipeline.add_validator("null_check", NullCheckValidator(["id"]), required=True)
        pipeline.add_validator("range", RangeValidator({"value": (0, 100)}), required=False)

        # Missing required field should stop pipeline
        result = pipeline.validate({"value": 50})  # Missing "id"
        assert not result.valid
        # Range validator shouldn't have run
        assert all("null" in i.code.lower() or "required" in i.code.lower() for i in result.issues)

    def test_pipeline_data_hash(self):
        """Test that pipeline computes data hash."""
        from qbitalabs.data.validation import ValidationPipeline

        pipeline = ValidationPipeline("test_pipeline")
        result = pipeline.validate({"key": "value"})

        assert result.data_hash != ""
        assert len(result.data_hash) == 16


class TestDataQualityProfiler:
    """Tests for DataQualityProfiler."""

    def test_profiler_basic(self):
        """Test basic profiling."""
        from qbitalabs.data.validation import DataQualityProfiler

        profiler = DataQualityProfiler()

        data = [
            {"id": 1, "name": "A", "value": 100},
            {"id": 2, "name": "B", "value": 200},
            {"id": 3, "name": "C", "value": 300},
        ]

        profile = profiler.profile(data)

        assert profile["record_count"] == 3
        assert "id" in profile["fields"]
        assert "name" in profile["fields"]
        assert "value" in profile["fields"]

    def test_profiler_null_handling(self):
        """Test profiling with null values."""
        from qbitalabs.data.validation import DataQualityProfiler

        profiler = DataQualityProfiler()

        data = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": None},
            {"id": 3, "name": "C"},
        ]

        profile = profiler.profile(data)

        name_field = profile["fields"]["name"]
        assert name_field["null_count"] == 1
        assert name_field["null_rate"] == pytest.approx(1/3, rel=0.01)

    def test_profiler_numeric_stats(self):
        """Test numeric field statistics."""
        from qbitalabs.data.validation import DataQualityProfiler

        profiler = DataQualityProfiler()

        data = [
            {"value": 10},
            {"value": 20},
            {"value": 30},
            {"value": 40},
            {"value": 50},
        ]

        profile = profiler.profile(data)

        value_field = profile["fields"]["value"]
        assert value_field["data_type"] == "numeric"
        assert value_field["min"] == 10
        assert value_field["max"] == 50
        assert value_field["mean"] == 30

    def test_profiler_quality_score(self):
        """Test overall quality score calculation."""
        from qbitalabs.data.validation import DataQualityProfiler

        profiler = DataQualityProfiler()

        # High quality data
        data = [{"id": i, "name": f"Name{i}"} for i in range(100)]
        profile = profiler.profile(data)
        assert profile["overall_quality"] > 0.8

        # Low quality data (many nulls)
        data = [{"id": i, "name": None} for i in range(100)]
        profile = profiler.profile(data)
        assert profile["overall_quality"] < 0.8


class TestPrebuildPipelines:
    """Tests for pre-built validation pipelines."""

    def test_molecule_pipeline(self):
        """Test pre-built molecule pipeline."""
        from qbitalabs.data.validation import create_molecule_pipeline

        pipeline = create_molecule_pipeline()

        # Valid molecule
        result = pipeline.validate({
            "smiles": "CCO",
            "molecular_weight": 46.07,
            "logp": -0.31,
        })
        assert result.valid

    def test_patient_pipeline(self):
        """Test pre-built patient pipeline."""
        from qbitalabs.data.validation import create_patient_pipeline

        pipeline = create_patient_pipeline()

        # Valid patient
        result = pipeline.validate({
            "patient_id": "P001",
            "age": 45,
            "sex": "M",
            "height_cm": 175.0,
            "weight_kg": 70.0,
        })
        assert result.valid

    def test_omics_pipeline(self):
        """Test pre-built omics pipeline."""
        from qbitalabs.data.validation import create_omics_pipeline

        pipeline = create_omics_pipeline()

        # Valid omics data
        result = pipeline.validate({
            "sample_id": "S001",
            "data_type": "transcriptomics",
            "platform": "RNA-seq",
            "values": {"BRCA1": 5.2, "TP53": 3.1},
        })
        assert result.valid
