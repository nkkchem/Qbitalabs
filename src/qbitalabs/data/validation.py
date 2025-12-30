"""
QBitaLabs Data Validation Framework

Comprehensive data validation with:
- Schema validation using Pydantic
- Data quality checks
- Statistical validation
- Domain-specific validators
- Composable validation pipelines

Authored by: QbitaLab
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DataQualityMetric(str, Enum):
    """Data quality metrics."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    code: str
    value: Any = None
    expected: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "value": str(self.value) if self.value is not None else None,
            "expected": str(self.expected) if self.expected is not None else None,
        }


@dataclass
class ValidationResult:
    """Result of a validation run."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validator_name: str = ""
    data_hash: str = ""

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.valid = False

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge two validation results."""
        return ValidationResult(
            valid=self.valid and other.valid,
            issues=self.issues + other.issues,
            metrics={**self.metrics, **other.metrics},
            validator_name=f"{self.validator_name}+{other.validator_name}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "validator_name": self.validator_name,
            "data_hash": self.data_hash,
            "error_count": len([i for i in self.issues if i.severity == ValidationSeverity.ERROR]),
            "warning_count": len([i for i in self.issues if i.severity == ValidationSeverity.WARNING]),
        }


# =============================================================================
# Pydantic Schema Models for Data Contracts
# =============================================================================

class MoleculeSchema(BaseModel):
    """Schema for molecular data."""
    smiles: str = Field(..., min_length=1, description="SMILES string")
    name: Optional[str] = Field(None, description="Molecule name")
    molecular_weight: Optional[float] = Field(None, ge=0, description="Molecular weight in Da")
    logp: Optional[float] = Field(None, ge=-10, le=20, description="LogP value")
    num_atoms: Optional[int] = Field(None, ge=1, description="Number of atoms")
    num_bonds: Optional[int] = Field(None, ge=0, description="Number of bonds")
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        """Basic SMILES validation."""
        # Check for valid SMILES characters
        valid_chars = set("CNOSPFClBrI[]()=#@+-.0123456789cnosp/\\")
        if not all(c in valid_chars for c in v):
            raise ValueError(f"Invalid SMILES characters in: {v}")
        # Check bracket balance
        if v.count("[") != v.count("]"):
            raise ValueError("Unbalanced brackets in SMILES")
        if v.count("(") != v.count(")"):
            raise ValueError("Unbalanced parentheses in SMILES")
        return v


class PatientSchema(BaseModel):
    """Schema for patient data."""
    patient_id: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., pattern="^(M|F|O)$")
    height_cm: Optional[float] = Field(None, ge=30, le=300)
    weight_kg: Optional[float] = Field(None, ge=1, le=700)
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    lab_results: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bmi_range(self) -> "PatientSchema":
        """Validate BMI is within reasonable range if both height and weight are provided."""
        if self.height_cm and self.weight_kg:
            bmi = self.weight_kg / ((self.height_cm / 100) ** 2)
            if bmi < 10 or bmi > 100:
                raise ValueError(f"Calculated BMI {bmi:.1f} is outside reasonable range")
        return self


class ClinicalTrialSchema(BaseModel):
    """Schema for clinical trial data."""
    trial_id: str = Field(..., pattern="^NCT[0-9]{8}$|^[A-Z]{2,5}-[0-9]+$")
    phase: str = Field(..., pattern="^(Phase [1-4]|Phase 1/2|Phase 2/3|Preclinical)$")
    status: str = Field(..., pattern="^(Recruiting|Active|Completed|Terminated|Suspended)$")
    start_date: datetime
    enrollment: int = Field(..., ge=0)
    primary_endpoint: str = Field(..., min_length=1)
    arms: List[str] = Field(..., min_length=1)


class OmicsDataSchema(BaseModel):
    """Schema for omics data (genomics, proteomics, metabolomics)."""
    sample_id: str = Field(..., min_length=1)
    data_type: str = Field(..., pattern="^(genomics|transcriptomics|proteomics|metabolomics)$")
    platform: str = Field(..., min_length=1)
    values: Dict[str, float] = Field(..., min_length=1)
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    batch_id: Optional[str] = None

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate omics values."""
        for key, value in v.items():
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"Invalid value for {key}: {value}")
        return v


class BiomarkerSchema(BaseModel):
    """Schema for biomarker data."""
    biomarker_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    value: float
    unit: str = Field(..., min_length=1)
    reference_range: Optional[Tuple[float, float]] = None
    timestamp: datetime

    @model_validator(mode="after")
    def check_reference_range(self) -> "BiomarkerSchema":
        """Check if value is within reference range."""
        if self.reference_range:
            low, high = self.reference_range
            if not (low <= self.value <= high):
                logger.warning(
                    "Biomarker outside reference range",
                    biomarker=self.name,
                    value=self.value,
                    range=self.reference_range,
                )
        return self


# =============================================================================
# Validator Base Class and Implementations
# =============================================================================

class Validator(ABC, Generic[T]):
    """Abstract base class for validators."""

    name: str = "base_validator"

    @abstractmethod
    def validate(self, data: T) -> ValidationResult:
        """Validate the data and return a result."""
        pass

    def __call__(self, data: T) -> ValidationResult:
        return self.validate(data)


class CompositeValidator(Validator[T]):
    """Combines multiple validators."""

    def __init__(self, validators: List[Validator[T]], name: str = "composite"):
        self.validators = validators
        self.name = name

    def validate(self, data: T) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        for validator in self.validators:
            sub_result = validator.validate(data)
            result = result.merge(sub_result)
        return result


class SchemaValidator(Validator[Dict[str, Any]]):
    """Validates data against a Pydantic schema."""

    def __init__(self, schema: type[BaseModel], name: Optional[str] = None):
        self.schema = schema
        self.name = name or f"schema_{schema.__name__}"

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        try:
            self.schema.model_validate(data)
        except Exception as e:
            result.add_issue(ValidationIssue(
                field="schema",
                message=str(e),
                severity=ValidationSeverity.ERROR,
                code="SCHEMA_VALIDATION_ERROR",
            ))
        return result


class RangeValidator(Validator[Dict[str, Any]]):
    """Validates numeric fields are within specified ranges."""

    def __init__(self, ranges: Dict[str, Tuple[float, float]], name: str = "range_validator"):
        self.ranges = ranges
        self.name = name

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        for field, (min_val, max_val) in self.ranges.items():
            if field in data:
                value = data[field]
                if value is not None and not (min_val <= value <= max_val):
                    result.add_issue(ValidationIssue(
                        field=field,
                        message=f"Value {value} outside range [{min_val}, {max_val}]",
                        severity=ValidationSeverity.ERROR,
                        code="RANGE_VIOLATION",
                        value=value,
                        expected=f"[{min_val}, {max_val}]",
                    ))
        return result


class NullCheckValidator(Validator[Dict[str, Any]]):
    """Validates required fields are not null/empty."""

    def __init__(self, required_fields: List[str], name: str = "null_check"):
        self.required_fields = required_fields
        self.name = name

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        for field in self.required_fields:
            if field not in data or data[field] is None or data[field] == "":
                result.add_issue(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is missing or null",
                    severity=ValidationSeverity.ERROR,
                    code="REQUIRED_FIELD_MISSING",
                ))
        return result


class UniqueValidator(Validator[List[Dict[str, Any]]]):
    """Validates uniqueness of specified fields across records."""

    def __init__(self, unique_fields: List[str], name: str = "unique_validator"):
        self.unique_fields = unique_fields
        self.name = name

    def validate(self, data: List[Dict[str, Any]]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)
        for field in self.unique_fields:
            values = [record.get(field) for record in data if record.get(field) is not None]
            duplicates = [v for v in values if values.count(v) > 1]
            if duplicates:
                result.add_issue(ValidationIssue(
                    field=field,
                    message=f"Duplicate values found: {set(duplicates)}",
                    severity=ValidationSeverity.ERROR,
                    code="DUPLICATE_VALUES",
                    value=list(set(duplicates)),
                ))
        return result


class StatisticalValidator(Validator[Dict[str, Any]]):
    """Validates statistical properties of numeric data."""

    def __init__(
        self,
        field: str,
        expected_mean: Optional[float] = None,
        expected_std: Optional[float] = None,
        max_outlier_ratio: float = 0.05,
        name: str = "statistical_validator",
    ):
        self.field = field
        self.expected_mean = expected_mean
        self.expected_std = expected_std
        self.max_outlier_ratio = max_outlier_ratio
        self.name = name

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)

        if self.field not in data:
            return result

        values = np.array(data[self.field])
        if len(values) == 0:
            return result

        mean = np.mean(values)
        std = np.std(values)

        # Store metrics
        result.metrics[f"{self.field}_mean"] = float(mean)
        result.metrics[f"{self.field}_std"] = float(std)

        # Check mean drift
        if self.expected_mean is not None:
            drift = abs(mean - self.expected_mean) / (self.expected_std or 1)
            if drift > 3:  # More than 3 sigma drift
                result.add_issue(ValidationIssue(
                    field=self.field,
                    message=f"Mean {mean:.4f} significantly different from expected {self.expected_mean:.4f}",
                    severity=ValidationSeverity.WARNING,
                    code="MEAN_DRIFT",
                    value=mean,
                    expected=self.expected_mean,
                ))

        # Check outlier ratio using IQR method
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((values < lower_bound) | (values > upper_bound))
        outlier_ratio = outliers / len(values)

        result.metrics[f"{self.field}_outlier_ratio"] = float(outlier_ratio)

        if outlier_ratio > self.max_outlier_ratio:
            result.add_issue(ValidationIssue(
                field=self.field,
                message=f"Outlier ratio {outlier_ratio:.2%} exceeds threshold {self.max_outlier_ratio:.2%}",
                severity=ValidationSeverity.WARNING,
                code="HIGH_OUTLIER_RATIO",
                value=outlier_ratio,
                expected=self.max_outlier_ratio,
            ))

        return result


class MolecularValidator(Validator[Dict[str, Any]]):
    """Domain-specific validator for molecular data."""

    name = "molecular_validator"

    # Common molecular constraints
    MAX_HEAVY_ATOMS = 100
    MAX_RINGS = 20
    VALID_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)

        smiles = data.get("smiles", "")
        if not smiles:
            result.add_issue(ValidationIssue(
                field="smiles",
                message="SMILES string is required",
                severity=ValidationSeverity.ERROR,
                code="MISSING_SMILES",
            ))
            return result

        # Validate SMILES syntax
        try:
            # Basic syntax checks
            if smiles.count("[") != smiles.count("]"):
                raise ValueError("Unbalanced brackets")
            if smiles.count("(") != smiles.count(")"):
                raise ValueError("Unbalanced parentheses")

            # Check for invalid patterns
            invalid_patterns = [
                (r"\[{2,}", "Double brackets"),
                (r"\({2,}", "Double parentheses"),
                (r"^[0-9]", "SMILES cannot start with a number"),
            ]
            for pattern, description in invalid_patterns:
                if re.search(pattern, smiles):
                    raise ValueError(description)

        except ValueError as e:
            result.add_issue(ValidationIssue(
                field="smiles",
                message=f"Invalid SMILES syntax: {e}",
                severity=ValidationSeverity.ERROR,
                code="INVALID_SMILES_SYNTAX",
                value=smiles,
            ))

        # Validate molecular weight if provided
        if "molecular_weight" in data:
            mw = data["molecular_weight"]
            if mw < 10 or mw > 5000:
                result.add_issue(ValidationIssue(
                    field="molecular_weight",
                    message=f"Molecular weight {mw} is outside typical drug-like range [10, 5000]",
                    severity=ValidationSeverity.WARNING,
                    code="UNUSUAL_MOLECULAR_WEIGHT",
                    value=mw,
                ))

        # Validate LogP (drug-likeness)
        if "logp" in data:
            logp = data["logp"]
            if logp < -5 or logp > 10:
                result.add_issue(ValidationIssue(
                    field="logp",
                    message=f"LogP {logp} violates Lipinski's rule of 5 extended range",
                    severity=ValidationSeverity.WARNING,
                    code="LIPINSKI_LOGP_VIOLATION",
                    value=logp,
                ))

        return result


class ClinicalValidator(Validator[Dict[str, Any]]):
    """Domain-specific validator for clinical data."""

    name = "clinical_validator"

    # Reference ranges for common lab values
    LAB_REFERENCE_RANGES = {
        "hemoglobin": (12.0, 17.5),  # g/dL
        "glucose": (70, 100),  # mg/dL fasting
        "creatinine": (0.6, 1.2),  # mg/dL
        "alt": (7, 56),  # U/L
        "ast": (10, 40),  # U/L
        "cholesterol_total": (0, 200),  # mg/dL
        "ldl": (0, 100),  # mg/dL
        "hdl": (40, 60),  # mg/dL
        "triglycerides": (0, 150),  # mg/dL
        "potassium": (3.5, 5.0),  # mEq/L
        "sodium": (136, 145),  # mEq/L
    }

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult(valid=True, validator_name=self.name)

        # Validate lab results
        lab_results = data.get("lab_results", {})
        for lab, value in lab_results.items():
            lab_key = lab.lower().replace(" ", "_")
            if lab_key in self.LAB_REFERENCE_RANGES:
                low, high = self.LAB_REFERENCE_RANGES[lab_key]
                if value < low or value > high:
                    severity = ValidationSeverity.WARNING
                    # Critical values get ERROR severity
                    if lab_key == "potassium" and (value < 2.5 or value > 6.5):
                        severity = ValidationSeverity.ERROR
                    if lab_key == "glucose" and (value < 40 or value > 500):
                        severity = ValidationSeverity.ERROR

                    result.add_issue(ValidationIssue(
                        field=f"lab_results.{lab}",
                        message=f"{lab} value {value} outside reference range [{low}, {high}]",
                        severity=severity,
                        code="LAB_OUT_OF_RANGE",
                        value=value,
                        expected=f"[{low}, {high}]",
                    ))

        # Validate vital signs
        vitals = data.get("vitals", {})
        if "heart_rate" in vitals:
            hr = vitals["heart_rate"]
            if hr < 40 or hr > 200:
                result.add_issue(ValidationIssue(
                    field="vitals.heart_rate",
                    message=f"Heart rate {hr} is concerning",
                    severity=ValidationSeverity.ERROR if (hr < 30 or hr > 220) else ValidationSeverity.WARNING,
                    code="ABNORMAL_HEART_RATE",
                    value=hr,
                ))

        if "blood_pressure_systolic" in vitals:
            sbp = vitals["blood_pressure_systolic"]
            if sbp < 70 or sbp > 200:
                result.add_issue(ValidationIssue(
                    field="vitals.blood_pressure_systolic",
                    message=f"Systolic BP {sbp} is abnormal",
                    severity=ValidationSeverity.ERROR if (sbp < 60 or sbp > 220) else ValidationSeverity.WARNING,
                    code="ABNORMAL_BLOOD_PRESSURE",
                    value=sbp,
                ))

        return result


# =============================================================================
# Data Quality Profiler
# =============================================================================

class DataQualityProfiler:
    """Profiles data quality metrics across a dataset."""

    def __init__(self):
        self._logger = structlog.get_logger("DataQualityProfiler")

    def profile(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive data quality profile."""
        if not data:
            return {"error": "Empty dataset"}

        profile = {
            "record_count": len(data),
            "fields": {},
            "overall_quality": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Analyze each field
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())

        for field in all_fields:
            values = [record.get(field) for record in data]
            profile["fields"][field] = self._analyze_field(field, values)

        # Calculate overall quality score
        field_scores = [f["quality_score"] for f in profile["fields"].values()]
        profile["overall_quality"] = np.mean(field_scores) if field_scores else 0.0

        return profile

    def _analyze_field(self, field: str, values: List[Any]) -> Dict[str, Any]:
        """Analyze a single field."""
        total = len(values)
        non_null = [v for v in values if v is not None and v != ""]
        null_count = total - len(non_null)

        analysis = {
            "total_count": total,
            "null_count": null_count,
            "null_rate": null_count / total if total > 0 else 0,
            "unique_count": len(set(str(v) for v in non_null)),
            "data_type": self._infer_type(non_null),
            "quality_score": 0.0,
        }

        # Add numeric statistics if applicable
        if analysis["data_type"] == "numeric":
            numeric_values = [float(v) for v in non_null if self._is_numeric(v)]
            if numeric_values:
                analysis["min"] = min(numeric_values)
                analysis["max"] = max(numeric_values)
                analysis["mean"] = np.mean(numeric_values)
                analysis["std"] = np.std(numeric_values)
                analysis["median"] = np.median(numeric_values)

        # Calculate quality score
        completeness = 1 - analysis["null_rate"]
        uniqueness = analysis["unique_count"] / len(non_null) if non_null else 0
        analysis["quality_score"] = (completeness * 0.6 + uniqueness * 0.4)

        return analysis

    def _infer_type(self, values: List[Any]) -> str:
        """Infer the data type of a field."""
        if not values:
            return "unknown"

        numeric_count = sum(1 for v in values if self._is_numeric(v))
        if numeric_count / len(values) > 0.9:
            return "numeric"

        if all(isinstance(v, bool) for v in values):
            return "boolean"

        return "string"

    def _is_numeric(self, value: Any) -> bool:
        """Check if a value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """Orchestrates multiple validators in a pipeline."""

    def __init__(self, name: str = "validation_pipeline"):
        self.name = name
        self.validators: List[Tuple[str, Validator, bool]] = []  # (name, validator, required)
        self._logger = structlog.get_logger("ValidationPipeline")

    def add_validator(
        self,
        name: str,
        validator: Validator,
        required: bool = True,
    ) -> "ValidationPipeline":
        """Add a validator to the pipeline."""
        self.validators.append((name, validator, required))
        return self

    def validate(self, data: Any) -> ValidationResult:
        """Run all validators and aggregate results."""
        combined_result = ValidationResult(valid=True, validator_name=self.name)
        combined_result.data_hash = self._compute_hash(data)

        for name, validator, required in self.validators:
            try:
                result = validator.validate(data)
                combined_result = combined_result.merge(result)

                self._logger.info(
                    "Validator completed",
                    validator=name,
                    valid=result.valid,
                    issues=len(result.issues),
                )

                # Stop on required validator failure if configured
                if required and not result.valid:
                    self._logger.warning("Required validator failed, stopping pipeline", validator=name)
                    break

            except Exception as e:
                self._logger.error("Validator error", validator=name, error=str(e))
                combined_result.add_issue(ValidationIssue(
                    field="pipeline",
                    message=f"Validator '{name}' raised exception: {e}",
                    severity=ValidationSeverity.ERROR,
                    code="VALIDATOR_EXCEPTION",
                ))
                if required:
                    break

        return combined_result

    def _compute_hash(self, data: Any) -> str:
        """Compute a hash of the data for tracking."""
        try:
            import json
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        except Exception:
            return "unhashable"


# =============================================================================
# Pre-built Validation Pipelines
# =============================================================================

def create_molecule_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for molecular data."""
    pipeline = ValidationPipeline("molecule_validation")
    pipeline.add_validator("schema", SchemaValidator(MoleculeSchema))
    pipeline.add_validator("molecular", MolecularValidator())
    pipeline.add_validator("ranges", RangeValidator({
        "molecular_weight": (10, 5000),
        "logp": (-10, 20),
        "num_atoms": (1, 500),
    }))
    return pipeline


def create_patient_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for patient data."""
    pipeline = ValidationPipeline("patient_validation")
    pipeline.add_validator("schema", SchemaValidator(PatientSchema))
    pipeline.add_validator("clinical", ClinicalValidator())
    pipeline.add_validator("required", NullCheckValidator(["patient_id", "age", "sex"]))
    return pipeline


def create_clinical_trial_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for clinical trial data."""
    pipeline = ValidationPipeline("clinical_trial_validation")
    pipeline.add_validator("schema", SchemaValidator(ClinicalTrialSchema))
    pipeline.add_validator("required", NullCheckValidator([
        "trial_id", "phase", "status", "start_date", "primary_endpoint"
    ]))
    return pipeline


def create_omics_pipeline() -> ValidationPipeline:
    """Create a validation pipeline for omics data."""
    pipeline = ValidationPipeline("omics_validation")
    pipeline.add_validator("schema", SchemaValidator(OmicsDataSchema))
    pipeline.add_validator("required", NullCheckValidator(["sample_id", "data_type", "values"]))
    return pipeline


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core classes
    "ValidationSeverity",
    "DataQualityMetric",
    "ValidationIssue",
    "ValidationResult",

    # Schemas
    "MoleculeSchema",
    "PatientSchema",
    "ClinicalTrialSchema",
    "OmicsDataSchema",
    "BiomarkerSchema",

    # Validators
    "Validator",
    "CompositeValidator",
    "SchemaValidator",
    "RangeValidator",
    "NullCheckValidator",
    "UniqueValidator",
    "StatisticalValidator",
    "MolecularValidator",
    "ClinicalValidator",

    # Pipeline
    "ValidationPipeline",
    "DataQualityProfiler",

    # Pre-built pipelines
    "create_molecule_pipeline",
    "create_patient_pipeline",
    "create_clinical_trial_pipeline",
    "create_omics_pipeline",
]
