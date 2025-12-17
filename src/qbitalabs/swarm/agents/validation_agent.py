"""
Validation Agent for QBitaLabs SWARM

Specializes in result validation and verification:
- Cross-validate findings
- Check consistency across agents
- Verify data quality
- Assess statistical significance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog

from qbitalabs.core.types import AgentRole, MessageType
from qbitalabs.swarm.base_agent import AgentMessage, BaseAgent

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    validated: bool = False
    confidence: float = 0.0
    checks_passed: int = 0
    checks_total: int = 0
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ValidationAgent(BaseAgent):
    """
    Agent specializing in validation and verification.

    Capabilities:
    - Cross-validate findings from multiple agents
    - Check data quality and consistency
    - Verify statistical significance
    - Validate molecular structures
    - Assess reproducibility

    Example:
        >>> agent = ValidationAgent()
        >>> result = await agent.process({
        ...     "task": "validate_finding",
        ...     "finding": {"molecule": "CCO", "energy": -1.234},
        ...     "sources": ["agent_1", "agent_2"]
        ... })
    """

    def __init__(self, **kwargs: Any):
        """Initialize the validation agent."""
        kwargs.setdefault("role", AgentRole.VALIDATION_AGENT)
        super().__init__(**kwargs)

        # Validation history
        self._validations: dict[str, ValidationResult] = {}

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process validation tasks.

        Args:
            input_data: Task input with keys:
                - task: validate_finding, check_consistency, verify_statistics
                - finding: Data to validate
                - sources: Source agent IDs

        Returns:
            Validation results.
        """
        task = input_data.get("task", "validate_finding")

        try:
            if task == "validate_finding":
                finding = input_data.get("finding", {})
                sources = input_data.get("sources", [])
                result = await self._validate_finding(finding, sources)

            elif task == "check_consistency":
                results = input_data.get("results", [])
                result = await self._check_consistency(results)

            elif task == "verify_statistics":
                data = input_data.get("data", {})
                result = await self._verify_statistics(data)

            elif task == "validate_molecule":
                smiles = input_data.get("smiles", "")
                result = await self._validate_molecule(smiles)

            elif task == "assess_quality":
                dataset = input_data.get("dataset", {})
                result = await self._assess_data_quality(dataset)

            else:
                result = {"error": f"Unknown task: {task}"}

            # Record validation
            if isinstance(result, dict) and "validation_id" in result:
                validation_id = result["validation_id"]
                if result.get("validated", False):
                    await self.deposit_pheromone(
                        f"validated:{validation_id}", 3.0
                    )

            self.tasks_completed += 1
            return result

        except Exception as e:
            self._logger.exception("Validation error", error=str(e))
            return {"error": str(e)}

    async def respond_to_signal(
        self, message: AgentMessage
    ) -> AgentMessage | None:
        """Respond to signals from other agents."""
        if message.message_type == MessageType.QUERY:
            result = await self.process(message.payload)
            return AgentMessage(
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                payload=result,
                correlation_id=message.correlation_id,
            )

        elif message.message_type == MessageType.EVENT:
            event_type = message.payload.get("event_type")

            if event_type == "new_finding":
                # Automatically validate new findings
                finding = message.payload.get("finding", {})
                sources = [str(message.sender_id)]
                await self._validate_finding(finding, sources)

        return None

    async def _validate_finding(
        self, finding: dict[str, Any], sources: list[str]
    ) -> dict[str, Any]:
        """Validate a scientific finding."""
        validation = ValidationResult()
        checks_total = 0
        checks_passed = 0

        # Check 1: Data completeness
        checks_total += 1
        required_fields = finding.get("required_fields", ["result"])
        missing = [f for f in required_fields if f not in finding]
        if not missing:
            checks_passed += 1
        else:
            validation.issues.append(f"Missing fields: {missing}")

        # Check 2: Value ranges
        checks_total += 1
        if "energy" in finding:
            energy = finding["energy"]
            if isinstance(energy, (int, float)) and -100 < energy < 100:
                checks_passed += 1
            else:
                validation.issues.append(f"Energy value out of expected range: {energy}")

        # Check 3: Source reliability
        checks_total += 1
        if len(sources) >= 2:
            checks_passed += 1
        else:
            validation.recommendations.append(
                "Consider obtaining confirmation from additional sources"
            )

        # Check 4: Consistency with known data
        checks_total += 1
        if self._context:
            # Check against quantum results cache
            if "molecule_id" in finding:
                cached = self._context.quantum_results.get(finding["molecule_id"])
                if cached and "energy" in finding and "energy" in cached:
                    diff = abs(finding["energy"] - cached["energy"])
                    if diff < 0.1:  # Within 0.1 Ha tolerance
                        checks_passed += 1
                    else:
                        validation.issues.append(
                            f"Energy differs from cached value by {diff:.3f}"
                        )
                else:
                    checks_passed += 1  # No cached data to compare
            else:
                checks_passed += 1

        # Calculate validation result
        validation.checks_passed = checks_passed
        validation.checks_total = checks_total
        validation.confidence = checks_passed / checks_total if checks_total > 0 else 0
        validation.validated = validation.confidence >= 0.7

        self._validations[validation.id] = validation

        return {
            "validation_id": validation.id,
            "validated": validation.validated,
            "confidence": validation.confidence,
            "checks_passed": validation.checks_passed,
            "checks_total": validation.checks_total,
            "issues": validation.issues,
            "recommendations": validation.recommendations,
        }

    async def _check_consistency(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Check consistency across multiple results."""
        if len(results) < 2:
            return {
                "consistent": True,
                "message": "Need at least 2 results to check consistency",
            }

        inconsistencies = []
        numeric_values: dict[str, list[float]] = {}

        # Collect numeric values for each key
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_values:
                        numeric_values[key] = []
                    numeric_values[key].append(value)

        # Check variance for each numeric key
        for key, values in numeric_values.items():
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                cv = (variance ** 0.5) / abs(mean) if mean != 0 else 0

                if cv > 0.1:  # More than 10% coefficient of variation
                    inconsistencies.append({
                        "field": key,
                        "values": values,
                        "coefficient_of_variation": cv,
                    })

        return {
            "consistent": len(inconsistencies) == 0,
            "results_compared": len(results),
            "inconsistencies": inconsistencies,
        }

    async def _verify_statistics(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Verify statistical claims."""
        p_value = data.get("p_value")
        sample_size = data.get("sample_size")
        effect_size = data.get("effect_size")

        issues = []
        recommendations = []

        # Check p-value
        if p_value is not None:
            if p_value < 0 or p_value > 1:
                issues.append(f"Invalid p-value: {p_value}")
            elif p_value < 0.001:
                recommendations.append("Very small p-value; verify calculation method")

        # Check sample size
        if sample_size is not None:
            if sample_size < 30:
                recommendations.append(
                    "Small sample size may limit generalizability"
                )

        # Check effect size
        if effect_size is not None:
            if abs(effect_size) > 2:
                recommendations.append(
                    "Large effect size; verify data quality"
                )

        # Statistical power analysis
        power_adequate = True
        if sample_size and effect_size:
            # Simplified power check
            min_sample = 20 / (effect_size ** 2) if effect_size != 0 else 1000
            power_adequate = sample_size >= min_sample

        return {
            "valid": len(issues) == 0,
            "power_adequate": power_adequate,
            "issues": issues,
            "recommendations": recommendations,
        }

    async def _validate_molecule(self, smiles: str) -> dict[str, Any]:
        """Validate a molecular structure."""
        if not smiles:
            return {"valid": False, "error": "No SMILES provided"}

        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"valid": False, "error": "Invalid SMILES"}

            issues = []

            # Check valence
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                issues.append(f"Valence issue: {e}")

            # Check for 3D generation
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except Exception:
                issues.append("Could not generate 3D coordinates")

            return {
                "valid": len(issues) == 0,
                "smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "issues": issues,
            }

        except ImportError:
            return {
                "valid": True,
                "warning": "RDKit not available for validation",
                "smiles": smiles,
            }

    async def _assess_data_quality(
        self, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess quality of a dataset."""
        issues = []
        quality_score = 1.0

        # Check for missing values
        missing_rate = dataset.get("missing_rate", 0)
        if missing_rate > 0.1:
            issues.append(f"High missing rate: {missing_rate:.1%}")
            quality_score -= 0.2

        # Check for duplicates
        duplicate_rate = dataset.get("duplicate_rate", 0)
        if duplicate_rate > 0.05:
            issues.append(f"Duplicate rate: {duplicate_rate:.1%}")
            quality_score -= 0.1

        # Check for outliers
        outlier_rate = dataset.get("outlier_rate", 0)
        if outlier_rate > 0.05:
            issues.append(f"High outlier rate: {outlier_rate:.1%}")
            quality_score -= 0.15

        return {
            "quality_score": max(0, quality_score),
            "issues": issues,
            "sample_count": dataset.get("sample_count", 0),
            "feature_count": dataset.get("feature_count", 0),
        }
