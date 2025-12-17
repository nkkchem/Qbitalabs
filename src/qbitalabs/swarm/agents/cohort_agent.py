"""
Cohort Agent for QBitaLabs SWARM

Specializes in cohort management and analysis:
- Cohort definition and selection
- Population analysis
- Subgroup identification
- Cohort comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from qbitalabs.core.types import AgentRole, MessageType
from qbitalabs.swarm.base_agent import AgentMessage, BaseAgent

logger = structlog.get_logger(__name__)


@dataclass
class Cohort:
    """A patient cohort."""

    cohort_id: str = ""
    name: str = ""
    description: str = ""
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    size: int = 0
    demographics: dict[str, Any] = field(default_factory=dict)


class CohortAgent(BaseAgent):
    """
    Agent specializing in cohort management and analysis.

    Capabilities:
    - Define and select patient cohorts
    - Analyze cohort characteristics
    - Identify subgroups
    - Compare cohorts
    - Track cohort outcomes

    Example:
        >>> agent = CohortAgent()
        >>> result = await agent.process({
        ...     "task": "define_cohort",
        ...     "criteria": {"age": ">50", "diagnosis": "diabetes"}
        ... })
    """

    def __init__(self, **kwargs: Any):
        """Initialize the cohort agent."""
        kwargs.setdefault("role", AgentRole.COHORT_MANAGER)
        super().__init__(**kwargs)

        # Cohort storage
        self._cohorts: dict[str, Cohort] = {}

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process cohort management tasks."""
        task = input_data.get("task", "define_cohort")

        try:
            if task == "define_cohort":
                criteria = input_data.get("criteria", {})
                name = input_data.get("name", "")
                result = await self._define_cohort(criteria, name)

            elif task == "analyze_cohort":
                cohort_id = input_data.get("cohort_id", "")
                result = await self._analyze_cohort(cohort_id)

            elif task == "find_subgroups":
                cohort_id = input_data.get("cohort_id", "")
                features = input_data.get("features", [])
                result = await self._find_subgroups(cohort_id, features)

            elif task == "compare_cohorts":
                cohort_ids = input_data.get("cohort_ids", [])
                result = await self._compare_cohorts(cohort_ids)

            elif task == "select_patients":
                criteria = input_data.get("criteria", {})
                result = await self._select_patients(criteria)

            else:
                result = {"error": f"Unknown task: {task}"}

            self.tasks_completed += 1
            return result

        except Exception as e:
            self._logger.exception("Cohort processing error", error=str(e))
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
        return None

    async def _define_cohort(
        self, criteria: dict[str, Any], name: str
    ) -> dict[str, Any]:
        """Define a new patient cohort."""
        from uuid import uuid4

        cohort_id = str(uuid4())[:8]

        # Parse criteria into inclusion/exclusion
        inclusion = []
        exclusion = []

        for key, value in criteria.items():
            if isinstance(value, str) and value.startswith("!"):
                exclusion.append(f"{key}: {value[1:]}")
            else:
                inclusion.append(f"{key}: {value}")

        # Simulated cohort size based on criteria stringency
        base_size = 10000
        size_reduction = len(inclusion) * 0.2 + len(exclusion) * 0.1
        estimated_size = int(base_size * (1 - min(size_reduction, 0.9)))

        cohort = Cohort(
            cohort_id=cohort_id,
            name=name or f"Cohort_{cohort_id}",
            description=f"Cohort defined with {len(inclusion)} inclusion and {len(exclusion)} exclusion criteria",
            inclusion_criteria=inclusion,
            exclusion_criteria=exclusion,
            size=estimated_size,
            demographics=await self._generate_demographics(estimated_size),
        )

        self._cohorts[cohort_id] = cohort

        return {
            "cohort_id": cohort.cohort_id,
            "name": cohort.name,
            "estimated_size": cohort.size,
            "inclusion_criteria": cohort.inclusion_criteria,
            "exclusion_criteria": cohort.exclusion_criteria,
            "demographics": cohort.demographics,
        }

    async def _generate_demographics(self, size: int) -> dict[str, Any]:
        """Generate simulated demographics for a cohort."""
        return {
            "total_patients": size,
            "age": {
                "mean": 55.2,
                "std": 12.4,
                "min": 18,
                "max": 89,
            },
            "sex": {
                "male": int(size * 0.48),
                "female": int(size * 0.52),
            },
            "ethnicity": {
                "caucasian": int(size * 0.65),
                "african_american": int(size * 0.15),
                "asian": int(size * 0.12),
                "hispanic": int(size * 0.06),
                "other": int(size * 0.02),
            },
        }

    async def _analyze_cohort(self, cohort_id: str) -> dict[str, Any]:
        """Analyze a cohort's characteristics."""
        cohort = self._cohorts.get(cohort_id)
        if not cohort:
            return {"error": f"Cohort not found: {cohort_id}"}

        # Simulated clinical characteristics
        clinical_characteristics = {
            "disease_duration_years": {"mean": 5.2, "std": 3.1},
            "comorbidity_count": {"mean": 2.1, "std": 1.4},
            "prior_treatments": {"mean": 1.8, "std": 1.2},
            "biomarker_positive": {
                "percentage": 0.45,
                "count": int(cohort.size * 0.45),
            },
        }

        # Outcomes summary
        outcomes = {
            "mortality_rate": 0.08,
            "hospitalization_rate": 0.22,
            "treatment_response_rate": 0.65,
        }

        return {
            "cohort_id": cohort_id,
            "name": cohort.name,
            "size": cohort.size,
            "demographics": cohort.demographics,
            "clinical_characteristics": clinical_characteristics,
            "outcomes": outcomes,
        }

    async def _find_subgroups(
        self, cohort_id: str, features: list[str]
    ) -> dict[str, Any]:
        """Identify subgroups within a cohort."""
        cohort = self._cohorts.get(cohort_id)
        if not cohort:
            return {"error": f"Cohort not found: {cohort_id}"}

        if not features:
            features = ["age", "biomarker_status", "disease_severity"]

        # Simulated subgroup identification
        subgroups = [
            {
                "subgroup_id": "sg_1",
                "name": "Elderly with high biomarker",
                "criteria": {"age": ">65", "biomarker_status": "positive"},
                "size": int(cohort.size * 0.15),
                "response_rate": 0.72,
                "distinguishing_features": ["Higher treatment response", "More comorbidities"],
            },
            {
                "subgroup_id": "sg_2",
                "name": "Young with severe disease",
                "criteria": {"age": "<50", "disease_severity": "high"},
                "size": int(cohort.size * 0.12),
                "response_rate": 0.58,
                "distinguishing_features": ["Faster progression", "Better tolerability"],
            },
            {
                "subgroup_id": "sg_3",
                "name": "Biomarker negative",
                "criteria": {"biomarker_status": "negative"},
                "size": int(cohort.size * 0.55),
                "response_rate": 0.42,
                "distinguishing_features": ["Lower baseline response", "Need alternative approaches"],
            },
        ]

        # Deposit pheromone for interesting subgroups
        for sg in subgroups:
            if sg["response_rate"] > 0.6:
                await self.deposit_pheromone(
                    f"subgroup:{sg['subgroup_id']}", 2.0
                )

        return {
            "cohort_id": cohort_id,
            "features_analyzed": features,
            "subgroups_found": len(subgroups),
            "subgroups": subgroups,
        }

    async def _compare_cohorts(
        self, cohort_ids: list[str]
    ) -> dict[str, Any]:
        """Compare multiple cohorts."""
        if len(cohort_ids) < 2:
            return {"error": "Need at least 2 cohorts to compare"}

        cohorts = []
        for cid in cohort_ids:
            cohort = self._cohorts.get(cid)
            if cohort:
                cohorts.append(cohort)

        if len(cohorts) < 2:
            return {"error": "Not enough valid cohorts found"}

        # Comparison metrics
        comparison = {
            "cohorts_compared": [c.cohort_id for c in cohorts],
            "size_comparison": {c.cohort_id: c.size for c in cohorts},
            "demographic_comparison": {},
            "significant_differences": [],
        }

        # Compare demographics
        for key in ["age", "sex"]:
            comparison["demographic_comparison"][key] = {
                c.cohort_id: c.demographics.get(key, {})
                for c in cohorts
            }

        # Identify significant differences (simulated)
        if cohorts[0].size > cohorts[1].size * 1.5:
            comparison["significant_differences"].append(
                "Significant size difference between cohorts"
            )

        return comparison

    async def _select_patients(
        self, criteria: dict[str, Any]
    ) -> dict[str, Any]:
        """Select patients based on criteria."""
        # Simulated patient selection
        base_pool = 50000

        # Apply criteria
        remaining = base_pool
        applied_criteria = []

        for key, value in criteria.items():
            # Each criterion reduces pool
            reduction = 0.2 if isinstance(value, str) else 0.3
            remaining = int(remaining * (1 - reduction))
            applied_criteria.append(f"{key}: {value}")

        # Generate patient IDs
        selected_patients = [
            f"PT_{i:05d}" for i in range(min(100, remaining))
        ]

        return {
            "criteria_applied": applied_criteria,
            "patients_screened": base_pool,
            "patients_selected": remaining,
            "sample_patient_ids": selected_patients[:10],
            "selection_rate": remaining / base_pool,
        }
