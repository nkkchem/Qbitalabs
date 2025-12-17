"""
Trial Design Agent for QBitaLabs SWARM

Specializes in clinical trial design:
- Protocol design
- Sample size calculation
- Endpoint selection
- Randomization strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from qbitalabs.core.types import AgentRole, MessageType
from qbitalabs.swarm.base_agent import AgentMessage, BaseAgent

logger = structlog.get_logger(__name__)


@dataclass
class TrialDesign:
    """Clinical trial design specification."""

    trial_id: str = ""
    phase: str = ""  # Phase 1, 2, 3, 4
    design_type: str = ""  # parallel, crossover, factorial
    primary_endpoint: str = ""
    secondary_endpoints: list[str] = field(default_factory=list)
    sample_size: int = 0
    duration_weeks: int = 0
    arms: list[dict[str, Any]] = field(default_factory=list)
    randomization: str = ""
    blinding: str = ""


class TrialDesignAgent(BaseAgent):
    """
    Agent specializing in clinical trial design.

    Capabilities:
    - Design trial protocols
    - Calculate sample sizes
    - Select appropriate endpoints
    - Optimize randomization
    - Generate statistical analysis plans

    Example:
        >>> agent = TrialDesignAgent()
        >>> result = await agent.process({
        ...     "task": "design_trial",
        ...     "phase": "2",
        ...     "indication": "breast cancer"
        ... })
    """

    def __init__(self, **kwargs: Any):
        """Initialize the trial design agent."""
        kwargs.setdefault("role", AgentRole.TRIAL_DESIGNER)
        super().__init__(**kwargs)

    async def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process trial design tasks."""
        task = input_data.get("task", "design_trial")

        try:
            if task == "design_trial":
                phase = input_data.get("phase", "2")
                indication = input_data.get("indication", "")
                result = await self._design_trial(phase, indication)

            elif task == "calculate_sample_size":
                params = input_data.get("params", {})
                result = await self._calculate_sample_size(params)

            elif task == "select_endpoints":
                indication = input_data.get("indication", "")
                phase = input_data.get("phase", "2")
                result = await self._select_endpoints(indication, phase)

            elif task == "optimize_randomization":
                design = input_data.get("design", {})
                result = await self._optimize_randomization(design)

            elif task == "generate_sap":
                trial_design = input_data.get("trial_design", {})
                result = await self._generate_statistical_analysis_plan(trial_design)

            else:
                result = {"error": f"Unknown task: {task}"}

            self.tasks_completed += 1
            return result

        except Exception as e:
            self._logger.exception("Trial design error", error=str(e))
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

    async def _design_trial(
        self, phase: str, indication: str
    ) -> dict[str, Any]:
        """Design a clinical trial."""
        from uuid import uuid4

        # Select endpoints based on phase and indication
        endpoints = await self._select_endpoints(indication, phase)

        # Calculate sample size
        sample_params = {
            "effect_size": 0.3,
            "alpha": 0.05,
            "power": 0.8,
            "design": "parallel",
        }
        sample_size_result = await self._calculate_sample_size(sample_params)

        # Phase-specific design elements
        phase_configs = {
            "1": {
                "design_type": "dose_escalation",
                "duration_weeks": 12,
                "arms": [
                    {"name": "Low dose", "n": 10},
                    {"name": "Medium dose", "n": 10},
                    {"name": "High dose", "n": 10},
                ],
                "blinding": "open_label",
            },
            "2": {
                "design_type": "parallel",
                "duration_weeks": 24,
                "arms": [
                    {"name": "Treatment", "n": sample_size_result["sample_size_per_arm"]},
                    {"name": "Placebo", "n": sample_size_result["sample_size_per_arm"]},
                ],
                "blinding": "double_blind",
            },
            "3": {
                "design_type": "parallel",
                "duration_weeks": 52,
                "arms": [
                    {"name": "Treatment", "n": sample_size_result["sample_size_per_arm"] * 2},
                    {"name": "Control", "n": sample_size_result["sample_size_per_arm"] * 2},
                ],
                "blinding": "double_blind",
            },
        }

        config = phase_configs.get(phase, phase_configs["2"])

        trial_design = TrialDesign(
            trial_id=str(uuid4())[:8],
            phase=f"Phase {phase}",
            design_type=config["design_type"],
            primary_endpoint=endpoints.get("primary", ""),
            secondary_endpoints=endpoints.get("secondary", []),
            sample_size=sample_size_result["total_sample_size"],
            duration_weeks=config["duration_weeks"],
            arms=config["arms"],
            randomization="stratified_block",
            blinding=config["blinding"],
        )

        return {
            "trial_id": trial_design.trial_id,
            "phase": trial_design.phase,
            "indication": indication,
            "design_type": trial_design.design_type,
            "primary_endpoint": trial_design.primary_endpoint,
            "secondary_endpoints": trial_design.secondary_endpoints,
            "sample_size": trial_design.sample_size,
            "duration_weeks": trial_design.duration_weeks,
            "arms": trial_design.arms,
            "randomization": trial_design.randomization,
            "blinding": trial_design.blinding,
        }

    async def _calculate_sample_size(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate required sample size."""
        effect_size = params.get("effect_size", 0.3)
        alpha = params.get("alpha", 0.05)
        power = params.get("power", 0.8)
        design = params.get("design", "parallel")

        # Simplified sample size calculation
        # For two-group comparison
        # n = 2 * ((z_alpha + z_beta) / effect_size)^2

        # Z-scores for common values
        z_alpha = 1.96 if alpha == 0.05 else 2.58  # two-sided
        z_beta = 0.84 if power == 0.8 else 1.28

        n_per_arm = int(2 * ((z_alpha + z_beta) / effect_size) ** 2)

        # Adjust for design
        if design == "crossover":
            n_per_arm = int(n_per_arm * 0.5)  # Crossover typically needs fewer subjects
        elif design == "factorial":
            n_per_arm = int(n_per_arm * 0.75)

        # Add dropout adjustment (15%)
        n_adjusted = int(n_per_arm / 0.85)

        return {
            "sample_size_per_arm": n_adjusted,
            "total_sample_size": n_adjusted * 2,
            "parameters": {
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power,
                "design": design,
            },
            "dropout_rate_assumed": 0.15,
        }

    async def _select_endpoints(
        self, indication: str, phase: str
    ) -> dict[str, Any]:
        """Select appropriate endpoints for trial."""
        # Indication-specific endpoints
        endpoint_library = {
            "cancer": {
                "primary": "Overall Survival (OS)",
                "secondary": [
                    "Progression-Free Survival (PFS)",
                    "Objective Response Rate (ORR)",
                    "Duration of Response (DOR)",
                    "Quality of Life (QoL)",
                ],
            },
            "cardiovascular": {
                "primary": "Major Adverse Cardiovascular Events (MACE)",
                "secondary": [
                    "All-cause mortality",
                    "Hospitalization for heart failure",
                    "Blood pressure reduction",
                ],
            },
            "diabetes": {
                "primary": "HbA1c change from baseline",
                "secondary": [
                    "Fasting plasma glucose",
                    "Body weight change",
                    "Hypoglycemic events",
                ],
            },
            "default": {
                "primary": "Efficacy measure",
                "secondary": ["Safety", "Tolerability", "Quality of Life"],
            },
        }

        # Find matching indication
        indication_lower = indication.lower()
        endpoints = endpoint_library.get("default")

        for key, value in endpoint_library.items():
            if key in indication_lower:
                endpoints = value
                break

        # Phase 1 typically focuses on safety
        if phase == "1":
            endpoints = {
                "primary": "Maximum Tolerated Dose (MTD)",
                "secondary": ["Safety", "Pharmacokinetics", "Pharmacodynamics"],
            }

        return endpoints

    async def _optimize_randomization(
        self, design: dict[str, Any]
    ) -> dict[str, Any]:
        """Optimize randomization strategy."""
        sample_size = design.get("sample_size", 100)
        arms = design.get("arms", 2)
        stratification_factors = design.get("stratification_factors", [])

        # Recommend randomization method
        if sample_size < 50:
            method = "simple"
            block_size = None
        elif len(stratification_factors) > 3:
            method = "minimization"
            block_size = None
        else:
            method = "stratified_block"
            block_size = arms * 2

        return {
            "recommended_method": method,
            "block_size": block_size,
            "stratification_factors": stratification_factors or ["site", "disease_stage"],
            "allocation_ratio": "1:1" if arms == 2 else f"1:" + ":1" * (arms - 1),
            "rationale": f"Based on sample size ({sample_size}) and stratification needs",
        }

    async def _generate_statistical_analysis_plan(
        self, trial_design: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate statistical analysis plan outline."""
        primary_endpoint = trial_design.get("primary_endpoint", "")
        design_type = trial_design.get("design_type", "parallel")

        # Select analysis methods based on endpoint type
        if "survival" in primary_endpoint.lower():
            primary_analysis = "Kaplan-Meier estimation with log-rank test"
            model = "Cox proportional hazards model"
        elif "response" in primary_endpoint.lower() or "rate" in primary_endpoint.lower():
            primary_analysis = "Chi-square or Fisher's exact test"
            model = "Logistic regression"
        else:
            primary_analysis = "ANCOVA with baseline as covariate"
            model = "Mixed model for repeated measures (MMRM)"

        return {
            "primary_analysis": {
                "endpoint": primary_endpoint,
                "method": primary_analysis,
                "model": model,
                "significance_level": 0.05,
                "hypothesis": "two-sided",
            },
            "populations": {
                "ITT": "All randomized patients",
                "PP": "Patients completing protocol without major violations",
                "Safety": "All patients receiving at least one dose",
            },
            "missing_data": {
                "primary_approach": "Multiple imputation",
                "sensitivity_analyses": [
                    "Complete case analysis",
                    "Last observation carried forward",
                    "Worst case imputation",
                ],
            },
            "interim_analyses": trial_design.get("interim_analyses", 1),
        }
