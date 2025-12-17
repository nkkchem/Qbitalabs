"""
Digital Twin Engine for QBitaLabs

Core engine for biological digital twin simulation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TwinStatus(str, Enum):
    """Status of a digital twin."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    SIMULATING = "simulating"
    PAUSED = "paused"
    ARCHIVED = "archived"


@dataclass
class TwinState:
    """
    State of a digital twin at a point in time.

    Captures multi-scale physiological state.
    """

    timestamp: datetime = field(default_factory=datetime.utcnow)
    simulation_time: float = 0.0  # Days since initialization

    # Molecular level
    gene_expression: dict[str, float] = field(default_factory=dict)
    protein_levels: dict[str, float] = field(default_factory=dict)
    metabolite_levels: dict[str, float] = field(default_factory=dict)

    # Cellular level
    cell_counts: dict[str, int] = field(default_factory=dict)
    cell_states: dict[str, dict[str, float]] = field(default_factory=dict)

    # Organ/System level
    organ_function: dict[str, float] = field(default_factory=dict)
    biomarkers: dict[str, float] = field(default_factory=dict)

    # Whole body
    vital_signs: dict[str, float] = field(default_factory=dict)
    disease_states: dict[str, float] = field(default_factory=dict)

    # Interventions
    active_medications: list[dict[str, Any]] = field(default_factory=list)
    recent_interventions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PatientProfile:
    """Patient profile for digital twin initialization."""

    patient_id: str
    age: float  # Years
    sex: str
    ethnicity: str | None = None

    # Genomic data
    genetic_variants: dict[str, str] = field(default_factory=dict)
    gene_expression_profile: dict[str, float] = field(default_factory=dict)

    # Medical history
    conditions: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)

    # Lifestyle
    bmi: float | None = None
    smoking_status: str = "never"
    alcohol_frequency: str = "none"
    exercise_level: str = "moderate"

    # Lab values
    lab_results: dict[str, float] = field(default_factory=dict)


class PatientTwin:
    """
    Digital twin of an individual patient.

    Provides personalized simulation of:
    - Disease progression
    - Drug response
    - Treatment outcomes
    - Aging trajectory

    Example:
        >>> profile = PatientProfile(patient_id="P001", age=45, sex="M")
        >>> twin = PatientTwin(profile)
        >>> await twin.initialize()
        >>> prediction = await twin.simulate_treatment(drug="aspirin", duration=30)
    """

    def __init__(
        self,
        profile: PatientProfile,
        engine: "DigitalTwinEngine | None" = None,
    ):
        """
        Initialize patient digital twin.

        Args:
            profile: Patient profile.
            engine: Parent twin engine.
        """
        self.twin_id = str(uuid4())[:8]
        self.profile = profile
        self.engine = engine

        self.status = TwinStatus.INITIALIZING
        self._state = TwinState()
        self._state_history: list[TwinState] = []

        self._models: dict[str, Any] = {}
        self._callbacks: list[Callable[[TwinState], None]] = []

        self._logger = structlog.get_logger(f"twin.{self.twin_id}")

    async def initialize(self) -> None:
        """Initialize the digital twin from patient profile."""
        self._logger.info(
            "Initializing twin",
            patient_id=self.profile.patient_id,
            age=self.profile.age,
        )

        # Initialize state from profile
        await self._initialize_molecular_state()
        await self._initialize_cellular_state()
        await self._initialize_organ_state()
        await self._initialize_disease_state()

        self.status = TwinStatus.ACTIVE
        self._state_history.append(self._state)

        self._logger.info("Twin initialized", twin_id=self.twin_id)

    async def _initialize_molecular_state(self) -> None:
        """Initialize molecular level state."""
        # Base gene expression (normalized)
        base_genes = [
            "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS",
            "AKT1", "PTEN", "MYC", "RB1", "VEGFA",
        ]

        self._state.gene_expression = {
            gene: np.random.lognormal(0, 0.5)
            for gene in base_genes
        }

        # Incorporate patient variants
        for variant, effect in self.profile.genetic_variants.items():
            if variant in self._state.gene_expression:
                if effect == "gain":
                    self._state.gene_expression[variant] *= 1.5
                elif effect == "loss":
                    self._state.gene_expression[variant] *= 0.5

        # Protein levels (correlated with gene expression)
        self._state.protein_levels = {
            gene: expr * np.random.uniform(0.8, 1.2)
            for gene, expr in self._state.gene_expression.items()
        }

        # Key metabolites
        self._state.metabolite_levels = {
            "glucose": 90 + np.random.normal(0, 10),
            "lactate": 1.0 + np.random.normal(0, 0.2),
            "atp": 3.0 + np.random.normal(0, 0.3),
            "cholesterol": 180 + np.random.normal(0, 30),
            "creatinine": 1.0 + np.random.normal(0, 0.2),
        }

    async def _initialize_cellular_state(self) -> None:
        """Initialize cellular level state."""
        # Immune cells
        self._state.cell_counts = {
            "t_cells": int(1e9 + np.random.normal(0, 1e8)),
            "b_cells": int(5e8 + np.random.normal(0, 5e7)),
            "nk_cells": int(2e8 + np.random.normal(0, 2e7)),
            "neutrophils": int(4e9 + np.random.normal(0, 4e8)),
            "macrophages": int(1e8 + np.random.normal(0, 1e7)),
        }

        # Cell state indicators
        self._state.cell_states = {
            "t_cells": {"activation": 0.2, "exhaustion": 0.1},
            "stem_cells": {"proliferation": 0.1, "senescence": 0.05 * self.profile.age / 100},
        }

    async def _initialize_organ_state(self) -> None:
        """Initialize organ/system level state."""
        # Age-dependent organ function (normalized 0-1)
        age_factor = max(0, 1 - (self.profile.age - 30) * 0.005)

        self._state.organ_function = {
            "heart": min(1.0, age_factor + np.random.normal(0, 0.05)),
            "liver": min(1.0, age_factor + np.random.normal(0, 0.05)),
            "kidney": min(1.0, age_factor + np.random.normal(0, 0.05)),
            "lung": min(1.0, age_factor + np.random.normal(0, 0.05)),
            "brain": min(1.0, age_factor + np.random.normal(0, 0.05)),
            "immune": min(1.0, age_factor + np.random.normal(0, 0.05)),
        }

        # Vital signs
        self._state.vital_signs = {
            "heart_rate": 70 + np.random.normal(0, 10),
            "blood_pressure_sys": 120 + np.random.normal(0, 10),
            "blood_pressure_dia": 80 + np.random.normal(0, 8),
            "respiratory_rate": 14 + np.random.normal(0, 2),
            "temperature": 36.6 + np.random.normal(0, 0.3),
            "oxygen_saturation": 98 + np.random.normal(0, 1),
        }

        # Biomarkers from lab results
        self._state.biomarkers = {
            "crp": self.profile.lab_results.get("crp", 1.0),
            "hba1c": self.profile.lab_results.get("hba1c", 5.5),
            "ldl": self.profile.lab_results.get("ldl", 100),
            "hdl": self.profile.lab_results.get("hdl", 50),
            "triglycerides": self.profile.lab_results.get("triglycerides", 150),
        }

    async def _initialize_disease_state(self) -> None:
        """Initialize disease states from conditions."""
        self._state.disease_states = {
            "cancer_risk": 0.0,
            "cardiovascular_risk": 0.0,
            "diabetes_risk": 0.0,
            "alzheimers_risk": 0.0,
            "inflammation": 0.0,
        }

        # Compute baseline risks
        for condition in self.profile.conditions:
            if "diabetes" in condition.lower():
                self._state.disease_states["diabetes_risk"] = 0.5
                self._state.disease_states["cardiovascular_risk"] += 0.2
            if "hypertension" in condition.lower():
                self._state.disease_states["cardiovascular_risk"] += 0.3
            if "cancer" in condition.lower():
                self._state.disease_states["cancer_risk"] = 0.4

        # Age-based risk
        self._state.disease_states["cardiovascular_risk"] += self.profile.age * 0.005
        self._state.disease_states["cancer_risk"] += self.profile.age * 0.003
        self._state.disease_states["alzheimers_risk"] = max(0, (self.profile.age - 65) * 0.01)

        # Cap risks
        for key in self._state.disease_states:
            self._state.disease_states[key] = min(1.0, self._state.disease_states[key])

    async def simulate(
        self,
        duration: float,
        timestep: float = 1.0,
        interventions: list[dict[str, Any]] | None = None,
    ) -> list[TwinState]:
        """
        Run forward simulation.

        Args:
            duration: Simulation duration in days.
            timestep: Timestep in days.
            interventions: List of interventions to apply.

        Returns:
            List of states over time.
        """
        self.status = TwinStatus.SIMULATING
        states = []
        interventions = interventions or []

        n_steps = int(duration / timestep)

        for step in range(n_steps):
            current_time = step * timestep

            # Apply scheduled interventions
            for intervention in interventions:
                if intervention["start_time"] <= current_time < intervention.get("end_time", float("inf")):
                    await self._apply_intervention(intervention)

            # Update state
            await self._step(timestep)

            # Record state
            new_state = self._copy_state()
            new_state.simulation_time = current_time + timestep
            states.append(new_state)
            self._state_history.append(new_state)

            # Callbacks
            for callback in self._callbacks:
                callback(new_state)

        self.status = TwinStatus.ACTIVE
        return states

    async def _step(self, dt: float) -> None:
        """Advance simulation by one timestep."""
        # Update molecular state
        await self._update_molecular(dt)

        # Update cellular state
        await self._update_cellular(dt)

        # Update organ function
        await self._update_organs(dt)

        # Update disease progression
        await self._update_diseases(dt)

        # Update vital signs
        await self._update_vitals(dt)

    async def _update_molecular(self, dt: float) -> None:
        """Update molecular level dynamics."""
        # Gene expression changes (stochastic)
        for gene in self._state.gene_expression:
            noise = np.random.normal(0, 0.01) * dt
            self._state.gene_expression[gene] *= (1 + noise)
            self._state.gene_expression[gene] = max(0.01, self._state.gene_expression[gene])

        # Protein turnover
        for protein in self._state.protein_levels:
            if protein in self._state.gene_expression:
                target = self._state.gene_expression[protein]
                current = self._state.protein_levels[protein]
                self._state.protein_levels[protein] += (target - current) * 0.1 * dt

        # Metabolite dynamics
        glucose = self._state.metabolite_levels["glucose"]
        # Simple glucose regulation
        target_glucose = 90
        self._state.metabolite_levels["glucose"] += (target_glucose - glucose) * 0.05 * dt

    async def _update_cellular(self, dt: float) -> None:
        """Update cellular level dynamics."""
        # Cell turnover
        for cell_type in self._state.cell_counts:
            turnover_rate = 0.01  # 1% per day
            change = int(self._state.cell_counts[cell_type] * turnover_rate * dt * np.random.normal(0, 0.1))
            self._state.cell_counts[cell_type] = max(0, self._state.cell_counts[cell_type] + change)

        # Senescence accumulation with age
        if "stem_cells" in self._state.cell_states:
            senescence_rate = 0.0001 * dt  # Per day
            self._state.cell_states["stem_cells"]["senescence"] += senescence_rate
            self._state.cell_states["stem_cells"]["senescence"] = min(1.0, self._state.cell_states["stem_cells"]["senescence"])

    async def _update_organs(self, dt: float) -> None:
        """Update organ function."""
        # Gradual decline with age
        aging_rate = 0.00005 * dt  # Per day

        for organ in self._state.organ_function:
            # Natural decline
            self._state.organ_function[organ] -= aging_rate
            self._state.organ_function[organ] = max(0.1, self._state.organ_function[organ])

        # Inter-organ effects
        if self._state.organ_function["kidney"] < 0.5:
            self._state.organ_function["heart"] -= 0.001 * dt

    async def _update_diseases(self, dt: float) -> None:
        """Update disease states."""
        # Risk accumulation
        inflammation = self._state.biomarkers.get("crp", 1.0) / 10

        self._state.disease_states["inflammation"] = inflammation
        self._state.disease_states["cardiovascular_risk"] += inflammation * 0.0001 * dt

        # Clamp values
        for disease in self._state.disease_states:
            self._state.disease_states[disease] = np.clip(
                self._state.disease_states[disease], 0, 1
            )

    async def _update_vitals(self, dt: float) -> None:
        """Update vital signs."""
        # Vital signs fluctuate around baseline
        baseline = {
            "heart_rate": 70,
            "blood_pressure_sys": 120,
            "blood_pressure_dia": 80,
            "respiratory_rate": 14,
            "temperature": 36.6,
            "oxygen_saturation": 98,
        }

        for vital, base in baseline.items():
            current = self._state.vital_signs[vital]
            # Mean reversion with noise
            self._state.vital_signs[vital] = (
                current + (base - current) * 0.1 * dt + np.random.normal(0, 1) * dt
            )

    async def _apply_intervention(self, intervention: dict[str, Any]) -> None:
        """Apply an intervention to the twin."""
        intervention_type = intervention.get("type", "medication")

        if intervention_type == "medication":
            await self._apply_medication(intervention)
        elif intervention_type == "lifestyle":
            await self._apply_lifestyle(intervention)
        elif intervention_type == "procedure":
            await self._apply_procedure(intervention)

        self._state.recent_interventions.append({
            **intervention,
            "applied_at": self._state.simulation_time,
        })

    async def _apply_medication(self, medication: dict[str, Any]) -> None:
        """Apply medication effects."""
        drug = medication.get("drug", "")
        dose = medication.get("dose", 1.0)

        # Drug-specific effects (simplified)
        if drug.lower() == "aspirin":
            self._state.biomarkers["crp"] *= (1 - 0.1 * dose)
            self._state.disease_states["cardiovascular_risk"] -= 0.05 * dose
        elif drug.lower() == "statin":
            self._state.biomarkers["ldl"] *= (1 - 0.3 * dose)
            self._state.disease_states["cardiovascular_risk"] -= 0.1 * dose
        elif drug.lower() == "metformin":
            self._state.metabolite_levels["glucose"] *= (1 - 0.15 * dose)
            self._state.disease_states["diabetes_risk"] -= 0.1 * dose

        self._state.active_medications.append(medication)

    async def _apply_lifestyle(self, lifestyle: dict[str, Any]) -> None:
        """Apply lifestyle intervention effects."""
        intervention = lifestyle.get("intervention", "")

        if intervention == "exercise":
            intensity = lifestyle.get("intensity", "moderate")
            multiplier = {"low": 0.5, "moderate": 1.0, "high": 1.5}.get(intensity, 1.0)

            self._state.organ_function["heart"] += 0.01 * multiplier
            self._state.disease_states["cardiovascular_risk"] -= 0.02 * multiplier

        elif intervention == "diet":
            diet_type = lifestyle.get("diet_type", "balanced")
            if diet_type == "low_carb":
                self._state.metabolite_levels["glucose"] -= 10
            elif diet_type == "mediterranean":
                self._state.biomarkers["crp"] *= 0.9

    async def _apply_procedure(self, procedure: dict[str, Any]) -> None:
        """Apply medical procedure effects."""
        procedure_type = procedure.get("procedure_type", "")

        if procedure_type == "surgery":
            target_organ = procedure.get("target_organ", "")
            if target_organ in self._state.organ_function:
                # Surgery temporarily affects function then improves
                self._state.organ_function[target_organ] *= 0.9

    async def simulate_treatment(
        self,
        drug: str,
        dose: float = 1.0,
        duration: float = 30,
        timestep: float = 1.0,
    ) -> dict[str, Any]:
        """
        Simulate treatment with a specific drug.

        Args:
            drug: Drug name.
            dose: Relative dose (1.0 = standard).
            duration: Treatment duration in days.
            timestep: Simulation timestep.

        Returns:
            Treatment outcome prediction.
        """
        # Store initial state
        initial_state = self._copy_state()

        # Create intervention
        intervention = {
            "type": "medication",
            "drug": drug,
            "dose": dose,
            "start_time": 0,
            "end_time": duration,
        }

        # Run simulation
        states = await self.simulate(duration, timestep, [intervention])

        # Analyze outcomes
        final_state = states[-1] if states else self._state

        return {
            "drug": drug,
            "duration": duration,
            "initial_disease_risk": initial_state.disease_states.copy(),
            "final_disease_risk": final_state.disease_states.copy(),
            "biomarker_changes": {
                k: final_state.biomarkers.get(k, 0) - initial_state.biomarkers.get(k, 0)
                for k in initial_state.biomarkers
            },
            "side_effects": self._predict_side_effects(drug, dose),
            "efficacy_score": self._compute_efficacy(initial_state, final_state),
        }

    def _predict_side_effects(self, drug: str, dose: float) -> list[dict[str, Any]]:
        """Predict drug side effects based on patient profile."""
        side_effects = []

        # Drug-specific side effects (simplified)
        if drug.lower() == "aspirin":
            gi_risk = 0.1 * dose
            if self.profile.age > 65:
                gi_risk *= 1.5
            side_effects.append({
                "effect": "gi_bleeding",
                "probability": gi_risk,
                "severity": "moderate",
            })

        return side_effects

    def _compute_efficacy(
        self,
        initial: TwinState,
        final: TwinState,
    ) -> float:
        """Compute treatment efficacy score."""
        # Compare disease risk reduction
        initial_risk = sum(initial.disease_states.values())
        final_risk = sum(final.disease_states.values())

        if initial_risk > 0:
            return max(0, (initial_risk - final_risk) / initial_risk)
        return 0.0

    def _copy_state(self) -> TwinState:
        """Create a copy of current state."""
        return TwinState(
            timestamp=datetime.utcnow(),
            simulation_time=self._state.simulation_time,
            gene_expression=self._state.gene_expression.copy(),
            protein_levels=self._state.protein_levels.copy(),
            metabolite_levels=self._state.metabolite_levels.copy(),
            cell_counts=self._state.cell_counts.copy(),
            cell_states={k: v.copy() for k, v in self._state.cell_states.items()},
            organ_function=self._state.organ_function.copy(),
            biomarkers=self._state.biomarkers.copy(),
            vital_signs=self._state.vital_signs.copy(),
            disease_states=self._state.disease_states.copy(),
            active_medications=self._state.active_medications.copy(),
            recent_interventions=self._state.recent_interventions.copy(),
        )

    @property
    def state(self) -> TwinState:
        """Get current state."""
        return self._state

    def get_state_history(self) -> list[TwinState]:
        """Get state history."""
        return self._state_history.copy()

    def add_callback(self, callback: Callable[[TwinState], None]) -> None:
        """Add state change callback."""
        self._callbacks.append(callback)


class DigitalTwinEngine:
    """
    Central engine for managing digital twins.

    Coordinates:
    - Twin creation and lifecycle
    - Population-level simulations
    - Model calibration
    - Data integration

    Example:
        >>> engine = DigitalTwinEngine()
        >>> await engine.initialize()
        >>> twin = await engine.create_twin(patient_profile)
        >>> predictions = await engine.run_virtual_trial(twins, drug)
    """

    def __init__(self):
        """Initialize the digital twin engine."""
        self._twins: dict[str, PatientTwin] = {}
        self._models: dict[str, Any] = {}

        self._logger = structlog.get_logger("twin_engine")

    async def initialize(self) -> None:
        """Initialize the engine."""
        self._logger.info("Digital Twin Engine initializing")
        # Load default models
        self._logger.info("Digital Twin Engine ready")

    async def create_twin(self, profile: PatientProfile) -> PatientTwin:
        """
        Create a new digital twin.

        Args:
            profile: Patient profile.

        Returns:
            Initialized patient twin.
        """
        twin = PatientTwin(profile, engine=self)
        await twin.initialize()

        self._twins[twin.twin_id] = twin

        self._logger.info(
            "Twin created",
            twin_id=twin.twin_id,
            patient_id=profile.patient_id,
        )

        return twin

    def get_twin(self, twin_id: str) -> PatientTwin | None:
        """Get a twin by ID."""
        return self._twins.get(twin_id)

    def list_twins(self) -> list[PatientTwin]:
        """List all active twins."""
        return list(self._twins.values())

    async def run_virtual_trial(
        self,
        twins: list[PatientTwin],
        intervention: dict[str, Any],
        duration: float = 90,
    ) -> dict[str, Any]:
        """
        Run a virtual clinical trial.

        Args:
            twins: List of patient twins.
            intervention: Treatment intervention.
            duration: Trial duration in days.

        Returns:
            Trial results.
        """
        self._logger.info(
            "Starting virtual trial",
            n_patients=len(twins),
            intervention=intervention.get("drug", "unknown"),
        )

        results = []

        for twin in twins:
            outcome = await twin.simulate_treatment(
                drug=intervention.get("drug", ""),
                dose=intervention.get("dose", 1.0),
                duration=duration,
            )
            results.append({
                "twin_id": twin.twin_id,
                "patient_id": twin.profile.patient_id,
                "outcome": outcome,
            })

        # Aggregate results
        efficacies = [r["outcome"]["efficacy_score"] for r in results]
        responders = sum(1 for e in efficacies if e > 0.1)

        return {
            "n_patients": len(twins),
            "intervention": intervention,
            "duration_days": duration,
            "response_rate": responders / len(twins) if twins else 0,
            "mean_efficacy": np.mean(efficacies) if efficacies else 0,
            "std_efficacy": np.std(efficacies) if efficacies else 0,
            "individual_results": results,
        }

    async def predict_population_trajectory(
        self,
        twins: list[PatientTwin],
        years: float = 10,
    ) -> dict[str, Any]:
        """
        Predict population health trajectory.

        Args:
            twins: List of patient twins.
            years: Prediction horizon.

        Returns:
            Population trajectory predictions.
        """
        duration = years * 365  # Convert to days

        trajectories = []
        for twin in twins:
            states = await twin.simulate(duration, timestep=30)  # Monthly
            trajectories.append({
                "twin_id": twin.twin_id,
                "cardiovascular_trajectory": [
                    s.disease_states.get("cardiovascular_risk", 0)
                    for s in states
                ],
                "cancer_trajectory": [
                    s.disease_states.get("cancer_risk", 0)
                    for s in states
                ],
            })

        return {
            "n_patients": len(twins),
            "years": years,
            "trajectories": trajectories,
        }
