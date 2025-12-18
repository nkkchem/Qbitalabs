"""
Physiological Models for Digital Twins

Multi-scale models for biological simulation:
- Metabolism and energy homeostasis
- Immune system dynamics
- Cardiovascular system
- Gene regulatory networks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PhysiologicalModel(ABC):
    """Base class for physiological models."""

    def __init__(self, name: str):
        """Initialize model."""
        self.name = name
        self._parameters: dict[str, float] = {}
        self._state: dict[str, float] = {}
        self._logger = structlog.get_logger(f"model.{name}")

    @abstractmethod
    def initialize(self, patient_params: dict[str, Any]) -> None:
        """Initialize model from patient parameters."""
        pass

    @abstractmethod
    def step(self, dt: float, inputs: dict[str, float]) -> dict[str, float]:
        """
        Advance model by one timestep.

        Args:
            dt: Timestep.
            inputs: External inputs.

        Returns:
            Model outputs.
        """
        pass

    def get_state(self) -> dict[str, float]:
        """Get current model state."""
        return self._state.copy()

    def set_parameter(self, name: str, value: float) -> None:
        """Set a model parameter."""
        self._parameters[name] = value


class MetabolismModel(PhysiologicalModel):
    """
    Metabolism and energy homeostasis model.

    Models:
    - Glucose-insulin dynamics
    - Lipid metabolism
    - Energy balance
    """

    def __init__(self):
        """Initialize metabolism model."""
        super().__init__("metabolism")

        # Default parameters
        self._parameters = {
            "insulin_sensitivity": 1.0,
            "glucose_production_rate": 2.0,  # mg/kg/min
            "glucose_clearance_rate": 0.05,
            "lipid_synthesis_rate": 0.1,
            "basal_metabolic_rate": 1500,  # kcal/day
        }

        # State variables
        self._state = {
            "glucose": 90,  # mg/dL
            "insulin": 10,  # μU/mL
            "free_fatty_acids": 0.5,  # mM
            "triglycerides": 150,  # mg/dL
            "energy_balance": 0,  # kcal
        }

    def initialize(self, patient_params: dict[str, Any]) -> None:
        """Initialize from patient parameters."""
        if "bmi" in patient_params:
            bmi = patient_params["bmi"]
            # Insulin resistance increases with BMI
            self._parameters["insulin_sensitivity"] = max(0.5, 2.0 - bmi / 30)

        if "diabetes" in patient_params.get("conditions", []):
            self._parameters["insulin_sensitivity"] *= 0.5

        if "glucose" in patient_params.get("lab_results", {}):
            self._state["glucose"] = patient_params["lab_results"]["glucose"]

    def step(self, dt: float, inputs: dict[str, float]) -> dict[str, float]:
        """Advance metabolism simulation."""
        # Glucose dynamics (simplified Bergman minimal model)
        glucose = self._state["glucose"]
        insulin = self._state["insulin"]

        glucose_input = inputs.get("glucose_intake", 0)
        exercise = inputs.get("exercise_intensity", 0)

        # Glucose production and clearance
        glucose_production = self._parameters["glucose_production_rate"] * dt
        glucose_clearance = (
            self._parameters["glucose_clearance_rate"]
            * glucose
            * insulin
            * self._parameters["insulin_sensitivity"]
            * dt
        )

        # Exercise increases glucose uptake
        exercise_uptake = exercise * glucose * 0.01 * dt

        self._state["glucose"] += glucose_production + glucose_input - glucose_clearance - exercise_uptake
        self._state["glucose"] = max(50, min(400, self._state["glucose"]))

        # Insulin response
        insulin_secretion = max(0, (glucose - 100) * 0.1)
        insulin_clearance = insulin * 0.1 * dt
        self._state["insulin"] += (insulin_secretion - insulin_clearance) * dt
        self._state["insulin"] = max(2, min(100, self._state["insulin"]))

        # Lipid dynamics
        ffa = self._state["free_fatty_acids"]
        triglycerides = self._state["triglycerides"]

        # Insulin suppresses lipolysis
        lipolysis = 0.1 * (1 / (1 + insulin / 10)) * dt
        lipogenesis = self._parameters["lipid_synthesis_rate"] * insulin / 10 * dt

        self._state["free_fatty_acids"] += lipolysis - lipogenesis * 0.1
        self._state["free_fatty_acids"] = max(0.1, min(2.0, self._state["free_fatty_acids"]))

        # Energy balance
        energy_intake = inputs.get("calorie_intake", self._parameters["basal_metabolic_rate"])
        energy_expenditure = self._parameters["basal_metabolic_rate"] * (1 + exercise * 0.5)
        self._state["energy_balance"] += (energy_intake - energy_expenditure) * dt

        return {
            "glucose": self._state["glucose"],
            "insulin": self._state["insulin"],
            "metabolic_health": self._compute_metabolic_health(),
        }

    def _compute_metabolic_health(self) -> float:
        """Compute metabolic health score."""
        score = 1.0

        # Penalize abnormal glucose
        if self._state["glucose"] > 126:
            score -= 0.3
        elif self._state["glucose"] > 100:
            score -= 0.1

        # Penalize high triglycerides
        if self._state["triglycerides"] > 200:
            score -= 0.2

        return max(0, score)


class ImmuneSystemModel(PhysiologicalModel):
    """
    Immune system dynamics model.

    Models:
    - Innate immune response
    - Adaptive immunity (T/B cells)
    - Inflammatory response
    """

    def __init__(self):
        """Initialize immune model."""
        super().__init__("immune")

        self._parameters = {
            "t_cell_activation_rate": 0.1,
            "t_cell_death_rate": 0.01,
            "b_cell_activation_rate": 0.05,
            "antibody_production_rate": 0.1,
            "inflammation_decay_rate": 0.1,
        }

        self._state = {
            "naive_t_cells": 1e9,
            "activated_t_cells": 1e6,
            "memory_t_cells": 1e8,
            "naive_b_cells": 5e8,
            "plasma_cells": 1e6,
            "antibody_level": 100,
            "inflammation": 0.1,
            "cytokines": {"il6": 1.0, "tnf_alpha": 1.0, "il10": 1.0},
        }

    def initialize(self, patient_params: dict[str, Any]) -> None:
        """Initialize from patient parameters."""
        age = patient_params.get("age", 50)

        # Immune aging (immunosenescence)
        if age > 60:
            self._state["naive_t_cells"] *= max(0.5, 1 - (age - 60) * 0.01)
            self._parameters["t_cell_activation_rate"] *= 0.8

        if "autoimmune" in str(patient_params.get("conditions", [])).lower():
            self._state["inflammation"] = 0.5

    def step(self, dt: float, inputs: dict[str, float]) -> dict[str, float]:
        """Advance immune simulation."""
        pathogen_load = inputs.get("pathogen_load", 0)
        stress_level = inputs.get("stress", 0)

        # T cell dynamics
        activation = (
            self._parameters["t_cell_activation_rate"]
            * self._state["naive_t_cells"]
            * pathogen_load
            * dt
        )
        death = self._parameters["t_cell_death_rate"] * self._state["activated_t_cells"] * dt

        self._state["naive_t_cells"] -= activation
        self._state["activated_t_cells"] += activation - death

        # Memory formation
        memory_formation = 0.01 * self._state["activated_t_cells"] * dt
        self._state["memory_t_cells"] += memory_formation

        # B cell and antibody dynamics
        b_activation = (
            self._parameters["b_cell_activation_rate"]
            * self._state["naive_b_cells"]
            * pathogen_load
            * dt
        )
        self._state["naive_b_cells"] -= b_activation
        self._state["plasma_cells"] += b_activation

        antibody_production = (
            self._parameters["antibody_production_rate"]
            * self._state["plasma_cells"]
            * dt
        )
        self._state["antibody_level"] += antibody_production - self._state["antibody_level"] * 0.01 * dt

        # Inflammation dynamics
        inflammation_increase = pathogen_load * 0.1 + stress_level * 0.05
        inflammation_decrease = self._parameters["inflammation_decay_rate"] * self._state["inflammation"]
        self._state["inflammation"] += (inflammation_increase - inflammation_decrease) * dt
        self._state["inflammation"] = max(0, min(1, self._state["inflammation"]))

        # Cytokine dynamics
        self._state["cytokines"]["il6"] = 1 + 5 * self._state["inflammation"]
        self._state["cytokines"]["tnf_alpha"] = 1 + 3 * self._state["inflammation"]
        self._state["cytokines"]["il10"] = 1 + 2 * self._state["activated_t_cells"] / 1e6

        return {
            "immune_function": self._compute_immune_function(),
            "inflammation": self._state["inflammation"],
            "antibody_level": self._state["antibody_level"],
        }

    def _compute_immune_function(self) -> float:
        """Compute immune function score."""
        t_cell_score = min(1, self._state["activated_t_cells"] / 1e7)
        antibody_score = min(1, self._state["antibody_level"] / 200)
        inflammation_penalty = max(0, self._state["inflammation"] - 0.3)

        return (t_cell_score + antibody_score) / 2 - inflammation_penalty


class CardiovascularModel(PhysiologicalModel):
    """
    Cardiovascular system model.

    Models:
    - Cardiac function
    - Blood pressure regulation
    - Vascular health
    """

    def __init__(self):
        """Initialize cardiovascular model."""
        super().__init__("cardiovascular")

        self._parameters = {
            "cardiac_output": 5.0,  # L/min
            "systemic_resistance": 1.0,
            "vessel_compliance": 1.0,
            "heart_rate_baseline": 70,
        }

        self._state = {
            "heart_rate": 70,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "stroke_volume": 70,  # mL
            "ejection_fraction": 0.6,
            "arterial_stiffness": 0.1,
            "atherosclerosis": 0.0,
        }

    def initialize(self, patient_params: dict[str, Any]) -> None:
        """Initialize from patient parameters."""
        age = patient_params.get("age", 50)

        # Age-related changes
        self._state["arterial_stiffness"] = 0.1 + age * 0.005
        self._parameters["vessel_compliance"] = max(0.5, 1 - age * 0.005)

        if "hypertension" in patient_params.get("conditions", []):
            self._state["systolic_bp"] += 20
            self._state["diastolic_bp"] += 10

        if "heart_failure" in patient_params.get("conditions", []):
            self._state["ejection_fraction"] = 0.35

    def step(self, dt: float, inputs: dict[str, float]) -> dict[str, float]:
        """Advance cardiovascular simulation."""
        exercise = inputs.get("exercise_intensity", 0)
        stress = inputs.get("stress", 0)
        medication = inputs.get("antihypertensive", 0)

        # Heart rate response
        hr_target = self._parameters["heart_rate_baseline"] * (1 + exercise + 0.2 * stress)
        self._state["heart_rate"] += (hr_target - self._state["heart_rate"]) * 0.1 * dt
        self._state["heart_rate"] = max(50, min(200, self._state["heart_rate"]))

        # Cardiac output
        co = (
            self._state["heart_rate"]
            * self._state["stroke_volume"]
            / 1000
        )

        # Blood pressure (simplified Windkessel)
        resistance = self._parameters["systemic_resistance"] * (1 + self._state["arterial_stiffness"])
        compliance = self._parameters["vessel_compliance"]

        map_pressure = co * resistance * 20
        pulse_pressure = map_pressure * 0.4 / compliance

        self._state["systolic_bp"] = map_pressure + pulse_pressure / 2
        self._state["diastolic_bp"] = map_pressure - pulse_pressure / 2

        # Medication effects
        if medication > 0:
            self._state["systolic_bp"] *= (1 - 0.1 * medication)
            self._state["diastolic_bp"] *= (1 - 0.1 * medication)

        # Atherosclerosis progression
        ldl = inputs.get("ldl", 100)
        if ldl > 130:
            self._state["atherosclerosis"] += 0.0001 * (ldl - 130) * dt

        return {
            "heart_rate": self._state["heart_rate"],
            "blood_pressure": f"{int(self._state['systolic_bp'])}/{int(self._state['diastolic_bp'])}",
            "cardiac_health": self._compute_cardiac_health(),
        }

    def _compute_cardiac_health(self) -> float:
        """Compute cardiac health score."""
        score = 1.0

        # Blood pressure penalty
        if self._state["systolic_bp"] > 140:
            score -= 0.2
        if self._state["diastolic_bp"] > 90:
            score -= 0.15

        # Ejection fraction
        if self._state["ejection_fraction"] < 0.4:
            score -= 0.3

        # Atherosclerosis
        score -= self._state["atherosclerosis"]

        return max(0, score)


class GeneRegulatoryModel(PhysiologicalModel):
    """
    Gene regulatory network model.

    Models:
    - Transcription factor dynamics
    - Gene expression regulation
    - Epigenetic modifications
    """

    def __init__(self):
        """Initialize gene regulatory model."""
        super().__init__("gene_regulatory")

        # Key regulatory genes
        self._genes = ["TP53", "MYC", "NFKB", "HIF1A", "FOXO3", "SIRT1"]

        self._parameters = {
            "transcription_rate": 0.1,
            "degradation_rate": 0.05,
            "epigenetic_drift_rate": 0.001,
        }

        self._state = {
            "expression": {gene: 1.0 for gene in self._genes},
            "methylation": {gene: 0.5 for gene in self._genes},
            "acetylation": {gene: 0.5 for gene in self._genes},
        }

        # Regulatory network (simplified)
        self._interactions = {
            "TP53": {"MYC": -0.5, "NFKB": -0.3},
            "MYC": {"TP53": -0.2},
            "NFKB": {"FOXO3": -0.4},
            "HIF1A": {"MYC": 0.3},
            "FOXO3": {"SIRT1": 0.3},
            "SIRT1": {"TP53": 0.2, "FOXO3": 0.2},
        }

    def initialize(self, patient_params: dict[str, Any]) -> None:
        """Initialize from patient parameters."""
        age = patient_params.get("age", 50)

        # Age-related epigenetic drift
        for gene in self._genes:
            drift = self._parameters["epigenetic_drift_rate"] * age
            self._state["methylation"][gene] += np.random.normal(0, drift)
            self._state["methylation"][gene] = np.clip(self._state["methylation"][gene], 0, 1)

        # Patient-specific variants
        for variant, effect in patient_params.get("genetic_variants", {}).items():
            if variant in self._state["expression"]:
                if effect == "gain":
                    self._state["expression"][variant] *= 1.5
                elif effect == "loss":
                    self._state["expression"][variant] *= 0.5

    def step(self, dt: float, inputs: dict[str, float]) -> dict[str, float]:
        """Advance gene regulatory simulation."""
        stress = inputs.get("stress", 0)
        hypoxia = inputs.get("hypoxia", 0)
        caloric_restriction = inputs.get("caloric_restriction", 0)

        # Update expression based on regulatory network
        new_expression = {}

        for gene in self._genes:
            base_rate = self._parameters["transcription_rate"]

            # Regulation from other genes
            regulation = 0
            for regulator, effect in self._interactions.get(gene, {}).items():
                regulation += effect * self._state["expression"].get(regulator, 1)

            # Environmental signals
            if gene == "TP53" and stress > 0.5:
                regulation += 0.5
            if gene == "HIF1A" and hypoxia > 0.3:
                regulation += 0.8
            if gene in ["FOXO3", "SIRT1"] and caloric_restriction > 0:
                regulation += 0.3

            # Epigenetic modulation
            methyl_effect = 1 - self._state["methylation"][gene]
            acetyl_effect = self._state["acetylation"][gene]

            # Update expression
            production = base_rate * (1 + regulation) * methyl_effect * acetyl_effect
            degradation = self._parameters["degradation_rate"] * self._state["expression"][gene]

            new_expression[gene] = self._state["expression"][gene] + (production - degradation) * dt
            new_expression[gene] = max(0.1, min(10, new_expression[gene]))

        self._state["expression"] = new_expression

        # Epigenetic drift
        for gene in self._genes:
            drift = np.random.normal(0, self._parameters["epigenetic_drift_rate"]) * dt
            self._state["methylation"][gene] += drift
            self._state["methylation"][gene] = np.clip(self._state["methylation"][gene], 0, 1)

        return {
            "expression": self._state["expression"].copy(),
            "tumor_suppressor_activity": self._state["expression"]["TP53"],
            "longevity_signature": self._compute_longevity_signature(),
        }

    def _compute_longevity_signature(self) -> float:
        """Compute longevity-associated gene signature."""
        longevity_genes = ["FOXO3", "SIRT1"]
        stress_genes = ["MYC", "NFKB"]

        positive = sum(self._state["expression"].get(g, 1) for g in longevity_genes)
        negative = sum(self._state["expression"].get(g, 1) for g in stress_genes)

        return positive / (negative + 0.1)
