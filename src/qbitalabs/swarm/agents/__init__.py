"""
SWARM Agent Implementations for QBitaLabs

Specialized agents for quantum-bio discovery:
- MolecularAgent: Molecular modeling and simulation
- PathwayAgent: Biological pathway analysis
- HypothesisAgent: Hypothesis generation and refinement
- ValidationAgent: Result validation and verification
- LiteratureAgent: Scientific literature review
- PatientRiskAgent: Patient risk assessment
- TrialDesignAgent: Clinical trial design
- CohortAgent: Cohort management and analysis
"""

from __future__ import annotations

from qbitalabs.swarm.agents.molecular_agent import MolecularAgent
from qbitalabs.swarm.agents.pathway_agent import PathwayAgent
from qbitalabs.swarm.agents.hypothesis_agent import HypothesisAgent
from qbitalabs.swarm.agents.validation_agent import ValidationAgent
from qbitalabs.swarm.agents.literature_agent import LiteratureAgent
from qbitalabs.swarm.agents.patient_risk_agent import PatientRiskAgent
from qbitalabs.swarm.agents.trial_design_agent import TrialDesignAgent
from qbitalabs.swarm.agents.cohort_agent import CohortAgent

__all__ = [
    "MolecularAgent",
    "PathwayAgent",
    "HypothesisAgent",
    "ValidationAgent",
    "LiteratureAgent",
    "PatientRiskAgent",
    "TrialDesignAgent",
    "CohortAgent",
]
