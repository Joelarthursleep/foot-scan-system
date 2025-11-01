"""
Evidence-Based Recommendations Engine
Implements treatment protocols, drug interaction checking, referral recommendations,
follow-up scheduling, and outcome tracking based on current medical evidence
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class EvidenceLevel(Enum):
    """Evidence quality levels based on medical research"""
    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3 = "3"    # Case-control studies
    LEVEL_4 = "4"    # Case series
    LEVEL_5 = "5"    # Expert opinion

class RecommendationStrength(Enum):
    """Recommendation strength grades"""
    STRONG_FOR = "A"      # Strong recommendation for
    WEAK_FOR = "B"        # Weak recommendation for
    WEAK_AGAINST = "C"    # Weak recommendation against
    STRONG_AGAINST = "D"  # Strong recommendation against
    INSUFFICIENT = "I"    # Insufficient evidence

@dataclass
class ClinicalEvidence:
    """Clinical evidence supporting a recommendation"""
    evidence_id: str
    title: str
    study_type: str  # 'rct', 'cohort', 'case_control', 'systematic_review', etc.
    evidence_level: EvidenceLevel
    sample_size: int
    primary_outcome: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    publication_year: int = 2020
    journal: str = ""
    authors: str = ""
    doi: Optional[str] = None
    summary: str = ""
    limitations: List[str] = field(default_factory=list)
    relevance_score: float = 1.0  # 0-1, how relevant to current case

@dataclass
class TreatmentRecommendation:
    """Evidence-based treatment recommendation"""
    recommendation_id: str
    condition: str
    intervention: str
    description: str
    strength: RecommendationStrength
    evidence_level: EvidenceLevel
    supporting_evidence: List[ClinicalEvidence]
    contraindications: List[str] = field(default_factory=list)
    precautions: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    treatment_duration: Optional[str] = None
    monitoring_requirements: List[str] = field(default_factory=list)
    cost_effectiveness: Optional[str] = None
    patient_factors: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DrugInteraction:
    """Drug interaction information"""
    drug1: str
    drug2: str
    interaction_type: str  # 'major', 'moderate', 'minor'
    mechanism: str
    clinical_significance: str
    management: str
    evidence_level: EvidenceLevel
    references: List[str] = field(default_factory=list)

@dataclass
class ReferralRecommendation:
    """Specialist referral recommendation"""
    specialty: str
    urgency: str  # 'routine', 'urgent', 'emergent'
    reason: str
    expected_outcome: str
    timing: str  # 'immediate', '1-2 weeks', '1 month', etc.
    preparation_required: List[str] = field(default_factory=list)
    supporting_evidence: List[ClinicalEvidence] = field(default_factory=list)

@dataclass
class FollowUpPlan:
    """Evidence-based follow-up scheduling"""
    interval: str  # '1 week', '2 weeks', '1 month', etc.
    reason: str
    assessment_focus: List[str]
    outcome_measures: List[str]
    modification_triggers: List[str] = field(default_factory=list)
    supporting_evidence: List[ClinicalEvidence] = field(default_factory=list)

class EvidenceDatabase:
    """Database of clinical evidence"""

    def __init__(self):
        self.evidence_repository = {}
        self.treatment_protocols = {}
        self.drug_interactions = {}
        self.clinical_guidelines = {}

        # Initialize with standard evidence
        self._load_standard_evidence()

    def _load_standard_evidence(self):
        """Load standard clinical evidence"""

        # Plantar Fasciitis Evidence
        pf_evidence = [
            ClinicalEvidence(
                evidence_id="pf_stretching_rct_2019",
                title="Effectiveness of stretching exercises for plantar fasciitis",
                study_type="rct",
                evidence_level=EvidenceLevel.LEVEL_1B,
                sample_size=240,
                primary_outcome="Pain reduction at 6 weeks",
                effect_size=2.3,
                confidence_interval=(1.8, 2.8),
                p_value=0.001,
                publication_year=2019,
                journal="J Foot Ankle Res",
                summary="Plantar fascia stretching significantly reduces pain and improves function",
                relevance_score=0.95
            ),
            ClinicalEvidence(
                evidence_id="pf_orthotic_systematic_2020",
                title="Systematic review of orthotic interventions for plantar fasciitis",
                study_type="systematic_review",
                evidence_level=EvidenceLevel.LEVEL_1A,
                sample_size=1200,
                primary_outcome="Pain and function improvement",
                effect_size=1.8,
                publication_year=2020,
                journal="Cochrane Database Syst Rev",
                summary="Custom orthotics provide moderate benefit for plantar fasciitis",
                relevance_score=0.90
            )
        ]

        # Diabetic Foot Evidence
        df_evidence = [
            ClinicalEvidence(
                evidence_id="df_offloading_rct_2018",
                title="Total contact casting vs removable cast walker for diabetic foot ulcers",
                study_type="rct",
                evidence_level=EvidenceLevel.LEVEL_1B,
                sample_size=160,
                primary_outcome="Ulcer healing at 12 weeks",
                effect_size=1.4,
                p_value=0.03,
                publication_year=2018,
                journal="Diabetes Care",
                summary="Total contact casting superior to removable devices for healing",
                relevance_score=0.92
            ),
            ClinicalEvidence(
                evidence_id="df_screening_cohort_2021",
                title="Effectiveness of structured diabetic foot screening programs",
                study_type="cohort",
                evidence_level=EvidenceLevel.LEVEL_2B,
                sample_size=5000,
                primary_outcome="Amputation rate reduction",
                effect_size=0.6,
                confidence_interval=(0.4, 0.9),
                publication_year=2021,
                journal="Diabetologia",
                summary="Structured screening reduces amputation risk by 40%",
                relevance_score=0.88
            )
        ]

        # Hallux Valgus Evidence
        hv_evidence = [
            ClinicalEvidence(
                evidence_id="hv_surgery_systematic_2020",
                title="Systematic review of surgical outcomes for hallux valgus",
                study_type="systematic_review",
                evidence_level=EvidenceLevel.LEVEL_1A,
                sample_size=3500,
                primary_outcome="Patient satisfaction and recurrence",
                effect_size=4.2,
                publication_year=2020,
                journal="Foot Ankle Int",
                summary="Surgical correction effective for severe hallux valgus",
                relevance_score=0.85
            )
        ]

        # Store evidence by condition
        self.evidence_repository['plantar_fasciitis'] = pf_evidence
        self.evidence_repository['diabetic_foot'] = df_evidence
        self.evidence_repository['hallux_valgus'] = hv_evidence

    def get_evidence_for_condition(self, condition: str) -> List[ClinicalEvidence]:
        """Get all evidence for a specific condition"""
        condition_key = condition.lower().replace(' ', '_')
        return self.evidence_repository.get(condition_key, [])

    def search_evidence(self, query: str, min_evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_4) -> List[ClinicalEvidence]:
        """Search evidence database"""
        results = []

        for condition_evidence in self.evidence_repository.values():
            for evidence in condition_evidence:
                if (query.lower() in evidence.title.lower() or
                    query.lower() in evidence.summary.lower()):
                    if self._compare_evidence_levels(evidence.evidence_level, min_evidence_level):
                        results.append(evidence)

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _compare_evidence_levels(self, level1: EvidenceLevel, level2: EvidenceLevel) -> bool:
        """Compare evidence levels (higher quality = lower number)"""
        level_order = {
            EvidenceLevel.LEVEL_1A: 1,
            EvidenceLevel.LEVEL_1B: 2,
            EvidenceLevel.LEVEL_2A: 3,
            EvidenceLevel.LEVEL_2B: 4,
            EvidenceLevel.LEVEL_3: 5,
            EvidenceLevel.LEVEL_4: 6,
            EvidenceLevel.LEVEL_5: 7
        }
        return level_order[level1] <= level_order[level2]

class TreatmentProtocolEngine:
    """Engine for generating evidence-based treatment protocols"""

    def __init__(self, evidence_db: EvidenceDatabase):
        self.evidence_db = evidence_db
        self.protocol_templates = {}
        self._initialize_protocols()

    def _initialize_protocols(self):
        """Initialize standard treatment protocols"""

        # Plantar Fasciitis Protocol
        pf_protocol = {
            'condition': 'plantar_fasciitis',
            'first_line': [
                {
                    'intervention': 'Plantar fascia stretching',
                    'strength': RecommendationStrength.STRONG_FOR,
                    'duration': '6-12 weeks',
                    'evidence_ids': ['pf_stretching_rct_2019']
                },
                {
                    'intervention': 'Activity modification',
                    'strength': RecommendationStrength.STRONG_FOR,
                    'duration': 'Ongoing',
                    'evidence_ids': []
                }
            ],
            'second_line': [
                {
                    'intervention': 'Custom orthotics',
                    'strength': RecommendationStrength.WEAK_FOR,
                    'duration': '3-6 months',
                    'evidence_ids': ['pf_orthotic_systematic_2020']
                },
                {
                    'intervention': 'Corticosteroid injection',
                    'strength': RecommendationStrength.WEAK_FOR,
                    'duration': 'Single injection',
                    'evidence_ids': []
                }
            ],
            'third_line': [
                {
                    'intervention': 'Extracorporeal shock wave therapy',
                    'strength': RecommendationStrength.WEAK_FOR,
                    'duration': '3-6 sessions',
                    'evidence_ids': []
                }
            ]
        }

        # Diabetic Foot Protocol
        df_protocol = {
            'condition': 'diabetic_foot_ulcer',
            'first_line': [
                {
                    'intervention': 'Pressure offloading',
                    'strength': RecommendationStrength.STRONG_FOR,
                    'duration': 'Until healing',
                    'evidence_ids': ['df_offloading_rct_2018']
                },
                {
                    'intervention': 'Wound debridement',
                    'strength': RecommendationStrength.STRONG_FOR,
                    'duration': 'As needed',
                    'evidence_ids': []
                }
            ],
            'infection_present': [
                {
                    'intervention': 'Antibiotic therapy',
                    'strength': RecommendationStrength.STRONG_FOR,
                    'duration': '7-14 days',
                    'evidence_ids': []
                }
            ],
            'prevention': [
                {
                    'intervention': 'Structured foot screening',
                    'strength': RecommendationStrength.STRONG_FOR,
                    'duration': 'Regular intervals',
                    'evidence_ids': ['df_screening_cohort_2021']
                }
            ]
        }

        self.protocol_templates['plantar_fasciitis'] = pf_protocol
        self.protocol_templates['diabetic_foot_ulcer'] = df_protocol

    def generate_treatment_recommendations(self, condition: str, patient_factors: Dict,
                                         previous_treatments: List[str] = None) -> List[TreatmentRecommendation]:
        """Generate evidence-based treatment recommendations"""

        recommendations = []
        condition_key = condition.lower().replace(' ', '_')

        if condition_key not in self.protocol_templates:
            logger.warning(f"No protocol template found for condition: {condition}")
            return recommendations

        protocol = self.protocol_templates[condition_key]
        evidence_list = self.evidence_db.get_evidence_for_condition(condition_key)

        previous_treatments = previous_treatments or []

        # Determine treatment line based on previous treatments
        if not previous_treatments:
            # First-line treatments
            treatment_line = protocol.get('first_line', [])
        elif len(previous_treatments) == 1:
            # Second-line treatments
            treatment_line = protocol.get('second_line', [])
        else:
            # Third-line treatments
            treatment_line = protocol.get('third_line', [])

        # Special considerations
        if condition_key == 'diabetic_foot_ulcer' and patient_factors.get('infection_present'):
            treatment_line.extend(protocol.get('infection_present', []))

        for treatment in treatment_line:
            # Find supporting evidence
            supporting_evidence = []
            for evidence_id in treatment.get('evidence_ids', []):
                for evidence in evidence_list:
                    if evidence.evidence_id == evidence_id:
                        supporting_evidence.append(evidence)

            # Check contraindications
            contraindications = self._check_contraindications(
                treatment['intervention'], patient_factors
            )

            recommendation = TreatmentRecommendation(
                recommendation_id=f"{condition_key}_{treatment['intervention'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
                condition=condition,
                intervention=treatment['intervention'],
                description=self._generate_treatment_description(treatment),
                strength=treatment['strength'],
                evidence_level=supporting_evidence[0].evidence_level if supporting_evidence else EvidenceLevel.LEVEL_5,
                supporting_evidence=supporting_evidence,
                contraindications=contraindications,
                treatment_duration=treatment.get('duration'),
                patient_factors=patient_factors
            )

            recommendations.append(recommendation)

        return recommendations

    def _check_contraindications(self, intervention: str, patient_factors: Dict) -> List[str]:
        """Check for contraindications to treatment"""
        contraindications = []

        intervention_lower = intervention.lower()

        # Corticosteroid injection contraindications
        if 'corticosteroid' in intervention_lower or 'steroid' in intervention_lower:
            if patient_factors.get('diabetes'):
                contraindications.append('Diabetes mellitus (relative contraindication)')
            if patient_factors.get('infection'):
                contraindications.append('Active infection')
            if patient_factors.get('immunocompromised'):
                contraindications.append('Immunocompromised state')

        # Surgery contraindications
        if 'surgery' in intervention_lower or 'surgical' in intervention_lower:
            if patient_factors.get('severe_vascular_disease'):
                contraindications.append('Severe peripheral vascular disease')
            if patient_factors.get('poor_healing_history'):
                contraindications.append('History of poor wound healing')

        # Physical therapy contraindications
        if 'therapy' in intervention_lower and 'physical' in intervention_lower:
            if patient_factors.get('severe_pain') and patient_factors['severe_pain'] > 8:
                contraindications.append('Severe pain limiting mobilization')

        return contraindications

    def _generate_treatment_description(self, treatment: Dict) -> str:
        """Generate detailed treatment description"""
        base_description = f"Implement {treatment['intervention']}"

        if treatment.get('duration'):
            base_description += f" for {treatment['duration']}"

        strength_descriptions = {
            RecommendationStrength.STRONG_FOR: "Strongly recommended",
            RecommendationStrength.WEAK_FOR: "Conditionally recommended",
            RecommendationStrength.WEAK_AGAINST: "Conditionally not recommended",
            RecommendationStrength.STRONG_AGAINST: "Not recommended"
        }

        strength_desc = strength_descriptions.get(treatment['strength'], "")
        if strength_desc:
            base_description += f". {strength_desc} based on current evidence."

        return base_description

class DrugInteractionChecker:
    """Drug interaction checking system"""

    def __init__(self):
        self.interaction_database = {}
        self._initialize_interactions()

    def _initialize_interactions(self):
        """Initialize common drug interactions"""

        interactions = [
            DrugInteraction(
                drug1="warfarin",
                drug2="aspirin",
                interaction_type="major",
                mechanism="Increased bleeding risk through additive anticoagulant effects",
                clinical_significance="Significantly increased risk of bleeding complications",
                management="Monitor INR closely, consider dose reduction, avoid if possible",
                evidence_level=EvidenceLevel.LEVEL_1A
            ),
            DrugInteraction(
                drug1="metformin",
                drug2="contrast_media",
                interaction_type="major",
                mechanism="Risk of lactic acidosis in setting of contrast-induced nephropathy",
                clinical_significance="Potentially fatal lactic acidosis",
                management="Hold metformin 48h before and after contrast, check renal function",
                evidence_level=EvidenceLevel.LEVEL_2A
            ),
            DrugInteraction(
                drug1="nsaids",
                drug2="ace_inhibitors",
                interaction_type="moderate",
                mechanism="NSAIDs reduce ACE inhibitor effectiveness and increase nephrotoxicity",
                clinical_significance="Reduced blood pressure control, acute kidney injury risk",
                management="Monitor blood pressure and renal function, use lowest effective dose",
                evidence_level=EvidenceLevel.LEVEL_1B
            ),
            DrugInteraction(
                drug1="tramadol",
                drug2="ssri",
                interaction_type="major",
                mechanism="Increased risk of serotonin syndrome",
                clinical_significance="Life-threatening serotonin syndrome",
                management="Avoid combination, use alternative analgesic",
                evidence_level=EvidenceLevel.LEVEL_2B
            )
        ]

        for interaction in interactions:
            key1 = f"{interaction.drug1}_{interaction.drug2}"
            key2 = f"{interaction.drug2}_{interaction.drug1}"
            self.interaction_database[key1] = interaction
            self.interaction_database[key2] = interaction

    def check_interactions(self, medications: List[str]) -> List[DrugInteraction]:
        """Check for drug interactions in medication list"""
        interactions = []

        for i, drug1 in enumerate(medications):
            for drug2 in medications[i+1:]:
                key = f"{drug1.lower()}_{drug2.lower()}"
                if key in self.interaction_database:
                    interactions.append(self.interaction_database[key])

        return interactions

    def check_new_medication(self, new_drug: str, current_medications: List[str]) -> List[DrugInteraction]:
        """Check interactions for a new medication against current regimen"""
        interactions = []

        for current_drug in current_medications:
            key = f"{new_drug.lower()}_{current_drug.lower()}"
            if key in self.interaction_database:
                interactions.append(self.interaction_database[key])

        return interactions

class ReferralEngine:
    """Specialist referral recommendation engine"""

    def __init__(self, evidence_db: EvidenceDatabase):
        self.evidence_db = evidence_db
        self.referral_criteria = {}
        self._initialize_referral_criteria()

    def _initialize_referral_criteria(self):
        """Initialize referral criteria"""

        self.referral_criteria = {
            'orthopedic_surgery': {
                'conditions': [
                    {
                        'condition': 'hallux_valgus',
                        'criteria': ['severe_deformity', 'conservative_failed', 'functional_limitation'],
                        'urgency': 'routine',
                        'preparation': ['weight_bearing_xrays', 'vascular_assessment']
                    },
                    {
                        'condition': 'stress_fracture',
                        'criteria': ['non_healing', 'multiple_fractures', 'high_risk_location'],
                        'urgency': 'urgent',
                        'preparation': ['mri', 'bone_density_scan']
                    }
                ]
            },
            'vascular_surgery': {
                'conditions': [
                    {
                        'condition': 'peripheral_arterial_disease',
                        'criteria': ['abi_less_than_0.9', 'non_healing_wounds', 'rest_pain'],
                        'urgency': 'urgent',
                        'preparation': ['ankle_brachial_index', 'doppler_ultrasound']
                    }
                ]
            },
            'infectious_disease': {
                'conditions': [
                    {
                        'condition': 'diabetic_foot_infection',
                        'criteria': ['osteomyelitis_suspected', 'failed_antibiotic_therapy', 'systemic_infection'],
                        'urgency': 'urgent',
                        'preparation': ['blood_cultures', 'bone_biopsy', 'imaging']
                    }
                ]
            },
            'endocrinology': {
                'conditions': [
                    {
                        'condition': 'diabetic_foot_complications',
                        'criteria': ['poor_glycemic_control', 'recurrent_ulcers', 'charcot_arthropathy'],
                        'urgency': 'routine',
                        'preparation': ['hba1c', 'comprehensive_metabolic_panel']
                    }
                ]
            }
        }

    def generate_referral_recommendations(self, conditions: Dict, patient_factors: Dict) -> List[ReferralRecommendation]:
        """Generate specialist referral recommendations"""

        recommendations = []

        for specialty, specialty_criteria in self.referral_criteria.items():
            for condition_criteria in specialty_criteria['conditions']:
                condition_name = condition_criteria['condition']

                # Check if patient has this condition
                if not self._patient_has_condition(condition_name, conditions):
                    continue

                # Check if referral criteria are met
                criteria_met = self._check_referral_criteria(
                    condition_criteria['criteria'], conditions, patient_factors
                )

                if criteria_met:
                    referral = ReferralRecommendation(
                        specialty=specialty,
                        urgency=condition_criteria['urgency'],
                        reason=f"{condition_name.replace('_', ' ').title()} meeting referral criteria",
                        expected_outcome=self._get_expected_outcome(specialty, condition_name),
                        timing=self._get_referral_timing(condition_criteria['urgency']),
                        preparation_required=condition_criteria.get('preparation', [])
                    )
                    recommendations.append(referral)

        return recommendations

    def _patient_has_condition(self, condition_name: str, conditions: Dict) -> bool:
        """Check if patient has the specified condition"""
        for name, condition in conditions.items():
            if condition_name in name.lower() and getattr(condition, 'detected', False):
                return True
        return False

    def _check_referral_criteria(self, criteria: List[str], conditions: Dict, patient_factors: Dict) -> bool:
        """Check if referral criteria are met"""
        criteria_met = 0
        required_criteria = max(1, len(criteria) // 2)  # At least half the criteria

        for criterion in criteria:
            if self._evaluate_criterion(criterion, conditions, patient_factors):
                criteria_met += 1

        return criteria_met >= required_criteria

    def _evaluate_criterion(self, criterion: str, conditions: Dict, patient_factors: Dict) -> bool:
        """Evaluate a specific referral criterion"""

        if criterion == 'severe_deformity':
            return any(getattr(c, 'severity', '') == 'severe' for c in conditions.values())

        elif criterion == 'conservative_failed':
            return patient_factors.get('failed_conservative_treatment', False)

        elif criterion == 'functional_limitation':
            return patient_factors.get('functional_score', 10) < 6

        elif criterion == 'abi_less_than_0.9':
            return patient_factors.get('ankle_brachial_index', 1.0) < 0.9

        elif criterion == 'non_healing_wounds':
            return patient_factors.get('wound_duration_weeks', 0) > 4

        elif criterion == 'poor_glycemic_control':
            return patient_factors.get('hba1c', 7) > 8

        # Default to False for unknown criteria
        return False

    def _get_expected_outcome(self, specialty: str, condition: str) -> str:
        """Get expected outcome from referral"""
        outcomes = {
            'orthopedic_surgery': 'Surgical evaluation and treatment planning',
            'vascular_surgery': 'Vascular assessment and revascularization if indicated',
            'infectious_disease': 'Antibiotic optimization and infection management',
            'endocrinology': 'Diabetes management optimization'
        }
        return outcomes.get(specialty, 'Specialist evaluation and management')

    def _get_referral_timing(self, urgency: str) -> str:
        """Get referral timing based on urgency"""
        timing_map = {
            'routine': '4-6 weeks',
            'urgent': '1-2 weeks',
            'emergent': 'Same day'
        }
        return timing_map.get(urgency, '2-4 weeks')

class FollowUpScheduler:
    """Evidence-based follow-up scheduling system"""

    def __init__(self, evidence_db: EvidenceDatabase):
        self.evidence_db = evidence_db
        self.follow_up_protocols = {}
        self._initialize_protocols()

    def _initialize_protocols(self):
        """Initialize follow-up protocols"""

        self.follow_up_protocols = {
            'plantar_fasciitis': {
                'initial_treatment': {
                    'interval': '2 weeks',
                    'assessment_focus': ['pain_level', 'functional_improvement'],
                    'outcome_measures': ['visual_analog_scale', 'foot_function_index']
                },
                'ongoing_treatment': {
                    'interval': '4 weeks',
                    'assessment_focus': ['treatment_response', 'compliance'],
                    'outcome_measures': ['pain_improvement', 'return_to_activity']
                },
                'treatment_failure': {
                    'interval': '2 weeks',
                    'assessment_focus': ['alternative_treatments', 'referral_need'],
                    'outcome_measures': ['treatment_modification']
                }
            },
            'diabetic_foot_ulcer': {
                'active_ulcer': {
                    'interval': '1-2 weeks',
                    'assessment_focus': ['wound_healing', 'infection_signs', 'offloading_compliance'],
                    'outcome_measures': ['wound_size', 'wound_grade', 'bacterial_culture']
                },
                'healed_ulcer': {
                    'interval': '1 month',
                    'assessment_focus': ['skin_integrity', 'footwear_adequacy', 'sensation'],
                    'outcome_measures': ['recurrence_prevention', 'foot_care_education']
                }
            },
            'high_risk_diabetic_foot': {
                'screening': {
                    'interval': '3 months',
                    'assessment_focus': ['neuropathy_progression', 'vascular_status', 'foot_deformities'],
                    'outcome_measures': ['monofilament_testing', 'pulse_assessment', 'skin_inspection']
                }
            }
        }

    def generate_follow_up_plan(self, conditions: Dict, patient_factors: Dict,
                              treatment_status: str = 'initial') -> List[FollowUpPlan]:
        """Generate evidence-based follow-up plans"""

        plans = []

        for condition_name, condition in conditions.items():
            if not getattr(condition, 'detected', False):
                continue

            condition_key = condition_name.lower().replace(' ', '_')

            if condition_key in self.follow_up_protocols:
                protocol = self.follow_up_protocols[condition_key]

                # Determine appropriate protocol phase
                if 'diabetic' in condition_key and 'ulcer' in condition_key:
                    if patient_factors.get('active_ulcer'):
                        phase = 'active_ulcer'
                    else:
                        phase = 'healed_ulcer'
                elif treatment_status == 'failed':
                    phase = 'treatment_failure'
                elif treatment_status == 'ongoing':
                    phase = 'ongoing_treatment'
                else:
                    phase = 'initial_treatment'

                if phase in protocol:
                    phase_protocol = protocol[phase]

                    plan = FollowUpPlan(
                        interval=phase_protocol['interval'],
                        reason=f"Monitor {condition_name.replace('_', ' ')} treatment response",
                        assessment_focus=phase_protocol['assessment_focus'],
                        outcome_measures=phase_protocol['outcome_measures'],
                        modification_triggers=self._get_modification_triggers(condition_key)
                    )

                    plans.append(plan)

        # Remove duplicates and prioritize
        return self._prioritize_follow_up_plans(plans)

    def _get_modification_triggers(self, condition: str) -> List[str]:
        """Get triggers for treatment modification"""

        triggers = {
            'plantar_fasciitis': [
                'No improvement in pain after 4 weeks',
                'Functional limitation persists',
                'Patient compliance issues'
            ],
            'diabetic_foot_ulcer': [
                'Wound size increase',
                'Signs of infection',
                'Non-compliance with offloading'
            ],
            'high_risk_diabetic_foot': [
                'New neuropathy symptoms',
                'Vascular status deterioration',
                'New foot deformities'
            ]
        }

        return triggers.get(condition, ['Lack of expected improvement'])

    def _prioritize_follow_up_plans(self, plans: List[FollowUpPlan]) -> List[FollowUpPlan]:
        """Prioritize follow-up plans by urgency"""

        # Sort by interval (shorter = higher priority)
        interval_priority = {
            '1 week': 1, '1-2 weeks': 2, '2 weeks': 3,
            '3 weeks': 4, '1 month': 5, '6 weeks': 6,
            '2 months': 7, '3 months': 8, '6 months': 9
        }

        return sorted(plans, key=lambda p: interval_priority.get(p.interval, 10))

class OutcomeTracker:
    """Treatment outcome tracking system"""

    def __init__(self):
        self.outcome_measures = {}
        self.patient_outcomes = {}
        self._initialize_outcome_measures()

    def _initialize_outcome_measures(self):
        """Initialize standard outcome measures"""

        self.outcome_measures = {
            'plantar_fasciitis': [
                {
                    'measure': 'Visual Analog Scale (VAS)',
                    'scale': '0-10',
                    'interpretation': 'Lower scores indicate less pain',
                    'minimal_clinically_important_difference': 2.0
                },
                {
                    'measure': 'Foot Function Index (FFI)',
                    'scale': '0-100',
                    'interpretation': 'Lower scores indicate better function',
                    'minimal_clinically_important_difference': 7.0
                }
            ],
            'diabetic_foot_ulcer': [
                {
                    'measure': 'Wound size (cm²)',
                    'scale': 'Continuous',
                    'interpretation': 'Smaller size indicates healing',
                    'minimal_clinically_important_difference': 0.5
                },
                {
                    'measure': 'Wagner Grade',
                    'scale': '0-5',
                    'interpretation': 'Lower grade indicates better healing',
                    'minimal_clinically_important_difference': 1.0
                }
            ]
        }

    def track_outcome(self, patient_id: str, condition: str, measure: str,
                     value: float, date: datetime = None) -> Dict:
        """Track a patient outcome measure"""

        if date is None:
            date = datetime.now()

        if patient_id not in self.patient_outcomes:
            self.patient_outcomes[patient_id] = {}

        if condition not in self.patient_outcomes[patient_id]:
            self.patient_outcomes[patient_id][condition] = {}

        if measure not in self.patient_outcomes[patient_id][condition]:
            self.patient_outcomes[patient_id][condition][measure] = []

        outcome_entry = {
            'value': value,
            'date': date,
            'clinically_significant': False,
            'trend': 'stable'
        }

        # Calculate trend and clinical significance
        previous_outcomes = self.patient_outcomes[patient_id][condition][measure]
        if previous_outcomes:
            last_outcome = previous_outcomes[-1]
            difference = value - last_outcome['value']

            # Get MCID for this measure
            mcid = self._get_mcid(condition, measure)
            if abs(difference) >= mcid:
                outcome_entry['clinically_significant'] = True

            # Determine trend
            if difference > mcid:
                outcome_entry['trend'] = 'improved' if self._is_improvement(condition, measure, difference) else 'worsened'
            elif difference < -mcid:
                outcome_entry['trend'] = 'worsened' if self._is_improvement(condition, measure, difference) else 'improved'

        self.patient_outcomes[patient_id][condition][measure].append(outcome_entry)

        return outcome_entry

    def _get_mcid(self, condition: str, measure: str) -> float:
        """Get minimal clinically important difference for a measure"""
        if condition in self.outcome_measures:
            for om in self.outcome_measures[condition]:
                if measure.lower() in om['measure'].lower():
                    return om['minimal_clinically_important_difference']
        return 0.5  # Default MCID

    def _is_improvement(self, condition: str, measure: str, difference: float) -> bool:
        """Determine if change represents improvement"""
        # For most measures, lower scores are better (pain, dysfunction)
        improvement_measures = ['pain', 'vas', 'function', 'wound_size', 'wagner']

        if any(term in measure.lower() for term in improvement_measures):
            return difference < 0  # Decrease is improvement

        return difference > 0  # Increase is improvement

    def generate_outcome_summary(self, patient_id: str, condition: str) -> Dict:
        """Generate outcome summary for patient and condition"""

        if (patient_id not in self.patient_outcomes or
            condition not in self.patient_outcomes[patient_id]):
            return {'error': 'No outcome data found'}

        condition_outcomes = self.patient_outcomes[patient_id][condition]
        summary = {
            'patient_id': patient_id,
            'condition': condition,
            'measures': {},
            'overall_trend': 'stable',
            'clinically_significant_changes': 0,
            'last_assessment': None
        }

        for measure, outcomes in condition_outcomes.items():
            if not outcomes:
                continue

            latest = outcomes[-1]
            first = outcomes[0] if len(outcomes) > 1 else latest

            measure_summary = {
                'latest_value': latest['value'],
                'first_value': first['value'],
                'change_from_baseline': latest['value'] - first['value'],
                'trend': latest['trend'],
                'clinically_significant': latest['clinically_significant'],
                'total_assessments': len(outcomes)
            }

            summary['measures'][measure] = measure_summary

            # Update overall summary
            if latest['clinically_significant']:
                summary['clinically_significant_changes'] += 1

            if summary['last_assessment'] is None or latest['date'] > summary['last_assessment']:
                summary['last_assessment'] = latest['date']

        # Determine overall trend
        trends = [m['trend'] for m in summary['measures'].values()]
        if trends.count('improved') > trends.count('worsened'):
            summary['overall_trend'] = 'improved'
        elif trends.count('worsened') > trends.count('improved'):
            summary['overall_trend'] = 'worsened'

        return summary

class EvidenceBasedRecommendationEngine:
    """Main evidence-based recommendation engine"""

    def __init__(self):
        self.evidence_db = EvidenceDatabase()
        self.treatment_engine = TreatmentProtocolEngine(self.evidence_db)
        self.drug_checker = DrugInteractionChecker()
        self.referral_engine = ReferralEngine(self.evidence_db)
        self.follow_up_scheduler = FollowUpScheduler(self.evidence_db)
        self.outcome_tracker = OutcomeTracker()

        logger.info("Evidence-Based Recommendation Engine initialized")

    def generate_comprehensive_recommendations(self, conditions: Dict, patient_factors: Dict,
                                            current_medications: List[str] = None,
                                            treatment_history: Dict = None) -> Dict:
        """Generate comprehensive evidence-based recommendations"""

        recommendations = {
            'treatment_recommendations': [],
            'drug_interactions': [],
            'referral_recommendations': [],
            'follow_up_plan': [],
            'evidence_summary': {},
            'recommendation_summary': "",
            'confidence_level': "moderate"
        }

        current_medications = current_medications or []
        treatment_history = treatment_history or {}

        # Generate treatment recommendations for each condition
        for condition_name, condition in conditions.items():
            if not getattr(condition, 'detected', False):
                continue

            previous_treatments = treatment_history.get(condition_name, [])

            treatment_recs = self.treatment_engine.generate_treatment_recommendations(
                condition_name, patient_factors, previous_treatments
            )
            recommendations['treatment_recommendations'].extend(treatment_recs)

        # Check drug interactions
        if current_medications:
            interactions = self.drug_checker.check_interactions(current_medications)
            recommendations['drug_interactions'] = interactions

        # Generate referral recommendations
        referrals = self.referral_engine.generate_referral_recommendations(
            conditions, patient_factors
        )
        recommendations['referral_recommendations'] = referrals

        # Generate follow-up plan
        follow_ups = self.follow_up_scheduler.generate_follow_up_plan(
            conditions, patient_factors
        )
        recommendations['follow_up_plan'] = follow_ups

        # Generate evidence summary
        recommendations['evidence_summary'] = self._generate_evidence_summary(
            recommendations['treatment_recommendations']
        )

        # Generate recommendation summary
        recommendations['recommendation_summary'] = self._generate_recommendation_summary(
            recommendations
        )

        # Assess overall confidence
        recommendations['confidence_level'] = self._assess_recommendation_confidence(
            recommendations['treatment_recommendations']
        )

        return recommendations

    def _generate_evidence_summary(self, treatment_recommendations: List[TreatmentRecommendation]) -> Dict:
        """Generate summary of evidence quality"""

        evidence_levels = [rec.evidence_level for rec in treatment_recommendations]
        recommendation_strengths = [rec.strength for rec in treatment_recommendations]

        level_counts = {}
        for level in evidence_levels:
            level_counts[level.value] = level_counts.get(level.value, 0) + 1

        strength_counts = {}
        for strength in recommendation_strengths:
            strength_counts[strength.value] = strength_counts.get(strength.value, 0) + 1

        return {
            'evidence_level_distribution': level_counts,
            'recommendation_strength_distribution': strength_counts,
            'high_quality_evidence_count': len([l for l in evidence_levels if l in [EvidenceLevel.LEVEL_1A, EvidenceLevel.LEVEL_1B]]),
            'strong_recommendations_count': len([s for s in recommendation_strengths if s == RecommendationStrength.STRONG_FOR])
        }

    def _generate_recommendation_summary(self, recommendations: Dict) -> str:
        """Generate human-readable recommendation summary"""

        treatment_count = len(recommendations['treatment_recommendations'])
        referral_count = len(recommendations['referral_recommendations'])
        interaction_count = len(recommendations['drug_interactions'])

        summary = f"Generated {treatment_count} evidence-based treatment recommendations. "

        if referral_count > 0:
            summary += f"{referral_count} specialist referral(s) recommended. "

        if interaction_count > 0:
            summary += f"Warning: {interaction_count} potential drug interaction(s) identified. "

        # Add evidence quality note
        evidence_summary = recommendations['evidence_summary']
        high_quality_count = evidence_summary.get('high_quality_evidence_count', 0)
        if high_quality_count > 0:
            summary += f"{high_quality_count} recommendation(s) based on high-quality evidence (RCTs or systematic reviews)."
        else:
            summary += "Recommendations based on available evidence; consider individual patient factors."

        return summary

    def _assess_recommendation_confidence(self, treatment_recommendations: List[TreatmentRecommendation]) -> str:
        """Assess overall confidence in recommendations"""

        if not treatment_recommendations:
            return "insufficient"

        # Count high-quality evidence and strong recommendations
        high_quality = len([r for r in treatment_recommendations
                           if r.evidence_level in [EvidenceLevel.LEVEL_1A, EvidenceLevel.LEVEL_1B]])

        strong_recommendations = len([r for r in treatment_recommendations
                                    if r.strength == RecommendationStrength.STRONG_FOR])

        total_recommendations = len(treatment_recommendations)

        high_quality_ratio = high_quality / total_recommendations
        strong_ratio = strong_recommendations / total_recommendations

        if high_quality_ratio >= 0.7 and strong_ratio >= 0.7:
            return "high"
        elif high_quality_ratio >= 0.4 and strong_ratio >= 0.4:
            return "moderate"
        elif high_quality_ratio >= 0.2 or strong_ratio >= 0.2:
            return "low"
        else:
            return "very_low"

    def track_treatment_outcome(self, patient_id: str, condition: str,
                              outcome_measure: str, value: float) -> Dict:
        """Track treatment outcome"""
        return self.outcome_tracker.track_outcome(patient_id, condition, outcome_measure, value)

    def get_patient_outcome_summary(self, patient_id: str, condition: str) -> Dict:
        """Get outcome summary for patient"""
        return self.outcome_tracker.generate_outcome_summary(patient_id, condition)