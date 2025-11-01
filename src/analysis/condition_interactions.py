"""
Multi-Condition Interaction Analysis Module
Models comorbidities, cascade effects, and treatment prioritization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of condition interactions"""
    CAUSAL = "causal"  # One condition causes another
    SYNERGISTIC = "synergistic"  # Conditions worsen each other
    COMPENSATORY = "compensatory"  # One develops due to other
    INDEPENDENT = "independent"  # No interaction
    PROTECTIVE = "protective"  # One prevents progression of other

@dataclass
class ConditionInteraction:
    """Represents interaction between two conditions"""
    condition_a: str
    condition_b: str
    interaction_type: InteractionType
    severity_multiplier: float  # How much worse combined vs separate
    cascade_probability: float  # Probability A leads to B
    timeline_months: int  # Expected time for cascade
    clinical_evidence: str
    treatment_considerations: str

@dataclass
class ComorbidityProfile:
    """Complete comorbidity analysis"""
    primary_condition: str  # Most severe/central condition
    secondary_conditions: List[str]
    interaction_network: Dict[Tuple[str, str], ConditionInteraction]
    cascade_risk_score: float  # 0-100, risk of progression
    combined_severity_score: float  # Adjusted for interactions
    treatment_priority_order: List[Tuple[str, str, float]]  # (condition, reason, priority)
    clinical_summary: str

class ConditionInteractionAnalyzer:
    """
    Analyzes interactions between multiple foot conditions:
    - Identifies causal relationships
    - Models synergistic effects
    - Prioritizes treatment targets
    - Predicts cascade progression
    """

    def __init__(self):
        """Initialize interaction analyzer"""
        self.logger = logging.getLogger(__name__)

        # Load interaction knowledge base
        self.interaction_rules = self._load_interaction_rules()
        self.cascade_pathways = self._load_cascade_pathways()
        self.severity_multipliers = self._load_severity_multipliers()

    def _load_interaction_rules(self) -> Dict[Tuple[str, str], ConditionInteraction]:
        """
        Load evidence-based interaction rules between conditions
        Based on podiatric literature and clinical guidelines
        """
        rules = {}

        # Hallux Valgus (Bunion) interactions
        rules[('hallux_valgus', 'metatarsalgia')] = ConditionInteraction(
            condition_a='hallux_valgus',
            condition_b='metatarsalgia',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.5,
            cascade_probability=0.65,
            timeline_months=24,
            clinical_evidence="Hallux valgus shifts weight to lesser metatarsals, causing overload",
            treatment_considerations="Treat bunion first to prevent metatarsalgia progression"
        )

        rules[('hallux_valgus', 'pes_planus')] = ConditionInteraction(
            condition_a='hallux_valgus',
            condition_b='pes_planus',
            interaction_type=InteractionType.SYNERGISTIC,
            severity_multiplier=1.4,
            cascade_probability=0.45,
            timeline_months=36,
            clinical_evidence="Arch collapse and bunion both involve medial column instability",
            treatment_considerations="Address both with orthotic support and possible surgery"
        )

        # Pes Planus (Flat Foot) interactions
        rules[('pes_planus', 'posterior_tibial_tendon_dysfunction')] = ConditionInteraction(
            condition_a='pes_planus',
            condition_b='posterior_tibial_tendon_dysfunction',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.8,
            cascade_probability=0.55,
            timeline_months=18,
            clinical_evidence="Progressive arch collapse stresses posterior tibial tendon",
            treatment_considerations="Early orthotic intervention critical to prevent tendon degeneration"
        )

        rules[('pes_planus', 'plantar_fasciitis')] = ConditionInteraction(
            condition_a='pes_planus',
            condition_b='plantar_fasciitis',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.6,
            cascade_probability=0.60,
            timeline_months=12,
            clinical_evidence="Arch collapse overstretches plantar fascia, causing inflammation",
            treatment_considerations="Arch support addresses root cause; stretching provides symptom relief"
        )

        rules[('pes_planus', 'knee_pain')] = ConditionInteraction(
            condition_a='pes_planus',
            condition_b='knee_pain',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.3,
            cascade_probability=0.40,
            timeline_months=24,
            clinical_evidence="Pronation alters knee biomechanics, increasing medial compartment stress",
            treatment_considerations="Foot orthotics can reduce knee pain by correcting alignment"
        )

        # Pes Cavus (High Arch) interactions
        rules[('pes_cavus', 'stress_fractures')] = ConditionInteraction(
            condition_a='pes_cavus',
            condition_b='stress_fractures',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=2.0,
            cascade_probability=0.45,
            timeline_months=6,
            clinical_evidence="High arches reduce shock absorption, concentrating stress",
            treatment_considerations="Cushioned orthotics and activity modification essential"
        )

        rules[('pes_cavus', 'ankle_instability')] = ConditionInteraction(
            condition_a='pes_cavus',
            condition_b='ankle_instability',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.5,
            cascade_probability=0.50,
            timeline_months=12,
            clinical_evidence="High arch reduces lateral stability, increasing sprain risk",
            treatment_considerations="Ankle strengthening exercises and bracing may be needed"
        )

        # Plantar Fasciitis interactions
        rules[('plantar_fasciitis', 'heel_spur')] = ConditionInteraction(
            condition_a='plantar_fasciitis',
            condition_b='heel_spur',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.2,
            cascade_probability=0.30,
            timeline_months=36,
            clinical_evidence="Chronic fascia tension causes calcification at insertion point",
            treatment_considerations="Spurs are typically asymptomatic; treat fasciitis"
        )

        # Hammertoe interactions
        rules[('hammertoe', 'metatarsalgia')] = ConditionInteraction(
            condition_a='hammertoe',
            condition_b='metatarsalgia',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.4,
            cascade_probability=0.55,
            timeline_months=18,
            clinical_evidence="Hammertoe alters metatarsal head pressure distribution",
            treatment_considerations="Correct hammer toe to relieve forefoot pressure"
        )

        rules[('hammertoe', 'corns_calluses')] = ConditionInteraction(
            condition_a='hammertoe',
            condition_b='corns_calluses',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=1.1,
            cascade_probability=0.80,
            timeline_months=6,
            clinical_evidence="Toe deformity causes friction and pressure points",
            treatment_considerations="Debridement provides relief; correct deformity for cure"
        )

        # Diabetes-related interactions
        rules[('diabetic_neuropathy', 'charcot_foot')] = ConditionInteraction(
            condition_a='diabetic_neuropathy',
            condition_b='charcot_foot',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=3.0,
            cascade_probability=0.15,
            timeline_months=12,
            clinical_evidence="Neuropathy causes undetected trauma, leading to joint destruction",
            treatment_considerations="Aggressive offloading and immobilization critical"
        )

        rules[('diabetic_neuropathy', 'ulceration')] = ConditionInteraction(
            condition_a='diabetic_neuropathy',
            condition_b='ulceration',
            interaction_type=InteractionType.CAUSAL,
            severity_multiplier=4.0,
            cascade_probability=0.25,
            timeline_months=24,
            clinical_evidence="Loss of sensation prevents pressure awareness",
            treatment_considerations="Regular foot inspections and pressure-relieving footwear essential"
        )

        # Morton's Neuroma interactions
        rules[('mortons_neuroma', 'metatarsalgia')] = ConditionInteraction(
            condition_a='mortons_neuroma',
            condition_b='metatarsalgia',
            interaction_type=InteractionType.SYNERGISTIC,
            severity_multiplier=1.6,
            cascade_probability=0.70,
            timeline_months=6,
            clinical_evidence="Neuroma and metatarsal overload often coexist and worsen each other",
            treatment_considerations="Address both mechanical issues and neuroma compression"
        )

        return rules

    def _load_cascade_pathways(self) -> Dict[str, List[Tuple[str, float, int]]]:
        """
        Load cascade progression pathways
        Returns: {initial_condition: [(subsequent_condition, probability, timeline_months), ...]}
        """
        return {
            'pes_planus': [
                ('plantar_fasciitis', 0.60, 12),
                ('posterior_tibial_tendon_dysfunction', 0.55, 18),
                ('hallux_valgus', 0.45, 36),
                ('knee_pain', 0.40, 24)
            ],
            'hallux_valgus': [
                ('metatarsalgia', 0.65, 24),
                ('hammertoe', 0.35, 36),
                ('arthritis_big_toe', 0.30, 48)
            ],
            'plantar_fasciitis': [
                ('heel_spur', 0.30, 36),
                ('achilles_tendinitis', 0.25, 18)
            ],
            'pes_cavus': [
                ('stress_fractures', 0.45, 6),
                ('ankle_instability', 0.50, 12),
                ('plantar_fasciitis', 0.35, 18)
            ],
            'diabetic_neuropathy': [
                ('ulceration', 0.25, 24),
                ('charcot_foot', 0.15, 12),
                ('amputation', 0.05, 60)
            ]
        }

    def _load_severity_multipliers(self) -> Dict[str, float]:
        """
        Load base severity weights for different conditions
        Higher weight = more impactful condition
        """
        return {
            'diabetic_neuropathy': 5.0,
            'charcot_foot': 4.5,
            'ulceration': 4.5,
            'posterior_tibial_tendon_dysfunction': 4.0,
            'hallux_valgus': 3.0,
            'pes_planus': 3.0,
            'pes_cavus': 3.0,
            'plantar_fasciitis': 2.5,
            'metatarsalgia': 2.0,
            'hammertoe': 2.0,
            'mortons_neuroma': 2.5,
            'heel_spur': 1.5,
            'ankle_instability': 2.5,
            'stress_fractures': 3.5,
            'arthritis': 3.0
        }

    def analyze_comorbidities(self,
                             detected_conditions: List[Dict[str, Any]]) -> ComorbidityProfile:
        """
        Comprehensive analysis of multiple co-existing conditions

        Args:
            detected_conditions: List of detected conditions with severity, confidence, etc.

        Returns:
            Complete comorbidity profile with interaction analysis
        """
        self.logger.info(f"Analyzing comorbidities for {len(detected_conditions)} conditions...")

        if len(detected_conditions) == 0:
            return self._create_empty_profile()

        if len(detected_conditions) == 1:
            return self._create_single_condition_profile(detected_conditions[0])

        # Build interaction network
        interaction_network = self._build_interaction_network(detected_conditions)

        # Identify primary condition (most central/severe)
        primary_condition = self._identify_primary_condition(
            detected_conditions, interaction_network
        )

        # Calculate cascade risk
        cascade_risk = self._calculate_cascade_risk(detected_conditions)

        # Calculate combined severity with interaction effects
        combined_severity = self._calculate_combined_severity(
            detected_conditions, interaction_network
        )

        # Determine treatment priority order
        treatment_priorities = self._prioritize_treatments(
            detected_conditions, interaction_network, primary_condition
        )

        # Generate clinical summary
        clinical_summary = self._generate_comorbidity_summary(
            detected_conditions, primary_condition, interaction_network,
            cascade_risk, treatment_priorities
        )

        secondary_conditions = [
            c['name'] for c in detected_conditions
            if c['name'] != primary_condition
        ]

        return ComorbidityProfile(
            primary_condition=primary_condition,
            secondary_conditions=secondary_conditions,
            interaction_network=interaction_network,
            cascade_risk_score=cascade_risk,
            combined_severity_score=combined_severity,
            treatment_priority_order=treatment_priorities,
            clinical_summary=clinical_summary
        )

    def _build_interaction_network(self,
                                   conditions: List[Dict[str, Any]]) -> Dict[Tuple[str, str], ConditionInteraction]:
        """Build network of interactions between detected conditions"""
        network = {}

        condition_names = [c['name'].lower().replace(' ', '_') for c in conditions]

        for i, cond_a_name in enumerate(condition_names):
            for cond_b_name in condition_names[i+1:]:
                # Check both directions
                key_ab = (cond_a_name, cond_b_name)
                key_ba = (cond_b_name, cond_a_name)

                if key_ab in self.interaction_rules:
                    network[key_ab] = self.interaction_rules[key_ab]
                elif key_ba in self.interaction_rules:
                    network[key_ba] = self.interaction_rules[key_ba]

        return network

    def _identify_primary_condition(self,
                                   conditions: List[Dict[str, Any]],
                                   network: Dict[Tuple[str, str], ConditionInteraction]) -> str:
        """
        Identify the primary (most central/severe) condition
        Uses combination of severity and network centrality
        """
        # Score each condition
        scores = {}

        for cond in conditions:
            cond_name = cond['name']
            cond_key = cond_name.lower().replace(' ', '_')

            # Base score from severity and confidence
            severity_map = {'mild': 1, 'moderate': 2, 'severe': 3}
            severity_score = severity_map.get(cond.get('severity', 'mild').lower(), 1)
            confidence_score = cond.get('confidence', 50) / 100

            base_score = severity_score * confidence_score

            # Add weight from condition type
            type_weight = self.severity_multipliers.get(cond_key, 1.0)

            # Count outgoing causal relationships (conditions this one causes)
            causal_out_count = sum(
                1 for (a, b), interaction in network.items()
                if a == cond_key and interaction.interaction_type == InteractionType.CAUSAL
            )

            # Higher score for conditions that cause other conditions
            network_score = causal_out_count * 0.5

            total_score = base_score * type_weight + network_score

            scores[cond_name] = total_score

        if not scores:
            return conditions[0]['name']

        primary = max(scores.items(), key=lambda x: x[1])[0]
        return primary

    def _calculate_cascade_risk(self, conditions: List[Dict[str, Any]]) -> float:
        """Calculate risk of condition cascade (0-100)"""
        if len(conditions) <= 1:
            return 20.0  # Low baseline risk

        risk_factors = 0
        max_factors = 10

        # Factor 1: Number of existing conditions
        risk_factors += min(len(conditions) / 3, 2.0)

        # Factor 2: Presence of high-risk conditions
        high_risk_conditions = ['diabetic_neuropathy', 'pes_planus', 'hallux_valgus']
        for cond in conditions:
            cond_key = cond['name'].lower().replace(' ', '_')
            if cond_key in high_risk_conditions:
                risk_factors += 2.0

        # Factor 3: Severity of existing conditions
        severe_count = sum(1 for c in conditions if c.get('severity', 'mild').lower() == 'severe')
        risk_factors += min(severe_count, 2.0)

        # Factor 4: Check for known cascade starters
        for cond in conditions:
            cond_key = cond['name'].lower().replace(' ', '_')
            if cond_key in self.cascade_pathways:
                pathways = self.cascade_pathways[cond_key]
                # Add risk based on cascade probabilities
                avg_cascade_prob = np.mean([p[1] for p in pathways])
                risk_factors += avg_cascade_prob * 2

        risk_score = min((risk_factors / max_factors) * 100, 100)
        return round(risk_score, 1)

    def _calculate_combined_severity(self,
                                    conditions: List[Dict[str, Any]],
                                    network: Dict[Tuple[str, str], ConditionInteraction]) -> float:
        """
        Calculate combined severity accounting for interactions
        Returns score 0-100
        """
        if not conditions:
            return 0.0

        # Sum base severities
        severity_map = {'normal': 0, 'mild': 25, 'moderate': 50, 'severe': 75}
        base_severities = [
            severity_map.get(c.get('severity', 'mild').lower(), 25) * (c.get('confidence', 50) / 100)
            for c in conditions
        ]
        base_total = sum(base_severities)

        # Apply interaction multipliers
        interaction_multiplier = 1.0
        for interaction in network.values():
            if interaction.interaction_type in [InteractionType.SYNERGISTIC, InteractionType.CAUSAL]:
                interaction_multiplier *= interaction.severity_multiplier

        # Cap multiplier at reasonable level
        interaction_multiplier = min(interaction_multiplier, 2.5)

        combined_severity = min(base_total * (interaction_multiplier ** 0.5), 100)

        return round(combined_severity, 1)

    def _prioritize_treatments(self,
                              conditions: List[Dict[str, Any]],
                              network: Dict[Tuple[str, str], ConditionInteraction],
                              primary_condition: str) -> List[Tuple[str, str, float]]:
        """
        Prioritize treatment order based on interactions
        Returns: [(condition_name, reason, priority_score), ...]
        """
        priorities = []

        for cond in conditions:
            cond_name = cond['name']
            cond_key = cond_name.lower().replace(' ', '_')

            # Base priority from severity
            severity_map = {'mild': 30, 'moderate': 60, 'severe': 90}
            base_priority = severity_map.get(cond.get('severity', 'mild').lower(), 30)

            # Boost if primary condition
            if cond_name == primary_condition:
                base_priority += 30
                reason = "Primary condition - treating this may prevent cascade"
            else:
                reason = "Secondary condition"

            # Boost if it causes other conditions
            causes_others = False
            for (a, b), interaction in network.items():
                if a == cond_key and interaction.interaction_type == InteractionType.CAUSAL:
                    base_priority += 20
                    causes_others = True
                    reason = f"Causal factor - treating this prevents {b.replace('_', ' ')}"
                    break

            # Boost for urgent conditions
            urgent_conditions = ['diabetic_neuropathy', 'ulceration', 'charcot_foot', 'stress_fractures']
            if cond_key in urgent_conditions:
                base_priority += 40
                reason = "Urgent condition - requires immediate intervention"

            # Type-based weight
            type_weight = self.severity_multipliers.get(cond_key, 1.0)
            final_priority = base_priority * (type_weight ** 0.3)

            priorities.append((cond_name, reason, round(final_priority, 1)))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[2], reverse=True)

        return priorities

    def _generate_comorbidity_summary(self,
                                     conditions: List[Dict[str, Any]],
                                     primary_condition: str,
                                     network: Dict[Tuple[str, str], ConditionInteraction],
                                     cascade_risk: float,
                                     priorities: List[Tuple[str, str, float]]) -> str:
        """Generate clinical summary of comorbidity analysis"""

        n_conditions = len(conditions)

        if cascade_risk >= 70:
            risk_level = "high"
            risk_text = "Urgent intervention recommended to prevent condition cascade."
        elif cascade_risk >= 40:
            risk_level = "moderate"
            risk_text = "Monitor closely and address primary condition promptly."
        else:
            risk_level = "low"
            risk_text = "Risk of additional conditions developing is low."

        summary = f"""
MULTI-CONDITION ANALYSIS ({n_conditions} conditions detected)

Primary Condition: {primary_condition}
This appears to be the central condition that may be driving other findings.

Cascade Risk: {risk_level.upper()} ({cascade_risk}/100)
{risk_text}

Condition Interactions:
"""

        # Add key interactions
        for (cond_a, cond_b), interaction in list(network.items())[:3]:
            summary += f"  • {cond_a.replace('_', ' ').title()} → {cond_b.replace('_', ' ').title()}: "
            summary += f"{interaction.clinical_evidence}\n"

        summary += f"\nTreatment Priority:\n"
        for i, (cond, reason, priority) in enumerate(priorities[:3], 1):
            summary += f"  {i}. {cond} - {reason}\n"

        summary += f"\nRecommendation: Address {priorities[0][0]} first as it has highest impact on overall foot health."

        return summary.strip()

    def _create_empty_profile(self) -> ComorbidityProfile:
        """Create profile for no conditions"""
        return ComorbidityProfile(
            primary_condition="None",
            secondary_conditions=[],
            interaction_network={},
            cascade_risk_score=0.0,
            combined_severity_score=0.0,
            treatment_priority_order=[],
            clinical_summary="No significant conditions detected."
        )

    def _create_single_condition_profile(self, condition: Dict[str, Any]) -> ComorbidityProfile:
        """Create profile for single condition"""
        cond_name = condition['name']
        cond_key = cond_name.lower().replace(' ', '_')

        # Check for cascade risk
        cascade_risk = 20.0
        if cond_key in self.cascade_pathways:
            avg_prob = np.mean([p[1] for p in self.cascade_pathways[cond_key]])
            cascade_risk = avg_prob * 100

        severity_map = {'mild': 25, 'moderate': 50, 'severe': 75}
        severity_score = severity_map.get(condition.get('severity', 'mild').lower(), 25)

        summary = f"Single condition detected: {cond_name}. "
        if cascade_risk > 40:
            summary += f"Moderate risk of developing additional conditions if left untreated. Early intervention recommended."
        else:
            summary += f"Low risk of cascade to other conditions. Standard monitoring appropriate."

        return ComorbidityProfile(
            primary_condition=cond_name,
            secondary_conditions=[],
            interaction_network={},
            cascade_risk_score=cascade_risk,
            combined_severity_score=severity_score,
            treatment_priority_order=[(cond_name, "Only condition present", severity_score)],
            clinical_summary=summary
        )
