"""
Clinical Terminology Mapper
Maps between SNOMED CT, ICD-10, and plain language descriptions

NHS Interoperability Requirements:
- SNOMED CT for clinical documentation
- ICD-10 for statistical reporting
- Read Codes v3 (legacy, being phased out)
- dm+d for medications (not applicable for this system)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .snomed_codes import SNOMEDCodes, get_snomed_code, get_snomed_description
from .icd10_codes import ICD10Codes, get_icd10_code, get_icd10_description, add_laterality_to_code


class CodingSystem(Enum):
    """Clinical coding systems"""
    SNOMED_CT = "snomed_ct"
    ICD10 = "icd10"
    READ_V3 = "read_v3"  # Legacy
    PLAIN_TEXT = "plain_text"


@dataclass
class ClinicalCoding:
    """
    Clinical code with multiple coding systems

    Example:
        coding = ClinicalCoding(
            snomed_code="202855006",
            snomed_term="Hallux valgus",
            icd10_code="M20.1",
            icd10_term="Hallux valgus (acquired)",
            plain_text="Bunion on left foot",
            laterality="left"
        )
    """
    snomed_code: Optional[str] = None
    snomed_term: Optional[str] = None
    icd10_code: Optional[str] = None
    icd10_term: Optional[str] = None
    plain_text: Optional[str] = None
    laterality: Optional[str] = None  # "left", "right", "bilateral"
    severity: Optional[str] = None  # "mild", "moderate", "severe"
    confidence: Optional[float] = None  # AI confidence score

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        return {
            "snomed_code": self.snomed_code,
            "snomed_term": self.snomed_term,
            "icd10_code": self.icd10_code,
            "icd10_term": self.icd10_term,
            "plain_text": self.plain_text,
            "laterality": self.laterality,
            "severity": self.severity,
            "confidence": self.confidence
        }

    def to_fhir_codeable_concept(self) -> Dict:
        """
        Convert to FHIR R4 CodeableConcept

        Used for HL7 FHIR interoperability

        Returns:
            FHIR CodeableConcept structure
        """
        codings = []

        if self.snomed_code:
            codings.append({
                "system": "http://snomed.info/sct",
                "code": self.snomed_code,
                "display": self.snomed_term or ""
            })

        if self.icd10_code:
            codings.append({
                "system": "http://hl7.org/fhir/sid/icd-10",
                "code": self.icd10_code,
                "display": self.icd10_term or ""
            })

        return {
            "coding": codings,
            "text": self.plain_text or self.snomed_term or self.icd10_term
        }


class TerminologyMapper:
    """
    Clinical terminology mapper

    Maps between:
    - Plain language condition names
    - SNOMED CT codes
    - ICD-10 codes
    - Clinical severity levels

    Usage:
        mapper = TerminologyMapper()

        # Map from plain text
        coding = mapper.map_condition(
            condition_name="hallux valgus",
            laterality="left",
            severity="moderate"
        )

        print(coding.snomed_code)  # "202855006"
        print(coding.icd10_code)   # "M20.12" (with laterality)

        # Get FHIR representation
        fhir_concept = coding.to_fhir_codeable_concept()
    """

    def __init__(self):
        """Initialize terminology mapper"""
        self._load_mappings()

    def _load_mappings(self):
        """Load SNOMED CT to ICD-10 mappings"""
        # Comprehensive mapping table
        self.snomed_to_icd10 = {
            # Hallux valgus
            SNOMEDCodes.HALLUX_VALGUS: ICD10Codes.HALLUX_VALGUS,
            SNOMEDCodes.HALLUX_VALGUS_MILD: ICD10Codes.HALLUX_VALGUS,
            SNOMEDCodes.HALLUX_VALGUS_MODERATE: ICD10Codes.HALLUX_VALGUS,
            SNOMEDCodes.HALLUX_VALGUS_SEVERE: ICD10Codes.HALLUX_VALGUS,

            # Hallux rigidus/varus
            SNOMEDCodes.HALLUX_RIGIDUS: ICD10Codes.HALLUX_RIGIDUS,
            SNOMEDCodes.HALLUX_VARUS: ICD10Codes.HALLUX_VARUS,

            # Flat foot
            SNOMEDCodes.PES_PLANUS: ICD10Codes.PES_PLANUS_ACQUIRED,
            SNOMEDCodes.PES_PLANUS_ACQUIRED: ICD10Codes.PES_PLANUS_ACQUIRED,
            SNOMEDCodes.PES_PLANUS_CONGENITAL: ICD10Codes.CONGENITAL_PES_PLANUS,

            # High arch
            SNOMEDCodes.PES_CAVUS: ICD10Codes.PES_CAVUS_ACQUIRED,
            SNOMEDCodes.PES_CAVUS_ACQUIRED: ICD10Codes.PES_CAVUS_ACQUIRED,
            SNOMEDCodes.PES_CAVUS_CONGENITAL: ICD10Codes.CONGENITAL_PES_CAVUS,

            # Toe deformities
            SNOMEDCodes.HAMMER_TOE: ICD10Codes.HAMMER_TOE,
            SNOMEDCodes.CLAW_TOE: ICD10Codes.TOE_DEFORMITY_UNSPECIFIED,
            SNOMEDCodes.MALLET_TOE: ICD10Codes.TOE_DEFORMITY_UNSPECIFIED,

            # Plantar fascia
            SNOMEDCodes.PLANTAR_FASCIITIS: ICD10Codes.PLANTAR_FASCIITIS_ALT,
            SNOMEDCodes.HEEL_SPUR: ICD10Codes.PLANTAR_FASCIITIS_ALT,

            # Achilles
            SNOMEDCodes.ACHILLES_TENDINITIS: ICD10Codes.ACHILLES_TENDINITIS,
            SNOMEDCodes.ACHILLES_TENDINOSIS: ICD10Codes.ACHILLES_TENDINITIS,

            # Metatarsalgia
            SNOMEDCodes.METATARSALGIA: ICD10Codes.METATARSALGIA,
            SNOMEDCodes.MORTON_NEUROMA: ICD10Codes.MORTON_NEUROMA,

            # Diabetes
            SNOMEDCodes.DIABETIC_FOOT: ICD10Codes.DIABETIC_FOOT_ULCER,

            # Ankle
            SNOMEDCodes.ANKLE_SPRAIN: ICD10Codes.ANKLE_SPRAIN,

            # Inflammatory
            SNOMEDCodes.GOUT_FOOT: ICD10Codes.GOUT_FOOT,
            SNOMEDCodes.RHEUMATOID_ARTHRITIS_FOOT: ICD10Codes.RHEUMATOID_ARTHRITIS_FOOT,
        }

        # Condition name to both codes
        self.condition_mappings = {
            "hallux valgus": {
                "snomed": SNOMEDCodes.HALLUX_VALGUS,
                "icd10": ICD10Codes.HALLUX_VALGUS,
                "plain_text": "Bunion"
            },
            "bunion": {
                "snomed": SNOMEDCodes.HALLUX_VALGUS,
                "icd10": ICD10Codes.HALLUX_VALGUS,
                "plain_text": "Bunion (Hallux valgus)"
            },
            "hallux rigidus": {
                "snomed": SNOMEDCodes.HALLUX_RIGIDUS,
                "icd10": ICD10Codes.HALLUX_RIGIDUS,
                "plain_text": "Stiff big toe"
            },
            "flat foot": {
                "snomed": SNOMEDCodes.PES_PLANUS,
                "icd10": ICD10Codes.PES_PLANUS_ACQUIRED,
                "plain_text": "Flat foot (fallen arches)"
            },
            "pes planus": {
                "snomed": SNOMEDCodes.PES_PLANUS,
                "icd10": ICD10Codes.PES_PLANUS_ACQUIRED,
                "plain_text": "Flat foot"
            },
            "high arch": {
                "snomed": SNOMEDCodes.PES_CAVUS,
                "icd10": ICD10Codes.PES_CAVUS_ACQUIRED,
                "plain_text": "High arched foot"
            },
            "pes cavus": {
                "snomed": SNOMEDCodes.PES_CAVUS,
                "icd10": ICD10Codes.PES_CAVUS_ACQUIRED,
                "plain_text": "High arched foot"
            },
            "hammer toe": {
                "snomed": SNOMEDCodes.HAMMER_TOE,
                "icd10": ICD10Codes.HAMMER_TOE,
                "plain_text": "Hammer toe deformity"
            },
            "plantar fasciitis": {
                "snomed": SNOMEDCodes.PLANTAR_FASCIITIS,
                "icd10": ICD10Codes.PLANTAR_FASCIITIS_ALT,
                "plain_text": "Plantar fasciitis (heel pain)"
            },
            "morton's neuroma": {
                "snomed": SNOMEDCodes.MORTON_NEUROMA,
                "icd10": ICD10Codes.MORTON_NEUROMA,
                "plain_text": "Morton's neuroma (nerve pain in forefoot)"
            },
            "diabetic foot": {
                "snomed": SNOMEDCodes.DIABETIC_FOOT,
                "icd10": ICD10Codes.DIABETIC_FOOT_ULCER,
                "plain_text": "Diabetic foot complication"
            },
            "achilles tendinitis": {
                "snomed": SNOMEDCodes.ACHILLES_TENDINITIS,
                "icd10": ICD10Codes.ACHILLES_TENDINITIS,
                "plain_text": "Achilles tendon inflammation"
            },
            "ankle sprain": {
                "snomed": SNOMEDCodes.ANKLE_SPRAIN,
                "icd10": ICD10Codes.ANKLE_SPRAIN,
                "plain_text": "Ankle sprain"
            },
            "metatarsalgia": {
                "snomed": SNOMEDCodes.METATARSALGIA,
                "icd10": ICD10Codes.METATARSALGIA,
                "plain_text": "Forefoot pain (metatarsalgia)"
            },
        }

    def map_condition(
        self,
        condition_name: str,
        laterality: Optional[str] = None,
        severity: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> ClinicalCoding:
        """
        Map condition name to clinical codes

        Args:
            condition_name: Plain text condition name
            laterality: "left", "right", or "bilateral"
            severity: "mild", "moderate", or "severe"
            confidence: AI confidence score (0.0-1.0)

        Returns:
            ClinicalCoding with all code systems

        Example:
            >>> mapper.map_condition("hallux valgus", laterality="left", severity="moderate")
            ClinicalCoding(
                snomed_code="202855006",
                snomed_term="Hallux valgus",
                icd10_code="M20.12",
                icd10_term="Hallux valgus (acquired), left foot",
                plain_text="Bunion on left foot - moderate severity",
                laterality="left",
                severity="moderate"
            )
        """
        condition_lower = condition_name.lower().strip()

        # Check if we have a mapping
        mapping = self.condition_mappings.get(condition_lower)

        if not mapping:
            # Try partial match
            for key, value in self.condition_mappings.items():
                if key in condition_lower or condition_lower in key:
                    mapping = value
                    break

        if not mapping:
            # Return basic coding with no standard codes
            return ClinicalCoding(
                plain_text=condition_name,
                laterality=laterality,
                severity=severity,
                confidence=confidence
            )

        # Build clinical coding
        snomed_code = mapping["snomed"]
        icd10_base = mapping["icd10"]
        plain_text = mapping["plain_text"]

        # Get SNOMED term
        snomed_term = get_snomed_description(snomed_code)
        if not snomed_term:
            snomed_term = condition_name

        # Get ICD-10 with laterality
        icd10_code = icd10_base
        if laterality and laterality.lower() in ["left", "right", "bilateral"]:
            icd10_code = add_laterality_to_code(icd10_base, laterality)

        icd10_term = get_icd10_description(icd10_base)

        # Enhance plain text with laterality and severity
        enhanced_text = plain_text
        if laterality:
            enhanced_text += f" on {laterality} foot"
        if severity:
            enhanced_text += f" - {severity} severity"

        return ClinicalCoding(
            snomed_code=snomed_code,
            snomed_term=snomed_term,
            icd10_code=icd10_code,
            icd10_term=icd10_term,
            plain_text=enhanced_text,
            laterality=laterality,
            severity=severity,
            confidence=confidence
        )

    def map_from_snomed(
        self,
        snomed_code: str,
        laterality: Optional[str] = None
    ) -> ClinicalCoding:
        """
        Map from SNOMED CT code to other systems

        Args:
            snomed_code: SNOMED CT code
            laterality: Optional laterality

        Returns:
            ClinicalCoding with ICD-10 mapping
        """
        icd10_code = self.snomed_to_icd10.get(snomed_code)

        if laterality and icd10_code:
            icd10_code = add_laterality_to_code(icd10_code, laterality)

        return ClinicalCoding(
            snomed_code=snomed_code,
            snomed_term=get_snomed_description(snomed_code),
            icd10_code=icd10_code,
            icd10_term=get_icd10_description(icd10_code) if icd10_code else None,
            laterality=laterality
        )

    def map_from_icd10(self, icd10_code: str) -> ClinicalCoding:
        """
        Map from ICD-10 code to other systems

        Args:
            icd10_code: ICD-10 code

        Returns:
            ClinicalCoding with SNOMED CT mapping
        """
        # Reverse lookup
        snomed_code = None
        for snomed, icd10 in self.snomed_to_icd10.items():
            if icd10 == icd10_code or icd10_code.startswith(icd10):
                snomed_code = snomed
                break

        return ClinicalCoding(
            snomed_code=snomed_code,
            snomed_term=get_snomed_description(snomed_code) if snomed_code else None,
            icd10_code=icd10_code,
            icd10_term=get_icd10_description(icd10_code)
        )

    def get_severity_specific_snomed(
        self,
        base_condition: str,
        severity: str
    ) -> Optional[str]:
        """
        Get severity-specific SNOMED code

        Args:
            base_condition: Base condition name (e.g., "hallux valgus")
            severity: "mild", "moderate", or "severe"

        Returns:
            Severity-specific SNOMED code if available
        """
        if base_condition.lower() == "hallux valgus":
            severity_map = {
                "mild": SNOMEDCodes.HALLUX_VALGUS_MILD,
                "moderate": SNOMEDCodes.HALLUX_VALGUS_MODERATE,
                "severe": SNOMEDCodes.HALLUX_VALGUS_SEVERE
            }
            return severity_map.get(severity.lower())

        return None

    def batch_map_conditions(
        self,
        conditions: List[Dict]
    ) -> List[ClinicalCoding]:
        """
        Map multiple conditions at once

        Args:
            conditions: List of condition dicts with keys:
                - condition_name: str
                - laterality: Optional[str]
                - severity: Optional[str]
                - confidence: Optional[float]

        Returns:
            List of ClinicalCoding objects
        """
        return [
            self.map_condition(
                condition_name=cond.get("condition_name", ""),
                laterality=cond.get("laterality"),
                severity=cond.get("severity"),
                confidence=cond.get("confidence")
            )
            for cond in conditions
        ]


# Export
__all__ = [
    "TerminologyMapper",
    "ClinicalCoding",
    "CodingSystem"
]
