"""
ICD-10 Clinical Classification Codes
International Classification of Diseases, 10th Revision

Used for:
- International disease classification
- Insurance billing (UK and international)
- Epidemiological reporting
- Clinical research

ICD-10-CM (Clinical Modification) codes for foot and ankle conditions
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ICD10Code:
    """ICD-10 code with description"""
    code: str
    description: str
    category: str
    includes: list = None
    excludes: list = None


class ICD10Codes:
    """
    ICD-10 codes for foot and ankle conditions

    Format: Letter + 2 digits + optional decimal + up to 2 more digits
    Example: M20.1 = Hallux valgus (acquired)
    """

    # ========== M20: ACQUIRED DEFORMITIES OF FINGERS AND TOES ==========
    HALLUX_VALGUS = "M20.1"  # Hallux valgus (acquired)
    HALLUX_RIGIDUS = "M20.2"  # Hallux rigidus
    HALLUX_VARUS = "M20.3"  # Hallux varus (acquired)
    OTHER_HALLUX_DEFORMITY = "M20.5"  # Other deformities of great toe
    HAMMER_TOE = "M20.4"  # Other hammer toe(s) (acquired)
    TOE_DEFORMITY_UNSPECIFIED = "M20.9"  # Unspecified acquired deformity of finger/toe

    # ========== M21: OTHER ACQUIRED DEFORMITIES OF LIMBS ==========
    PES_PLANUS_ACQUIRED = "M21.4"  # Flat foot [pes planus] (acquired)
    PES_CAVUS_ACQUIRED = "M21.6"  # Other acquired deformities of ankle/foot
    CLAW_FOOT_ACQUIRED = "M21.53"  # Acquired clawfoot
    CAVOVARUS_FOOT = "M21.54"  # Acquired cavovarus deformity of foot

    # ========== Q66: CONGENITAL DEFORMITIES OF FEET ==========
    CONGENITAL_PES_PLANUS = "Q66.5"  # Congenital pes planus
    CONGENITAL_PES_CAVUS = "Q66.7"  # Congenital pes cavus
    CONGENITAL_VARUS_FOOT = "Q66.3"  # Other congenital varus deformities of feet
    CONGENITAL_VALGUS_FOOT = "Q66.6"  # Other congenital valgus deformities of feet
    CLUBFOOT = "Q66.0"  # Congenital talipes equinovarus
    METATARSUS_ADDUCTUS = "Q66.2"  # Metatarsus varus
    CONGENITAL_HALLUX_VALGUS = "Q66.6"  # Other congenital valgus deformities

    # ========== M77: OTHER ENTHESOPATHIES ==========
    PLANTAR_FASCIITIS = "M72.2"  # Plantar fascial fibromatosis (NOTE: Often coded as M77.3)
    PLANTAR_FASCIITIS_ALT = "M77.3"  # Calcaneal spur
    ACHILLES_TENDINITIS = "M76.6"  # Achilles tendinitis
    ACHILLES_BURSITIS = "M76.6"  # Achilles bursitis

    # ========== M79: OTHER SOFT TISSUE DISORDERS ==========
    METATARSALGIA = "M77.4"  # Metatarsalgia
    MORTON_NEUROMA = "G57.6"  # Lesion of plantar nerve (Morton's metatarsalgia)

    # ========== S93: SPRAINS AND STRAINS ==========
    ANKLE_SPRAIN = "S93.4"  # Sprain of ankle
    LIGAMENT_INJURY_ANKLE = "S93.2"  # Rupture of ligaments at ankle and foot level

    # ========== S92: FRACTURES ==========
    CALCANEAL_FRACTURE = "S92.0"  # Fracture of calcaneus
    TALUS_FRACTURE = "S92.1"  # Fracture of talus
    METATARSAL_FRACTURE = "S92.3"  # Fracture of metatarsal bone
    TOE_FRACTURE = "S92.4"  # Fracture of great toe
    STRESS_FRACTURE_FOOT = "M84.37"  # Stress fracture, ankle and foot

    # ========== M86: OSTEOMYELITIS ==========
    OSTEOMYELITIS_FOOT = "M86.67"  # Other chronic osteomyelitis, ankle and foot

    # ========== M19: OTHER ARTHROSIS ==========
    OSTEOARTHRITIS_FOOT = "M19.07"  # Primary osteoarthritis, ankle and foot
    HALLUX_RIGIDUS_ARTHROSIS = "M19.07"  # (Hallux rigidus is arthrosis of 1st MTP joint)

    # ========== E11: TYPE 2 DIABETES WITH COMPLICATIONS ==========
    DIABETIC_FOOT_ULCER = "E11.621"  # Type 2 diabetes with foot ulcer
    DIABETIC_NEUROPATHY = "E11.40"  # Type 2 diabetes with diabetic neuropathy
    DIABETIC_PERIPHERAL_ANGIOPATHY = "E11.51"  # Type 2 diabetes with diabetic peripheral angiopathy

    # For Type 1 diabetes, use E10.xxx codes

    # ========== L97: NON-PRESSURE CHRONIC ULCER OF LOWER LIMB ==========
    FOOT_ULCER = "L97.5"  # Non-pressure chronic ulcer of other part of foot
    TOE_ULCER = "L97.5"  # Non-pressure chronic ulcer of other part of foot

    # ========== I70: ATHEROSCLEROSIS ==========
    PERIPHERAL_ARTERIAL_DISEASE = "I70.2"  # Atherosclerosis of arteries of extremities
    GANGRENE_FOOT = "I70.26"  # Atherosclerosis with gangrene

    # ========== M10: GOUT ==========
    GOUT_FOOT = "M10.07"  # Idiopathic gout, ankle and foot
    GOUT_BIG_TOE = "M10.07"  # Podagra (gout of big toe)

    # ========== M05/M06: RHEUMATOID ARTHRITIS ==========
    RHEUMATOID_ARTHRITIS_FOOT = "M06.07"  # Rheumatoid arthritis without rheumatoid factor, ankle/foot

    # ========== L60: NAIL DISORDERS ==========
    INGROWN_NAIL = "L60.0"  # Ingrowing nail
    ONYCHOMYCOSIS = "B35.1"  # Tinea unguium (fungal nail)

    # ========== M67: OTHER DISORDERS OF SYNOVIUM AND TENDON ==========
    GANGLION_FOOT = "M67.47"  # Ganglion, ankle and foot
    TENDON_RUPTURE = "M66.37"  # Spontaneous rupture of flexor tendons, ankle and foot

    # ========== G57: MONONEUROPATHIES OF LOWER LIMB ==========
    TARSAL_TUNNEL_SYNDROME = "G57.5"  # Tarsal tunnel syndrome
    PLANTAR_NERVE_LESION = "G57.6"  # Lesion of plantar nerve

    # ========== M92: JUVENILE OSTEOCHONDROSIS ==========
    SEVER_DISEASE = "M92.6"  # Juvenile osteochondrosis of tarsus (Sever's disease)
    KOHLER_DISEASE = "M92.6"  # Osteochondrosis of tarsal navicular (Köhler's disease)

    # ========== Z: FACTORS INFLUENCING HEALTH STATUS ==========
    HISTORY_FOOT_SURGERY = "Z87.81"  # Personal history of (corrected) congenital malformations
    ORTHOTIC_FITTING = "Z46.89"  # Encounter for fitting of other devices
    FOOT_SCREENING_DIABETIC = "Z13.29"  # Encounter for screening for other suspected endocrine disorder


# ICD-10 code database
ICD10_CODE_DATABASE: Dict[str, ICD10Code] = {
    "M20.1": ICD10Code(
        code="M20.1",
        description="Hallux valgus (acquired)",
        category="Acquired deformities of toes",
        includes=["Bunion"]
    ),
    "M20.2": ICD10Code(
        code="M20.2",
        description="Hallux rigidus",
        category="Acquired deformities of toes",
        includes=["Arthritis of first MTP joint"]
    ),
    "M21.4": ICD10Code(
        code="M21.4",
        description="Flat foot [pes planus] (acquired)",
        category="Other acquired deformities of limbs",
        includes=["Acquired flat foot", "Fallen arches"]
    ),
    "M21.6": ICD10Code(
        code="M21.6",
        description="Other acquired deformities of ankle and foot",
        category="Other acquired deformities of limbs",
        includes=["Acquired pes cavus"]
    ),
    "M72.2": ICD10Code(
        code="M72.2",
        description="Plantar fascial fibromatosis",
        category="Fibroblastic disorders",
        includes=["Plantar fasciitis"]
    ),
    "M77.3": ICD10Code(
        code="M77.3",
        description="Calcaneal spur",
        category="Other enthesopathies",
        includes=["Heel spur", "Plantar fasciitis with spur"]
    ),
    "G57.6": ICD10Code(
        code="G57.6",
        description="Lesion of plantar nerve",
        category="Mononeuropathies of lower limb",
        includes=["Morton's neuroma", "Morton's metatarsalgia"]
    ),
    "E11.621": ICD10Code(
        code="E11.621",
        description="Type 2 diabetes mellitus with foot ulcer",
        category="Diabetes mellitus",
        includes=["Diabetic foot", "Diabetic ulcer of foot"]
    ),
    "S93.4": ICD10Code(
        code="S93.4",
        description="Sprain and strain of ankle",
        category="Injuries to ankle and foot",
        includes=["Ankle sprain", "Ligament injury"]
    ),
    "M20.4": ICD10Code(
        code="M20.4",
        description="Other hammer toe(s) (acquired)",
        category="Acquired deformities of toes",
        includes=["Hammer toe", "Contracted toe"]
    ),
}


def get_icd10_code(condition_name: str) -> Optional[str]:
    """
    Get ICD-10 code for condition name

    Args:
        condition_name: Condition name

    Returns:
        ICD-10 code or None if not found

    Example:
        >>> get_icd10_code("hallux valgus")
        "M20.1"
    """
    condition_lower = condition_name.lower().strip()

    mapping = {
        "hallux valgus": ICD10Codes.HALLUX_VALGUS,
        "bunion": ICD10Codes.HALLUX_VALGUS,
        "hallux rigidus": ICD10Codes.HALLUX_RIGIDUS,
        "hallux varus": ICD10Codes.HALLUX_VARUS,
        "flat foot": ICD10Codes.PES_PLANUS_ACQUIRED,
        "pes planus": ICD10Codes.PES_PLANUS_ACQUIRED,
        "fallen arches": ICD10Codes.PES_PLANUS_ACQUIRED,
        "high arch": ICD10Codes.PES_CAVUS_ACQUIRED,
        "pes cavus": ICD10Codes.PES_CAVUS_ACQUIRED,
        "hammer toe": ICD10Codes.HAMMER_TOE,
        "plantar fasciitis": ICD10Codes.PLANTAR_FASCIITIS_ALT,
        "heel spur": ICD10Codes.PLANTAR_FASCIITIS_ALT,
        "morton's neuroma": ICD10Codes.MORTON_NEUROMA,
        "morton neuroma": ICD10Codes.MORTON_NEUROMA,
        "metatarsalgia": ICD10Codes.METATARSALGIA,
        "achilles tendinitis": ICD10Codes.ACHILLES_TENDINITIS,
        "achilles tendonitis": ICD10Codes.ACHILLES_TENDINITIS,
        "ankle sprain": ICD10Codes.ANKLE_SPRAIN,
        "diabetic foot": ICD10Codes.DIABETIC_FOOT_ULCER,
        "diabetic foot ulcer": ICD10Codes.DIABETIC_FOOT_ULCER,
        "gout": ICD10Codes.GOUT_FOOT,
        "ingrown toenail": ICD10Codes.INGROWN_NAIL,
        "fungal nail": ICD10Codes.ONYCHOMYCOSIS,
        "osteoarthritis foot": ICD10Codes.OSTEOARTHRITIS_FOOT,
    }

    return mapping.get(condition_lower)


def get_icd10_description(code: str) -> Optional[str]:
    """
    Get description for ICD-10 code

    Args:
        code: ICD-10 code

    Returns:
        Description or None if not found

    Example:
        >>> get_icd10_description("M20.1")
        "Hallux valgus (acquired)"
    """
    icd10_code = ICD10_CODE_DATABASE.get(code)
    return icd10_code.description if icd10_code else None


def get_laterality_suffix(laterality: str) -> str:
    """
    Get ICD-10 laterality suffix

    Args:
        laterality: "left", "right", or "bilateral"

    Returns:
        Laterality code (1=right, 2=left, 3=bilateral)

    Example:
        >>> get_laterality_suffix("left")
        "2"
    """
    laterality_map = {
        "right": "1",
        "left": "2",
        "bilateral": "3",
        "unspecified": "9"
    }
    return laterality_map.get(laterality.lower(), "9")


def add_laterality_to_code(base_code: str, laterality: str) -> str:
    """
    Add laterality to ICD-10 code

    Many foot/ankle codes require 7th character for laterality

    Args:
        base_code: Base ICD-10 code
        laterality: "left", "right", "bilateral", or "unspecified"

    Returns:
        Code with laterality suffix

    Example:
        >>> add_laterality_to_code("M20.1", "left")
        "M20.12"
    """
    suffix = get_laterality_suffix(laterality)

    # If code already has decimal, add after it
    if "." in base_code:
        return f"{base_code}{suffix}"
    else:
        # Add decimal first
        return f"{base_code}.{suffix}"


# Export
__all__ = [
    "ICD10Codes",
    "ICD10Code",
    "ICD10_CODE_DATABASE",
    "get_icd10_code",
    "get_icd10_description",
    "get_laterality_suffix",
    "add_laterality_to_code"
]
