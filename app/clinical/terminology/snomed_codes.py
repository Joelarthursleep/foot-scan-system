"""
SNOMED CT Clinical Terminology Codes
Systematized Nomenclature of Medicine - Clinical Terms

NHS Digital requirement for clinical coding
Used for: diagnoses, procedures, anatomical structures, clinical findings

SNOMED CT UK Edition - most commonly used foot and ankle codes
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


@dataclass
class SNOMEDConcept:
    """SNOMED CT concept"""
    code: str
    fsn: str  # Fully Specified Name
    preferred_term: str
    semantic_tag: str
    parent_codes: list = None


class SNOMEDCodes:
    """
    SNOMED CT codes for foot and ankle conditions

    Categories:
    - Structural deformities (hallux valgus, flat foot, etc.)
    - Inflammatory conditions
    - Traumatic conditions
    - Neurological conditions
    - Vascular conditions
    """

    # ========== HALLUX VALGUS (BUNIONS) ==========
    HALLUX_VALGUS = "202855006"  # Hallux valgus (disorder)
    HALLUX_VALGUS_MILD = "427943007"  # Mild hallux valgus
    HALLUX_VALGUS_MODERATE = "427944001"  # Moderate hallux valgus
    HALLUX_VALGUS_SEVERE = "427945000"  # Severe hallux valgus
    HALLUX_VALGUS_BILATERAL = "239846004"  # Bilateral hallux valgus

    HALLUX_RIGIDUS = "202857003"  # Hallux rigidus (disorder)
    HALLUX_VARUS = "203080007"  # Hallux varus (disorder)
    HALLUX_LIMITUS = "429365007"  # Hallux limitus

    # ========== FLAT FOOT (PES PLANUS) ==========
    PES_PLANUS = "53226007"  # Pes planus (disorder)
    PES_PLANUS_ACQUIRED = "203088000"  # Acquired pes planus
    PES_PLANUS_CONGENITAL = "72015007"  # Congenital pes planus
    PES_PLANUS_FLEXIBLE = "249014005"  # Flexible flat foot
    PES_PLANUS_RIGID = "249015006"  # Rigid flat foot

    POSTERIOR_TIBIAL_TENDON_DYSFUNCTION = "203102002"  # PTTD - common cause

    # ========== HIGH ARCH (PES CAVUS) ==========
    PES_CAVUS = "67811000"  # Pes cavus (disorder)
    PES_CAVUS_CONGENITAL = "72014006"  # Congenital pes cavus
    PES_CAVUS_ACQUIRED = "203087005"  # Acquired pes cavus

    # ========== TOE DEFORMITIES ==========
    HAMMER_TOE = "64549002"  # Hammer toe (disorder)
    CLAW_TOE = "18506004"  # Claw toe (disorder)
    MALLET_TOE = "58350007"  # Mallet toe (disorder)
    OVERLAPPING_TOE = "203085002"  # Overlapping toe

    # ========== METATARSAL PATHOLOGY ==========
    METATARSALGIA = "202882003"  # Metatarsalgia (disorder)
    MORTON_NEUROMA = "53053002"  # Morton's metatarsalgia
    MORTON_TOE = "203098003"  # Morton's toe (long 2nd toe)

    STRESS_FRACTURE_METATARSAL = "58196009"  # Stress fracture of metatarsal
    METATARSAL_FRACTURE = "79610001"  # Fracture of metatarsal bone

    # ========== PLANTAR FASCIA ==========
    PLANTAR_FASCIITIS = "202882003"  # Plantar fasciitis (disorder)
    PLANTAR_FIBROMATOSIS = "31018006"  # Plantar fibromatosis
    HEEL_SPUR = "202888004"  # Calcaneal spur

    # ========== ACHILLES TENDON ==========
    ACHILLES_TENDINITIS = "240036003"  # Achilles tendinitis
    ACHILLES_TENDINOSIS = "429668003"  # Achilles tendinosis
    ACHILLES_RUPTURE = "48511005"  # Rupture of Achilles tendon
    ACHILLES_BURSITIS = "236088001"  # Retrocalcaneal bursitis

    # ========== ANKLE PATHOLOGY ==========
    ANKLE_SPRAIN = "44465007"  # Sprain of ankle
    ANKLE_FRACTURE = "16114001"  # Fracture of ankle
    ANKLE_INSTABILITY = "203113001"  # Chronic ankle instability
    ANKLE_ARTHRITIS = "239873007"  # Arthritis of ankle

    # ========== DIABETES-RELATED ==========
    DIABETIC_FOOT = "230578006"  # Diabetic foot (disorder)
    DIABETIC_NEUROPATHY_FOOT = "302227002"  # Diabetic neuropathic arthropathy
    DIABETIC_ULCER_FOOT = "399223003"  # Diabetic foot ulcer
    CHARCOT_FOOT = "111235007"  # Charcot's arthropathy of foot

    # ========== INFLAMMATORY CONDITIONS ==========
    GOUT_FOOT = "90560007"  # Gout of foot
    RHEUMATOID_ARTHRITIS_FOOT = "27355003"  # Rheumatoid arthritis of foot
    PSORIATIC_ARTHRITIS_FOOT = "156370009"  # Psoriatic arthropathy of foot

    # ========== BIOMECHANICAL ==========
    OVERPRONATION = "297171005"  # Excessive pronation of foot
    OVERSUPINATION = "297172003"  # Excessive supination of foot
    EQUINUS = "397995004"  # Equinus deformity

    # ========== SKIN/NAIL CONDITIONS ==========
    CALLUS = "76986008"  # Callus (disorder)
    CORN = "65258006"  # Corn (disorder)
    INGROWN_TOENAIL = "93631009"  # Ingrown toenail
    FUNGAL_NAIL = "414941008"  # Onychomycosis

    # ========== CONGENITAL DEFORMITIES ==========
    CLUBFOOT = "87465003"  # Congenital talipes equinovarus
    METATARSUS_ADDUCTUS = "87244009"  # Metatarsus adductus
    TARSAL_COALITION = "268074009"  # Tarsal coalition

    # ========== ANATOMICAL STRUCTURES ==========
    FOOT_STRUCTURE = "56459004"  # Structure of foot
    ANKLE_REGION = "70258002"  # Ankle region structure
    HALLUX = "76986000"  # Structure of great toe
    LESSER_TOES = "87342007"  # Structure of lesser toes
    METATARSAL = "108371006"  # Metatarsal bone structure
    CALCANEUS = "80144004"  # Calcaneal structure
    TALUS = "108317008"  # Talar structure
    NAVICULAR = "67411009"  # Navicular bone of foot

    # ========== PROCEDURES ==========
    BUNIONECTOMY = "116402007"  # Bunionectomy
    OSTEOTOMY_METATARSAL = "445070000"  # Osteotomy of metatarsal
    ARTHRODESIS_ANKLE = "84762005"  # Arthrodesis of ankle
    TENDON_REPAIR = "265457000"  # Repair of tendon

    # Clinical findings severity qualifiers
    MILD = "255604002"  # Mild (qualifier)
    MODERATE = "6736007"  # Moderate (qualifier)
    SEVERE = "24484000"  # Severe (qualifier)

    # Laterality qualifiers
    LEFT = "7771000"  # Left (qualifier)
    RIGHT = "24028007"  # Right (qualifier)
    BILATERAL = "51440002"  # Bilateral (qualifier)


# SNOMED CT concept database
SNOMED_CONCEPTS: Dict[str, SNOMEDConcept] = {
    "202855006": SNOMEDConcept(
        code="202855006",
        fsn="Hallux valgus (disorder)",
        preferred_term="Hallux valgus",
        semantic_tag="disorder",
        parent_codes=["deformity of great toe"]
    ),
    "53226007": SNOMEDConcept(
        code="53226007",
        fsn="Pes planus (disorder)",
        preferred_term="Flat foot",
        semantic_tag="disorder",
        parent_codes=["deformity of foot"]
    ),
    "67811000": SNOMEDConcept(
        code="67811000",
        fsn="Pes cavus (disorder)",
        preferred_term="High arched foot",
        semantic_tag="disorder",
        parent_codes=["deformity of foot"]
    ),
    "202857003": SNOMEDConcept(
        code="202857003",
        fsn="Hallux rigidus (disorder)",
        preferred_term="Hallux rigidus",
        semantic_tag="disorder",
        parent_codes=["arthritis of great toe"]
    ),
    "64549002": SNOMEDConcept(
        code="64549002",
        fsn="Hammer toe (disorder)",
        preferred_term="Hammer toe",
        semantic_tag="disorder",
        parent_codes=["deformity of toe"]
    ),
    "202882003": SNOMEDConcept(
        code="202882003",
        fsn="Plantar fasciitis (disorder)",
        preferred_term="Plantar fasciitis",
        semantic_tag="disorder",
        parent_codes=["disorder of plantar fascia"]
    ),
    "53053002": SNOMEDConcept(
        code="53053002",
        fsn="Morton's metatarsalgia (disorder)",
        preferred_term="Morton's neuroma",
        semantic_tag="disorder",
        parent_codes=["metatarsalgia"]
    ),
    "230578006": SNOMEDConcept(
        code="230578006",
        fsn="Diabetic foot (disorder)",
        preferred_term="Diabetic foot",
        semantic_tag="disorder",
        parent_codes=["diabetes complication"]
    ),
}


def get_snomed_code(condition_name: str) -> Optional[str]:
    """
    Get SNOMED CT code for condition name

    Args:
        condition_name: Condition name (e.g., "hallux valgus")

    Returns:
        SNOMED CT code or None if not found

    Example:
        >>> get_snomed_code("hallux valgus")
        "202855006"
    """
    condition_lower = condition_name.lower().strip()

    # Direct mapping
    mapping = {
        "hallux valgus": SNOMEDCodes.HALLUX_VALGUS,
        "bunion": SNOMEDCodes.HALLUX_VALGUS,
        "flat foot": SNOMEDCodes.PES_PLANUS,
        "pes planus": SNOMEDCodes.PES_PLANUS,
        "high arch": SNOMEDCodes.PES_CAVUS,
        "pes cavus": SNOMEDCodes.PES_CAVUS,
        "hallux rigidus": SNOMEDCodes.HALLUX_RIGIDUS,
        "hammer toe": SNOMEDCodes.HAMMER_TOE,
        "claw toe": SNOMEDCodes.CLAW_TOE,
        "plantar fasciitis": SNOMEDCodes.PLANTAR_FASCIITIS,
        "morton's neuroma": SNOMEDCodes.MORTON_NEUROMA,
        "diabetic foot": SNOMEDCodes.DIABETIC_FOOT,
        "achilles tendinitis": SNOMEDCodes.ACHILLES_TENDINITIS,
        "metatarsalgia": SNOMEDCodes.METATARSALGIA,
    }

    return mapping.get(condition_lower)


def get_snomed_description(code: str) -> Optional[str]:
    """
    Get preferred term for SNOMED CT code

    Args:
        code: SNOMED CT code

    Returns:
        Preferred term or None if not found

    Example:
        >>> get_snomed_description("202855006")
        "Hallux valgus"
    """
    concept = SNOMED_CONCEPTS.get(code)
    return concept.preferred_term if concept else None


# Export
__all__ = [
    "SNOMEDCodes",
    "SNOMEDConcept",
    "SNOMED_CONCEPTS",
    "get_snomed_code",
    "get_snomed_description"
]
