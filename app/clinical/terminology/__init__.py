"""
Clinical Terminology Module
SNOMED CT and ICD-10 mappings for NHS interoperability
"""

from .snomed_codes import SNOMEDCodes, get_snomed_code, get_snomed_description
from .icd10_codes import ICD10Codes, get_icd10_code, get_icd10_description
from .terminology_mapper import TerminologyMapper, ClinicalCoding

__all__ = [
    "SNOMEDCodes",
    "ICD10Codes",
    "TerminologyMapper",
    "ClinicalCoding",
    "get_snomed_code",
    "get_snomed_description",
    "get_icd10_code",
    "get_icd10_description"
]
