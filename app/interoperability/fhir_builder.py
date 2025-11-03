"""
FHIR R4 Resource Builder
Converts foot scan diagnostic data to HL7 FHIR R4 resources

FHIR R4 Specification: http://hl7.org/fhir/R4/
NHS FHIR Implementation Guide: https://fhir.nhs.uk/

Supported FHIR Resources:
- Patient (demographics)
- Observation (clinical measurements, findings)
- DiagnosticReport (foot scan results)
- Condition (diagnoses)
- ServiceRequest (orders)
- ImagingStudy (3D scan metadata)
- Media (STL files)

NHS Integration:
- GP Connect compatible
- NHS Digital standards
- SNOMED CT coding
- Read codes v3 (legacy support)
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict

from ..clinical.terminology import TerminologyMapper, ClinicalCoding


class FHIRObservationStatus(Enum):
    """FHIR Observation status codes"""
    REGISTERED = "registered"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


class FHIRDiagnosticReportStatus(Enum):
    """FHIR DiagnosticReport status codes"""
    REGISTERED = "registered"
    PARTIAL = "partial"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    APPENDED = "appended"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


class FHIRConditionClinicalStatus(Enum):
    """FHIR Condition clinical status"""
    ACTIVE = "active"
    RECURRENCE = "recurrence"
    RELAPSE = "relapse"
    INACTIVE = "inactive"
    REMISSION = "remission"
    RESOLVED = "resolved"


@dataclass
class FHIRObservation:
    """
    FHIR R4 Observation resource

    Used for clinical measurements and findings from foot scan
    Example: Hallux valgus angle, arch height index, etc.
    """
    resource_type: str = "Observation"
    id: str = None
    status: str = FHIRObservationStatus.FINAL.value
    category: List[Dict] = None
    code: Dict = None
    subject: Dict = None
    effective_date_time: str = None
    issued: str = None
    performer: List[Dict] = None
    value_quantity: Dict = None
    value_codeable_concept: Dict = None
    interpretation: List[Dict] = None
    note: List[Dict] = None
    body_site: Dict = None
    method: Dict = None
    device: Dict = None
    reference_range: List[Dict] = None
    derived_from: List[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to FHIR JSON"""
        data = {}
        for key, value in asdict(self).items():
            if value is not None:
                # Convert snake_case to camelCase for FHIR
                fhir_key = ''.join(word.capitalize() if i > 0 else word
                                  for i, word in enumerate(key.split('_')))
                data[fhir_key] = value
        return data


@dataclass
class FHIRDiagnosticReport:
    """
    FHIR R4 DiagnosticReport resource

    Complete diagnostic report for foot scan analysis
    """
    resource_type: str = "DiagnosticReport"
    id: str = None
    status: str = FHIRDiagnosticReportStatus.FINAL.value
    category: List[Dict] = None
    code: Dict = None
    subject: Dict = None
    effective_date_time: str = None
    issued: str = None
    performer: List[Dict] = None
    results_interpreter: List[Dict] = None
    specimen: List[Dict] = None
    result: List[Dict] = None  # References to Observation resources
    imaging_study: List[Dict] = None
    media: List[Dict] = None
    conclusion: str = None
    conclusion_code: List[Dict] = None
    presented_form: List[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to FHIR JSON"""
        data = {}
        for key, value in asdict(self).items():
            if value is not None:
                fhir_key = ''.join(word.capitalize() if i > 0 else word
                                  for i, word in enumerate(key.split('_')))
                data[fhir_key] = value
        return data


class FHIRBuilder:
    """
    FHIR R4 Resource Builder

    Converts foot scan diagnostic data to HL7 FHIR R4 resources

    Usage:
        builder = FHIRBuilder(
            system_url="https://footscan.nhs.uk/fhir",
            organization_id="NHS-Trust-123"
        )

        # Create patient
        patient = builder.create_patient(
            nhs_number="1234567890",
            given_name="John",
            family_name="Doe",
            birth_date="1980-01-15",
            gender="male"
        )

        # Create observations for clinical findings
        observations = builder.create_observations_from_features(
            features=clinical_feature_set,
            patient_reference=f"Patient/{patient['id']}",
            scan_id="SCAN-123"
        )

        # Create diagnostic report
        report = builder.create_diagnostic_report(
            patient_reference=f"Patient/{patient['id']}",
            observations=observations,
            diagnoses=diagnosis_list,
            scan_id="SCAN-123"
        )
    """

    def __init__(
        self,
        system_url: str = "https://footscan.nhs.uk/fhir",
        organization_id: str = "NHS-FootScan-System",
        organization_name: str = "NHS Foot Scan Diagnostic System"
    ):
        """
        Initialize FHIR builder

        Args:
            system_url: Base URL for FHIR system identifiers
            organization_id: Organization identifier
            organization_name: Organization display name
        """
        self.system_url = system_url
        self.organization_id = organization_id
        self.organization_name = organization_name
        self.terminology_mapper = TerminologyMapper()

    def create_patient(
        self,
        nhs_number: str,
        given_name: str,
        family_name: str,
        birth_date: str,
        gender: str,
        address: Optional[Dict] = None,
        telecom: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Create FHIR Patient resource

        Args:
            nhs_number: 10-digit NHS Number
            given_name: First name
            family_name: Last name
            birth_date: Birth date (YYYY-MM-DD)
            gender: "male", "female", "other", "unknown"
            address: Optional address dict
            telecom: Optional contact details

        Returns:
            FHIR Patient resource as dict
        """
        patient_id = str(uuid.uuid4())

        patient = {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {
                "profile": [
                    "https://fhir.nhs.uk/StructureDefinition/UKCore-Patient"
                ]
            },
            "identifier": [
                {
                    "system": "https://fhir.nhs.uk/Id/nhs-number",
                    "value": nhs_number
                }
            ],
            "name": [
                {
                    "use": "official",
                    "family": family_name,
                    "given": [given_name]
                }
            ],
            "gender": gender,
            "birthDate": birth_date
        }

        if address:
            patient["address"] = [address]

        if telecom:
            patient["telecom"] = telecom

        return patient

    def create_observation_from_measurement(
        self,
        patient_reference: str,
        code: str,
        display: str,
        value: float,
        unit: str,
        system: str = "http://snomed.info/sct",
        status: str = FHIRObservationStatus.FINAL.value,
        effective_datetime: Optional[str] = None,
        interpretation: Optional[str] = None,
        reference_range_low: Optional[float] = None,
        reference_range_high: Optional[float] = None,
        body_site: Optional[str] = None,
        laterality: Optional[str] = None
    ) -> Dict:
        """
        Create FHIR Observation for clinical measurement

        Args:
            patient_reference: Reference to Patient resource (e.g., "Patient/123")
            code: SNOMED CT or LOINC code
            display: Human-readable name
            value: Measured value
            unit: Unit of measurement
            system: Code system URL
            status: Observation status
            effective_datetime: When measurement was taken
            interpretation: "normal", "high", "low", etc.
            reference_range_low: Normal range lower bound
            reference_range_high: Normal range upper bound
            body_site: Anatomical location
            laterality: "left", "right", "bilateral"

        Returns:
            FHIR Observation resource
        """
        observation_id = str(uuid.uuid4())

        if effective_datetime is None:
            effective_datetime = datetime.utcnow().isoformat() + "Z"

        observation = {
            "resourceType": "Observation",
            "id": observation_id,
            "status": status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "imaging",
                            "display": "Imaging"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": system,
                        "code": code,
                        "display": display
                    }
                ],
                "text": display
            },
            "subject": {
                "reference": patient_reference
            },
            "effectiveDateTime": effective_datetime,
            "issued": datetime.utcnow().isoformat() + "Z",
            "valueQuantity": {
                "value": value,
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit
            },
            "performer": [
                {
                    "reference": f"Organization/{self.organization_id}",
                    "display": self.organization_name
                }
            ],
            "device": {
                "display": "3D Foot Scanner - AI Diagnostic System v2.0"
            }
        }

        # Add interpretation if provided
        if interpretation:
            interpretation_map = {
                "normal": {"code": "N", "display": "Normal"},
                "high": {"code": "H", "display": "High"},
                "low": {"code": "L", "display": "Low"},
                "critical": {"code": "HH", "display": "Critical high"}
            }

            interp = interpretation_map.get(interpretation.lower())
            if interp:
                observation["interpretation"] = [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": interp["code"],
                                "display": interp["display"]
                            }
                        ]
                    }
                ]

        # Add reference range
        if reference_range_low is not None or reference_range_high is not None:
            observation["referenceRange"] = [
                {
                    "low": {
                        "value": reference_range_low,
                        "unit": unit
                    } if reference_range_low is not None else None,
                    "high": {
                        "value": reference_range_high,
                        "unit": unit
                    } if reference_range_high is not None else None
                }
            ]
            # Remove None values
            observation["referenceRange"][0] = {k: v for k, v in observation["referenceRange"][0].items() if v is not None}

        # Add body site
        if body_site:
            observation["bodySite"] = {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": body_site,
                        "display": self._get_body_site_display(body_site, laterality)
                    }
                ]
            }

        return observation

    def _get_body_site_display(self, body_site: str, laterality: Optional[str]) -> str:
        """Get human-readable body site display"""
        body_site_map = {
            "56459004": "Foot",
            "76986000": "Great toe",
            "87342007": "Lesser toes",
            "108371006": "Metatarsal",
            "80144004": "Calcaneus",
            "67411009": "Navicular"
        }

        display = body_site_map.get(body_site, "Foot")

        if laterality:
            display = f"{laterality.capitalize()} {display}"

        return display

    def create_observations_from_features(
        self,
        features: Any,  # CompleteClinicalFeatureSet
        patient_reference: str,
        scan_id: str,
        laterality: str = "unknown"
    ) -> List[Dict]:
        """
        Create multiple FHIR Observations from clinical feature set

        Args:
            features: CompleteClinicalFeatureSet object
            patient_reference: Reference to Patient resource
            scan_id: Scan identifier
            laterality: "left", "right", or "bilateral"

        Returns:
            List of FHIR Observation resources
        """
        observations = []
        effective_datetime = datetime.fromtimestamp(features.extraction_timestamp).isoformat() + "Z"

        morph = features.morphological
        if morph:
            # Hallux valgus angle (key diagnostic measurement)
            observations.append(
                self.create_observation_from_measurement(
                    patient_reference=patient_reference,
                    code="FOOTSCAN-HVA",  # Custom code (would map to LOINC in production)
                    display="Hallux valgus angle",
                    value=morph.hallux_valgus_angle_degrees,
                    unit="degrees",
                    effective_datetime=effective_datetime,
                    interpretation=self._interpret_hva(morph.hallux_valgus_angle_degrees),
                    reference_range_low=0.0,
                    reference_range_high=15.0,
                    body_site="76986000",  # Great toe
                    laterality=laterality
                )
            )

            # Arch height index
            observations.append(
                self.create_observation_from_measurement(
                    patient_reference=patient_reference,
                    code="FOOTSCAN-AHI",
                    display="Arch height index",
                    value=morph.arch_height_index,
                    unit="ratio",
                    effective_datetime=effective_datetime,
                    interpretation=self._interpret_arch_height(morph.arch_height_index),
                    reference_range_low=0.25,
                    reference_range_high=0.35,
                    body_site="56459004",  # Foot
                    laterality=laterality
                )
            )

            # Foot length
            observations.append(
                self.create_observation_from_measurement(
                    patient_reference=patient_reference,
                    code="FOOTSCAN-LENGTH",
                    display="Foot length",
                    value=morph.length_mm,
                    unit="mm",
                    effective_datetime=effective_datetime,
                    body_site="56459004",
                    laterality=laterality
                )
            )

            # Calcaneal pitch angle
            observations.append(
                self.create_observation_from_measurement(
                    patient_reference=patient_reference,
                    code="FOOTSCAN-CPA",
                    display="Calcaneal pitch angle",
                    value=morph.calcaneal_pitch_angle_degrees,
                    unit="degrees",
                    effective_datetime=effective_datetime,
                    interpretation=self._interpret_calcaneal_angle(morph.calcaneal_pitch_angle_degrees),
                    reference_range_low=18.0,
                    reference_range_high=25.0,
                    body_site="80144004",  # Calcaneus
                    laterality=laterality
                )
            )

        biomech = features.biomechanical
        if biomech:
            # Center of pressure offset
            observations.append(
                self.create_observation_from_measurement(
                    patient_reference=patient_reference,
                    code="FOOTSCAN-COP",
                    display="Center of pressure offset",
                    value=biomech.cop_offset_from_center_mm,
                    unit="mm",
                    effective_datetime=effective_datetime,
                    body_site="56459004",
                    laterality=laterality
                )
            )

        return observations

    def _interpret_hva(self, angle: float) -> str:
        """Interpret hallux valgus angle"""
        if angle <= 15.0:
            return "normal"
        elif angle <= 20.0:
            return "high"  # Mild
        elif angle <= 40.0:
            return "high"  # Moderate
        else:
            return "critical"  # Severe

    def _interpret_arch_height(self, index: float) -> str:
        """Interpret arch height index"""
        if 0.25 <= index <= 0.35:
            return "normal"
        elif index < 0.25:
            return "low"  # Flat foot
        else:
            return "high"  # High arch

    def _interpret_calcaneal_angle(self, angle: float) -> str:
        """Interpret calcaneal pitch angle"""
        if 18.0 <= angle <= 25.0:
            return "normal"
        elif angle < 18.0:
            return "low"  # Flat foot indicator
        else:
            return "high"

    def create_condition(
        self,
        patient_reference: str,
        clinical_coding: ClinicalCoding,
        clinical_status: str = FHIRConditionClinicalStatus.ACTIVE.value,
        verification_status: str = "confirmed",
        onset_datetime: Optional[str] = None,
        recorded_date: Optional[str] = None,
        evidence: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Create FHIR Condition resource

        Args:
            patient_reference: Reference to Patient
            clinical_coding: ClinicalCoding with SNOMED/ICD-10
            clinical_status: Clinical status of condition
            verification_status: "provisional", "differential", "confirmed"
            onset_datetime: When condition started
            recorded_date: When condition was recorded
            evidence: Supporting evidence (references to Observations)

        Returns:
            FHIR Condition resource
        """
        condition_id = str(uuid.uuid4())

        if recorded_date is None:
            recorded_date = datetime.utcnow().isoformat() + "Z"

        # Build coding from ClinicalCoding
        codings = []
        if clinical_coding.snomed_code:
            codings.append({
                "system": "http://snomed.info/sct",
                "code": clinical_coding.snomed_code,
                "display": clinical_coding.snomed_term
            })
        if clinical_coding.icd10_code:
            codings.append({
                "system": "http://hl7.org/fhir/sid/icd-10",
                "code": clinical_coding.icd10_code,
                "display": clinical_coding.icd10_term
            })

        condition = {
            "resourceType": "Condition",
            "id": condition_id,
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": clinical_status
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": verification_status
                    }
                ]
            },
            "code": {
                "coding": codings,
                "text": clinical_coding.plain_text
            },
            "subject": {
                "reference": patient_reference
            },
            "recordedDate": recorded_date
        }

        # Add laterality
        if clinical_coding.laterality:
            laterality_map = {
                "left": {"code": "7771000", "display": "Left"},
                "right": {"code": "24028007", "display": "Right"},
                "bilateral": {"code": "51440002", "display": "Bilateral"}
            }
            lat = laterality_map.get(clinical_coding.laterality.lower())
            if lat:
                condition["bodySite"] = [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": lat["code"],
                                "display": lat["display"]
                            }
                        ]
                    }
                ]

        # Add severity
        if clinical_coding.severity:
            severity_map = {
                "mild": {"code": "255604002", "display": "Mild"},
                "moderate": {"code": "6736007", "display": "Moderate"},
                "severe": {"code": "24484000", "display": "Severe"}
            }
            sev = severity_map.get(clinical_coding.severity.lower())
            if sev:
                condition["severity"] = {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": sev["code"],
                            "display": sev["display"]
                        }
                    ]
                }

        # Add evidence
        if evidence:
            condition["evidence"] = evidence

        # Add onset
        if onset_datetime:
            condition["onsetDateTime"] = onset_datetime

        return condition

    def create_diagnostic_report(
        self,
        patient_reference: str,
        observations: List[Dict],
        conditions: List[Dict],
        scan_id: str,
        conclusion: str,
        status: str = FHIRDiagnosticReportStatus.FINAL.value,
        effective_datetime: Optional[str] = None
    ) -> Dict:
        """
        Create FHIR DiagnosticReport

        Args:
            patient_reference: Reference to Patient
            observations: List of Observation resources
            conditions: List of Condition resources
            scan_id: Scan identifier
            conclusion: Clinical conclusion text
            status: Report status
            effective_datetime: When scan was performed

        Returns:
            FHIR DiagnosticReport resource
        """
        report_id = str(uuid.uuid4())

        if effective_datetime is None:
            effective_datetime = datetime.utcnow().isoformat() + "Z"

        # Build result references
        result_references = [
            {"reference": f"Observation/{obs['id']}"}
            for obs in observations
        ]

        # Build conclusion codes from conditions
        conclusion_codes = []
        for condition in conditions:
            if "code" in condition and "coding" in condition["code"]:
                conclusion_codes.extend(condition["code"]["coding"])

        report = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "IMG",
                            "display": "Diagnostic Imaging"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": self.system_url,
                        "code": "FOOT-SCAN-3D",
                        "display": "3D Foot Scan Diagnostic Analysis"
                    }
                ],
                "text": "3D Foot Scan with AI Diagnostic Analysis"
            },
            "subject": {
                "reference": patient_reference
            },
            "effectiveDateTime": effective_datetime,
            "issued": datetime.utcnow().isoformat() + "Z",
            "performer": [
                {
                    "reference": f"Organization/{self.organization_id}",
                    "display": self.organization_name
                }
            ],
            "result": result_references,
            "conclusion": conclusion,
            "conclusionCode": [
                {
                    "coding": conclusion_codes
                }
            ] if conclusion_codes else None
        }

        # Remove None values
        report = {k: v for k, v in report.items() if v is not None}

        return report

    def create_bundle(
        self,
        resources: List[Dict],
        bundle_type: str = "transaction"
    ) -> Dict:
        """
        Create FHIR Bundle

        Args:
            resources: List of FHIR resources
            bundle_type: "document", "message", "transaction", "collection"

        Returns:
            FHIR Bundle resource
        """
        bundle_id = str(uuid.uuid4())

        entries = []
        for resource in resources:
            entry = {
                "fullUrl": f"{self.system_url}/{resource['resourceType']}/{resource['id']}",
                "resource": resource
            }

            if bundle_type == "transaction":
                entry["request"] = {
                    "method": "POST",
                    "url": resource["resourceType"]
                }

            entries.append(entry)

        bundle = {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": bundle_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": entries
        }

        return bundle


# Export
__all__ = [
    "FHIRBuilder",
    "FHIRObservation",
    "FHIRDiagnosticReport",
    "FHIRObservationStatus",
    "FHIRDiagnosticReportStatus",
    "FHIRConditionClinicalStatus"
]
