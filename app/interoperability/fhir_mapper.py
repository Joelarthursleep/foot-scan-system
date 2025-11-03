"""
FHIR Mapper
High-level mapper to convert complete diagnostic data to FHIR Bundle

Simplifies FHIR resource creation for common workflows
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .fhir_builder import FHIRBuilder
from ..clinical.terminology import TerminologyMapper


class FHIRMapper:
    """
    High-level FHIR mapper

    Simplifies conversion of foot scan diagnostics to FHIR Bundle

    Usage:
        mapper = FHIRMapper()

        # Convert complete diagnostic session to FHIR Bundle
        bundle = mapper.create_diagnostic_bundle(
            patient_data={
                "nhs_number": "1234567890",
                "given_name": "John",
                "family_name": "Doe",
                "birth_date": "1980-01-15",
                "gender": "male"
            },
            scan_data={
                "scan_id": "SCAN-123",
                "laterality": "left",
                "features": clinical_feature_set,
                "diagnoses": [
                    {
                        "condition_name": "hallux valgus",
                        "severity": "moderate",
                        "confidence": 0.92
                    }
                ]
            }
        )

        # Post to FHIR server
        # POST bundle to https://fhir.nhs.uk/r4
    """

    def __init__(
        self,
        system_url: str = "https://footscan.nhs.uk/fhir",
        organization_id: str = "NHS-FootScan-System"
    ):
        """Initialize FHIR mapper"""
        self.builder = FHIRBuilder(
            system_url=system_url,
            organization_id=organization_id
        )
        self.terminology_mapper = TerminologyMapper()

    def create_diagnostic_bundle(
        self,
        patient_data: Dict,
        scan_data: Dict,
        bundle_type: str = "transaction"
    ) -> Dict:
        """
        Create complete FHIR Bundle for diagnostic session

        Args:
            patient_data: Patient demographics
                - nhs_number: str
                - given_name: str
                - family_name: str
                - birth_date: str (YYYY-MM-DD)
                - gender: str ("male", "female", "other", "unknown")

            scan_data: Scan and diagnostic data
                - scan_id: str
                - laterality: str ("left", "right", "bilateral")
                - features: CompleteClinicalFeatureSet
                - diagnoses: List[Dict] with:
                    * condition_name: str
                    * severity: str
                    * confidence: float
                    * laterality: Optional[str]

            bundle_type: "transaction" or "collection"

        Returns:
            FHIR Bundle containing Patient, Observations, Conditions, DiagnosticReport
        """
        resources = []

        # 1. Create Patient
        patient = self.builder.create_patient(
            nhs_number=patient_data["nhs_number"],
            given_name=patient_data["given_name"],
            family_name=patient_data["family_name"],
            birth_date=patient_data["birth_date"],
            gender=patient_data["gender"]
        )
        resources.append(patient)
        patient_ref = f"Patient/{patient['id']}"

        # 2. Create Observations from clinical features
        observations = self.builder.create_observations_from_features(
            features=scan_data["features"],
            patient_reference=patient_ref,
            scan_id=scan_data["scan_id"],
            laterality=scan_data.get("laterality", "unknown")
        )
        resources.extend(observations)

        # 3. Create Conditions from diagnoses
        conditions = []
        for diagnosis in scan_data.get("diagnoses", []):
            # Map to SNOMED CT / ICD-10
            clinical_coding = self.terminology_mapper.map_condition(
                condition_name=diagnosis["condition_name"],
                laterality=diagnosis.get("laterality", scan_data.get("laterality")),
                severity=diagnosis.get("severity"),
                confidence=diagnosis.get("confidence")
            )

            # Create evidence references
            evidence = [
                {
                    "detail": [
                        {"reference": f"Observation/{obs['id']}"}
                        for obs in observations
                    ]
                }
            ]

            condition = self.builder.create_condition(
                patient_reference=patient_ref,
                clinical_coding=clinical_coding,
                clinical_status="active",
                verification_status="confirmed" if diagnosis.get("confidence", 0) >= 0.8 else "provisional",
                evidence=evidence
            )
            conditions.append(condition)
            resources.append(condition)

        # 4. Create DiagnosticReport
        conclusion = self._generate_conclusion(scan_data.get("diagnoses", []))
        report = self.builder.create_diagnostic_report(
            patient_reference=patient_ref,
            observations=observations,
            conditions=conditions,
            scan_id=scan_data["scan_id"],
            conclusion=conclusion,
            status="final"
        )
        resources.append(report)

        # 5. Create Bundle
        bundle = self.builder.create_bundle(
            resources=resources,
            bundle_type=bundle_type
        )

        return bundle

    def _generate_conclusion(self, diagnoses: List[Dict]) -> str:
        """Generate clinical conclusion text"""
        if not diagnoses:
            return "No significant findings detected."

        conclusion_parts = ["3D foot scan analysis reveals:"]

        for i, diagnosis in enumerate(diagnoses, 1):
            condition = diagnosis["condition_name"]
            severity = diagnosis.get("severity", "")
            laterality = diagnosis.get("laterality", "")

            if severity:
                conclusion_parts.append(f"{i}. {severity.capitalize()} {condition} ({laterality} foot)")
            else:
                conclusion_parts.append(f"{i}. {condition.capitalize()} ({laterality} foot)")

        conclusion_parts.append("\nRecommendation: Clinical correlation and further evaluation as appropriate.")

        return " ".join(conclusion_parts)

    def export_to_json(self, bundle: Dict, file_path: str):
        """Export FHIR Bundle to JSON file"""
        import json
        with open(file_path, 'w') as f:
            json.dump(bundle, f, indent=2)

    def validate_bundle(self, bundle: Dict) -> Dict[str, Any]:
        """
        Validate FHIR Bundle

        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []

        # Basic validation
        if bundle.get("resourceType") != "Bundle":
            errors.append("Resource is not a Bundle")

        if "entry" not in bundle:
            errors.append("Bundle has no entries")

        # Check for required resources
        resource_types = [entry["resource"]["resourceType"] for entry in bundle.get("entry", [])]

        if "Patient" not in resource_types:
            errors.append("Bundle missing Patient resource")

        if "DiagnosticReport" not in resource_types:
            warnings.append("Bundle missing DiagnosticReport")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "resource_count": len(bundle.get("entry", [])),
            "resource_types": list(set(resource_types))
        }


# Export
__all__ = ["FHIRMapper"]
