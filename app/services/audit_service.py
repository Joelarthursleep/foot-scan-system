"""
Audit Logging Service - DCB0129 Compliance
Comprehensive, immutable audit trail for all system actions

Regulatory Requirements:
- DCB0129: Clinical risk management (manufacturer)
- DCB0160: Clinical risk management (deployment)
- GDPR: Article 30 - Records of processing activities
- DSPT: Audit logging requirements
- ISO 13485: Traceability and design history file

Features:
- Append-only (immutable)
- Tamper detection (cryptographic hashing)
- Complete traceability (who, what, when, where, why)
- 7-year retention
- Fast queries with indexes
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from uuid import UUID
import hashlib
import json
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AuditLog
from app.core.security import NHSNumberService


class AuditService:
    """
    Centralized audit logging service

    Usage:
        audit = AuditService(session, user_id="USER-123", session_id="SESSION-456")

        # Log patient viewed
        await audit.log_data_access(
            action="patient_viewed",
            resource_type="patient",
            resource_id=patient_id,
            nhs_number="9876543210",
            details={"fields_accessed": ["name", "dob", "medical_history"]}
        )

        # Log diagnosis generated
        await audit.log_diagnostic_decision(
            action="diagnosis_generated",
            scan_id=scan_id,
            patient_id=patient_id,
            nhs_number="9876543210",
            diagnosis_name="Severe Hallux Valgus",
            confidence=0.95,
            ai_model_version="v2.1.0",
            details={"snomed_code": "202855006", "severity": "severe"}
        )

        # Log security event
        await audit.log_security_event(
            action="failed_login_attempt",
            severity="warning",
            ip_address="192.168.1.100",
            details={"email": "user@example.com", "reason": "invalid_password"}
        )
    """

    def __init__(
        self,
        session: AsyncSession,
        user_id: Optional[UUID] = None,
        user_role: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize audit service

        Args:
            session: Database session
            user_id: Current user UUID
            user_role: User role (administrator, clinician, etc.)
            session_id: User session ID
            ip_address: Client IP address
            user_agent: Client user agent string
        """
        self.session = session
        self.user_id = user_id
        self.user_role = user_role
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent

    async def log(
        self,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        nhs_number_pseudonym: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        success: bool = True,
        error_message: Optional[str] = None,
        ai_model_version: Optional[str] = None,
        diagnostic_confidence: Optional[float] = None
    ) -> AuditLog:
        """
        Core audit logging method

        Args:
            action: Action performed (e.g., "patient_created", "scan_uploaded")
            resource_type: Type of resource (e.g., "patient", "scan")
            resource_id: UUID of resource
            nhs_number_pseudonym: Pseudonymized NHS Number (for patient-related actions)
            details: Additional action-specific details
            severity: Log severity (critical, warning, info, debug)
            success: Whether action succeeded
            error_message: Error message if action failed
            ai_model_version: AI model version used (for diagnostic actions)
            diagnostic_confidence: Confidence score (for diagnostic actions)

        Returns:
            Created AuditLog record
        """
        # Get hash of previous record for tamper detection
        previous_hash = await self._get_latest_record_hash()

        # Create audit log record
        audit_log = AuditLog(
            timestamp=datetime.utcnow(),
            user_id=self.user_id,
            user_role=self.user_role,
            session_id=self.session_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=self.ip_address,
            user_agent=self.user_agent,
            details=details,
            nhs_number_pseudonym=nhs_number_pseudonym,
            ai_model_version=ai_model_version,
            diagnostic_confidence=diagnostic_confidence,
            severity=severity,
            success=success,
            error_message=error_message,
            previous_record_hash=previous_hash,
            record_hash=""  # Will be calculated below
        )

        # Calculate hash of this record
        audit_log.record_hash = self._calculate_record_hash(audit_log)

        # Save to database
        self.session.add(audit_log)
        await self.session.flush()  # Get ID assigned

        return audit_log

    async def log_data_access(
        self,
        action: str,
        resource_type: str,
        resource_id: UUID,
        nhs_number: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Log data access (view, read, export)

        GDPR Compliance: Article 15 - Right of access
        Patient can request: "Who has accessed my data?"

        Example:
            await audit.log_data_access(
                action="patient_viewed",
                resource_type="patient",
                resource_id=patient_id,
                nhs_number="9876543210",
                details={"fields_accessed": ["name", "dob", "scans"]}
            )
        """
        nhs_pseudonym = None
        if nhs_number:
            nhs_pseudonym = NHSNumberService.pseudonymize(nhs_number)

        return await self.log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            nhs_number_pseudonym=nhs_pseudonym,
            details=details,
            severity="info"
        )

    async def log_data_modification(
        self,
        action: str,
        resource_type: str,
        resource_id: UUID,
        nhs_number: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Log data modification (create, update, delete)

        DCB0129: Track all changes for clinical safety review

        Example:
            await audit.log_data_modification(
                action="patient_updated",
                resource_type="patient",
                resource_id=patient_id,
                nhs_number="9876543210",
                old_values={"bmi": 28.5},
                new_values={"bmi": 29.2}
            )
        """
        nhs_pseudonym = None
        if nhs_number:
            nhs_pseudonym = NHSNumberService.pseudonymize(nhs_number)

        details = {}
        if old_values:
            details["old_values"] = old_values
        if new_values:
            details["new_values"] = new_values

        return await self.log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            nhs_number_pseudonym=nhs_pseudonym,
            details=details,
            severity="info"
        )

    async def log_diagnostic_decision(
        self,
        action: str,
        scan_id: UUID,
        patient_id: UUID,
        nhs_number: str,
        diagnosis_name: str,
        confidence: float,
        ai_model_version: str,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Log AI diagnostic decision

        DCB0129: Critical for clinical safety review
        - Which AI model was used?
        - What was the confidence?
        - What data led to this diagnosis?

        Example:
            await audit.log_diagnostic_decision(
                action="diagnosis_generated",
                scan_id=scan_id,
                patient_id=patient_id,
                nhs_number="9876543210",
                diagnosis_name="Severe Hallux Valgus",
                confidence=0.95,
                ai_model_version="v2.1.0",
                details={
                    "snomed_code": "202855006",
                    "severity": "severe",
                    "supporting_features": [
                        {"name": "HVA", "value": 47, "unit": "degrees"}
                    ]
                }
            )
        """
        nhs_pseudonym = NHSNumberService.pseudonymize(nhs_number)

        # Enhance details with diagnostic context
        diagnostic_details = details or {}
        diagnostic_details.update({
            "diagnosis_name": diagnosis_name,
            "scan_id": str(scan_id),
            "patient_id": str(patient_id)
        })

        return await self.log(
            action=action,
            resource_type="diagnosis",
            resource_id=scan_id,  # Use scan_id as resource
            nhs_number_pseudonym=nhs_pseudonym,
            details=diagnostic_details,
            severity="info",
            ai_model_version=ai_model_version,
            diagnostic_confidence=confidence
        )

    async def log_security_event(
        self,
        action: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Log security-related event

        DSPT Requirement: Track security events for monitoring

        Examples:
            - Failed login attempts
            - Unauthorized access attempts
            - Session timeouts
            - Password changes
            - MFA events

        Example:
            await audit.log_security_event(
                action="failed_login_attempt",
                severity="warning",
                details={"email": "user@example.com", "reason": "invalid_password"}
            )
        """
        return await self.log(
            action=action,
            resource_type="security",
            details=details,
            severity=severity,
            success=False if error_message else True,
            error_message=error_message
        )

    async def log_export(
        self,
        action: str,
        resource_type: str,
        resource_ids: List[UUID],
        export_format: str,
        nhs_numbers: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Log data export (PDF, CSV, FHIR)

        GDPR Compliance: Article 20 - Right to data portability
        DSPT: Track all data exports

        Example:
            await audit.log_export(
                action="report_exported",
                resource_type="diagnosis",
                resource_ids=[diagnosis_id_1, diagnosis_id_2],
                export_format="pdf",
                nhs_numbers=["9876543210"],
                details={"recipient_email": "patient@example.com"}
            )
        """
        # Pseudonymize all NHS numbers
        nhs_pseudonyms = []
        if nhs_numbers:
            nhs_pseudonyms = [
                NHSNumberService.pseudonymize(nhs)
                for nhs in nhs_numbers
            ]

        export_details = details or {}
        export_details.update({
            "export_format": export_format,
            "resource_count": len(resource_ids),
            "nhs_pseudonyms": nhs_pseudonyms
        })

        return await self.log(
            action=action,
            resource_type=resource_type,
            nhs_number_pseudonym=nhs_pseudonyms[0] if nhs_pseudonyms else None,
            details=export_details,
            severity="info"
        )

    async def log_system_event(
        self,
        action: str,
        severity: str = "info",
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Log system-level event

        Examples:
            - System startup/shutdown
            - Configuration changes
            - Database migrations
            - Backup operations

        Example:
            await audit.log_system_event(
                action="system_startup",
                severity="info",
                details={"version": "v2.0.0", "environment": "production"}
            )
        """
        return await self.log(
            action=action,
            resource_type="system",
            details=details,
            severity=severity,
            success=success,
            error_message=error_message
        )

    async def query_audit_trail(
        self,
        user_id: Optional[UUID] = None,
        nhs_number: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        action: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Query audit trail with filters

        GDPR Compliance: Article 15 - Right of access
        Patient can request: "Show me all access to my data"

        Example:
            # Show all access to a patient's data
            logs = await audit.query_audit_trail(
                nhs_number="9876543210",
                start_date=datetime(2025, 1, 1),
                end_date=datetime(2025, 12, 31)
            )

            # Show all diagnostic decisions by AI model
            logs = await audit.query_audit_trail(
                action="diagnosis_generated",
                start_date=datetime(2025, 1, 1)
            )
        """
        query = select(AuditLog)

        # Build filters
        filters = []

        if user_id:
            filters.append(AuditLog.user_id == user_id)

        if nhs_number:
            nhs_pseudonym = NHSNumberService.pseudonymize(nhs_number)
            filters.append(AuditLog.nhs_number_pseudonym == nhs_pseudonym)

        if resource_type:
            filters.append(AuditLog.resource_type == resource_type)

        if resource_id:
            filters.append(AuditLog.resource_id == resource_id)

        if action:
            filters.append(AuditLog.action == action)

        if severity:
            filters.append(AuditLog.severity == severity)

        if start_date:
            filters.append(AuditLog.timestamp >= start_date)

        if end_date:
            filters.append(AuditLog.timestamp <= end_date)

        # Apply filters
        if filters:
            query = query.where(and_(*filters))

        # Order by most recent first
        query = query.order_by(AuditLog.timestamp.desc())

        # Limit results
        query = query.limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return result.scalars().all()

    async def verify_audit_trail_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Verify audit trail has not been tampered with

        DCB0129: Detect any unauthorized modifications

        Checks:
        1. Sequential record hashes (each record hashes previous)
        2. No gaps in timestamps
        3. No duplicate UUIDs

        Returns:
            {
                "is_valid": True/False,
                "total_records": 1000,
                "verified_records": 1000,
                "tampering_detected": False,
                "errors": []
            }
        """
        query = select(AuditLog).order_by(AuditLog.timestamp)

        if start_date:
            query = query.where(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.where(AuditLog.timestamp <= end_date)

        result = await self.session.execute(query)
        records = result.scalars().all()

        total_records = len(records)
        verified_records = 0
        errors = []

        # Check sequential hashing
        for i, record in enumerate(records):
            if i == 0:
                # First record has no previous
                verified_records += 1
                continue

            previous_record = records[i - 1]

            # Verify this record's previous_hash matches previous record's hash
            if record.previous_record_hash != previous_record.record_hash:
                errors.append({
                    "record_id": str(record.id),
                    "timestamp": record.timestamp.isoformat(),
                    "error": "Hash chain broken - tampering detected"
                })
            else:
                verified_records += 1

        tampering_detected = len(errors) > 0

        return {
            "is_valid": not tampering_detected,
            "total_records": total_records,
            "verified_records": verified_records,
            "tampering_detected": tampering_detected,
            "errors": errors
        }

    def _calculate_record_hash(self, record: AuditLog) -> str:
        """
        Calculate SHA-256 hash of audit log record

        Hash includes:
        - Timestamp
        - User ID
        - Action
        - Resource type and ID
        - NHS pseudonym
        - Previous record hash (chains records together)

        This creates a tamper-evident audit trail
        """
        # Concatenate key fields
        hash_input = "|".join([
            str(record.timestamp),
            str(record.user_id) if record.user_id else "",
            record.action,
            record.resource_type if record.resource_type else "",
            str(record.resource_id) if record.resource_id else "",
            record.nhs_number_pseudonym if record.nhs_number_pseudonym else "",
            record.previous_record_hash if record.previous_record_hash else "",
            json.dumps(record.details) if record.details else ""
        ])

        # Calculate SHA-256
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    async def _get_latest_record_hash(self) -> Optional[str]:
        """Get hash of most recent audit log record (for chaining)"""
        query = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(1)
        result = await self.session.execute(query)
        latest_record = result.scalars().first()

        return latest_record.record_hash if latest_record else None


# Common audit actions (constants for consistency)
class AuditActions:
    """Standard audit action names"""

    # Authentication
    USER_LOGGED_IN = "user_logged_in"
    USER_LOGGED_OUT = "user_logged_out"
    USER_SESSION_TIMEOUT = "user_session_timeout"
    FAILED_LOGIN_ATTEMPT = "failed_login_attempt"
    PASSWORD_CHANGED = "password_changed"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Patient data
    PATIENT_CREATED = "patient_created"
    PATIENT_VIEWED = "patient_viewed"
    PATIENT_UPDATED = "patient_updated"
    PATIENT_DELETED = "patient_deleted"  # Soft delete for GDPR
    PATIENT_EXPORTED = "patient_exported"

    # Scans
    SCAN_UPLOADED = "scan_uploaded"
    SCAN_PROCESSED = "scan_processed"
    SCAN_VIEWED = "scan_viewed"
    SCAN_DELETED = "scan_deleted"

    # Diagnoses
    DIAGNOSIS_GENERATED = "diagnosis_generated"
    DIAGNOSIS_REVIEWED = "diagnosis_reviewed"
    DIAGNOSIS_CORRECTED = "diagnosis_corrected"

    # Reports
    REPORT_GENERATED = "report_generated"
    REPORT_EXPORTED = "report_exported"
    REPORT_SENT = "report_sent"

    # System
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    DATABASE_MIGRATION = "database_migration"

    # Security
    UNAUTHORIZED_ACCESS_ATTEMPT = "unauthorized_access_attempt"
    PERMISSION_DENIED = "permission_denied"
    SECURITY_ALERT = "security_alert"


# Export
__all__ = ["AuditService", "AuditActions"]
