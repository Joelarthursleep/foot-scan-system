"""
Database Layer - Medical-Grade Data Management
PostgreSQL with audit logging, encryption, and compliance

Regulatory Compliance:
- DCB0129: Audit trail requirements
- GDPR: Data protection, retention, right to erasure
- DSPT: Data security standards
- ISO 13485: Design control, traceability
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, CheckConstraint, event
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


# Base class for all models
Base = declarative_base()


class TimestampMixin:
    """
    Mixin for created_at and updated_at timestamps

    Automatically tracks:
    - When record was created
    - When record was last modified
    """
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Patient(Base, TimestampMixin):
    """
    Patient demographic information

    GDPR Compliance:
    - PII fields encrypted at application layer
    - Pseudonymized NHS number for logging
    - Supports right to erasure
    """
    __tablename__ = "patients"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # NHS Number (encrypted in database, validated before storage)
    nhs_number_encrypted = Column(String(255), unique=True, nullable=False, index=True)
    nhs_number_pseudonym = Column(String(50), unique=True, nullable=False, index=True)

    # Demographics (encrypted)
    first_name_encrypted = Column(String(255), nullable=False)
    last_name_encrypted = Column(String(255), nullable=False)
    date_of_birth_encrypted = Column(String(255), nullable=False)
    sex = Column(String(10), nullable=False)  # male, female, other

    # Contact (encrypted)
    email_encrypted = Column(String(255), nullable=True)
    phone_encrypted = Column(String(255), nullable=True)

    # Medical history (stored as encrypted JSON)
    medical_history_encrypted = Column(Text, nullable=True)

    # Risk factors (not encrypted - aggregate data for analytics)
    diabetes = Column(Boolean, default=False)
    rheumatoid_arthritis = Column(Boolean, default=False)
    peripheral_neuropathy = Column(Boolean, default=False)
    bmi = Column(Float, nullable=True)
    age = Column(Integer, nullable=True)  # Computed from DOB

    # Data protection
    consent_research = Column(Boolean, default=False)
    consent_anonymous_analytics = Column(Boolean, default=False)
    data_retention_expiry = Column(DateTime, nullable=True)  # When to delete (7 years default)

    # Audit metadata
    created_by = Column(UUID(as_uuid=True), nullable=True)
    updated_by = Column(UUID(as_uuid=True), nullable=True)
    is_deleted = Column(Boolean, default=False)  # Soft delete for GDPR
    deleted_at = Column(DateTime, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index('idx_patient_nhs_pseudonym', 'nhs_number_pseudonym'),
        Index('idx_patient_created_at', 'created_at'),
        Index('idx_patient_age_sex', 'age', 'sex'),
    )


class Scan(Base, TimestampMixin):
    """
    Individual foot scan record

    Traceability:
    - Links to patient
    - AI model version used
    - Processing parameters
    - Quality metrics
    """
    __tablename__ = "scans"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(String(50), unique=True, nullable=False, index=True)  # Human-readable ID

    # Foreign keys
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)

    # Scan metadata
    scan_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    scan_type = Column(String(50), nullable=False)  # "3d_stl", "pressure_mat", "gait_video"
    laterality = Column(String(20), nullable=False)  # "left", "right", "bilateral"

    # File storage
    left_stl_path = Column(String(500), nullable=True)  # Path to encrypted STL file
    right_stl_path = Column(String(500), nullable=True)
    left_stl_encrypted = Column(Boolean, default=True)
    right_stl_encrypted = Column(Boolean, default=True)
    left_stl_checksum = Column(String(64), nullable=True)  # SHA-256 for integrity
    right_stl_checksum = Column(String(64), nullable=True)

    # Processing status
    processing_status = Column(String(20), nullable=False, default="pending")
    # Values: pending, processing, completed, failed
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)

    # Quality metrics
    scan_quality_score = Column(Float, nullable=True)  # 0.0-1.0
    point_cloud_density = Column(Integer, nullable=True)  # Number of points
    missing_data_percentage = Column(Float, nullable=True)
    noise_level_mm = Column(Float, nullable=True)

    # AI/ML traceability (DCB0129 requirement)
    ai_model_version = Column(String(50), nullable=True)
    diagnostic_engine_version = Column(String(50), nullable=True)
    processing_parameters = Column(JSONB, nullable=True)  # All parameters used

    # Audit metadata
    uploaded_by = Column(UUID(as_uuid=True), nullable=True)
    processed_by = Column(UUID(as_uuid=True), nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_scan_patient', 'patient_id'),
        Index('idx_scan_date', 'scan_date'),
        Index('idx_scan_status', 'processing_status'),
    )


class Diagnosis(Base, TimestampMixin):
    """
    Diagnostic findings from scan analysis

    Clinical Safety (DCB0129):
    - Full traceability of diagnostic reasoning
    - Confidence scores
    - Differential diagnoses
    - Evidence links
    """
    __tablename__ = "diagnoses"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign keys
    scan_id = Column(UUID(as_uuid=True), ForeignKey('scans.id'), nullable=False, index=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)

    # Condition details
    condition_name = Column(String(255), nullable=False)
    snomed_code = Column(String(20), nullable=True, index=True)  # SNOMED CT code
    icd10_code = Column(String(10), nullable=True, index=True)  # ICD-10 code

    # Diagnostic confidence (ISO 14971 - risk management)
    confidence_level = Column(String(20), nullable=False)
    # Values: definitive, probable, possible, uncertain, indeterminate
    confidence_score = Column(Float, nullable=False)  # 0.0-1.0

    # Clinical details
    severity = Column(String(20), nullable=False)  # critical, severe, moderate, mild, minimal
    laterality = Column(String(20), nullable=False)  # left, right, bilateral
    onset_type = Column(String(20), nullable=True)  # acute, chronic, progressive

    # Supporting evidence (stored as JSON for flexibility)
    supporting_features = Column(JSONB, nullable=True)
    differential_diagnoses = Column(JSONB, nullable=True)
    diagnostic_criteria_met = Column(JSONB, nullable=True)
    diagnostic_criteria_not_met = Column(JSONB, nullable=True)

    # Clinical justification (full text)
    clinical_justification = Column(Text, nullable=False)

    # Evidence base
    evidence_level = Column(String(50), nullable=True)  # Level 1A, 1B, 2A, etc.
    clinical_guidelines = Column(JSONB, nullable=True)  # ["NICE CG181", ...]
    peer_reviewed_studies = Column(JSONB, nullable=True)  # ["PMID:12345678", ...]

    # Clinical impact
    functional_impact = Column(Text, nullable=True)
    pain_score_estimate = Column(Integer, nullable=True)  # 0-10
    mobility_impact = Column(String(20), nullable=True)  # none, mild, moderate, severe
    quality_of_life_impact = Column(Text, nullable=True)

    # Management recommendations
    management_recommendations = Column(JSONB, nullable=True)
    referral_urgency = Column(String(20), nullable=True)  # emergency, urgent, routine, optional
    specialist_type = Column(String(100), nullable=True)  # podiatry, orthopedics, etc.

    # Safety (DCB0129)
    contraindications = Column(JSONB, nullable=True)
    red_flags = Column(JSONB, nullable=True)

    # AI traceability
    ai_model_version = Column(String(50), nullable=False)
    rule_engine_version = Column(String(50), nullable=False)
    explainability_data = Column(JSONB, nullable=True)  # SHAP values, feature importance

    # Clinical review
    clinician_reviewed = Column(Boolean, default=False)
    clinician_agreed = Column(Boolean, nullable=True)
    clinician_comments = Column(Text, nullable=True)
    reviewed_by = Column(UUID(as_uuid=True), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_diagnosis_scan', 'scan_id'),
        Index('idx_diagnosis_patient', 'patient_id'),
        Index('idx_diagnosis_condition', 'condition_name'),
        Index('idx_diagnosis_severity', 'severity'),
        Index('idx_diagnosis_snomed', 'snomed_code'),
    )


class AuditLog(Base):
    """
    Comprehensive audit trail for all system actions

    DCB0129 Compliance:
    - Immutable log (append-only)
    - Complete traceability
    - 7-year retention
    - Tamper detection

    NEVER UPDATE OR DELETE - APPEND ONLY
    """
    __tablename__ = "audit_logs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # When
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Who
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    user_role = Column(String(50), nullable=True)
    session_id = Column(String(100), nullable=True, index=True)

    # What
    action = Column(String(100), nullable=False, index=True)
    # Examples: "patient_created", "scan_uploaded", "diagnosis_generated",
    # "patient_viewed", "report_exported", "user_logged_in"

    resource_type = Column(String(50), nullable=True, index=True)
    # Examples: "patient", "scan", "diagnosis", "user", "report"

    resource_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Where
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)

    # Details
    details = Column(JSONB, nullable=True)
    # Store action-specific details (e.g., which fields changed)

    # NHS Number (pseudonymized for privacy)
    nhs_number_pseudonym = Column(String(50), nullable=True, index=True)

    # AI/ML traceability
    ai_model_version = Column(String(50), nullable=True)
    diagnostic_confidence = Column(Float, nullable=True)

    # Security
    severity = Column(String(20), nullable=False, default="info")
    # Values: critical, warning, info, debug

    # Result
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)

    # Tamper detection (hash of previous record)
    previous_record_hash = Column(String(64), nullable=True)
    record_hash = Column(String(64), nullable=False)

    # Indexes for fast queries
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_nhs_pseudonym', 'nhs_number_pseudonym'),
        Index('idx_audit_session', 'session_id'),
    )


class HazardLog(Base, TimestampMixin):
    """
    Hazard and risk log per DCB0129 requirements

    Clinical Safety Officer maintains this log
    """
    __tablename__ = "hazard_logs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hazard_id = Column(String(50), unique=True, nullable=False)  # HAZ-001, HAZ-002, etc.

    # Hazard details
    hazard_title = Column(String(255), nullable=False)
    hazard_description = Column(Text, nullable=False)
    hazard_category = Column(String(50), nullable=False)
    # Categories: diagnostic_error, data_loss, security_breach, system_failure,
    # user_error, integration_failure

    # Risk assessment (ISO 14971)
    severity = Column(String(20), nullable=False)  # catastrophic, critical, marginal, negligible
    probability = Column(String(20), nullable=False)  # frequent, probable, occasional, remote, improbable
    risk_level = Column(String(20), nullable=False)  # high, medium, low
    risk_score = Column(Integer, nullable=False)  # Calculated: severity x probability

    # Mitigation
    mitigation_measures = Column(JSONB, nullable=True)
    residual_risk_level = Column(String(20), nullable=True)
    residual_risk_score = Column(Integer, nullable=True)

    # Status
    status = Column(String(20), nullable=False, default="open")
    # Values: open, mitigated, closed, monitoring

    # Ownership
    identified_by = Column(UUID(as_uuid=True), nullable=True)
    assigned_to = Column(UUID(as_uuid=True), nullable=True)
    clinical_safety_officer_reviewed = Column(Boolean, default=False)

    # Tracking
    identified_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    target_resolution_date = Column(DateTime, nullable=True)
    actual_resolution_date = Column(DateTime, nullable=True)

    # Related incidents
    related_incidents = Column(JSONB, nullable=True)
    related_patient_safety_events = Column(JSONB, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_hazard_status', 'status'),
        Index('idx_hazard_risk_level', 'risk_level'),
        Index('idx_hazard_category', 'hazard_category'),
    )


class User(Base, TimestampMixin):
    """
    System users (clinicians, administrators, radiographers)

    Authentication & Authorization:
    - Password hashed with Argon2
    - Role-based access control (RBAC)
    - Session management
    - MFA support
    """
    __tablename__ = "users"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Profile
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    title = Column(String(50), nullable=True)  # Dr., Mr., Mrs., etc.
    professional_registration = Column(String(100), nullable=True)  # GMC, NMC, HCPC number

    # Role-based access control
    role = Column(String(50), nullable=False, default="clinician")
    # Roles: administrator, clinical_safety_officer, clinician,
    # radiographer, audit_viewer, patient

    permissions = Column(JSONB, nullable=True)
    # Granular permissions beyond role

    # Multi-factor authentication
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255), nullable=True)  # TOTP secret

    # Session management
    last_login = Column(DateTime, nullable=True)
    last_activity = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)

    # Audit
    created_by = Column(UUID(as_uuid=True), nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_role', 'role'),
        Index('idx_user_active', 'is_active'),
    )


# Database connection and session management
class DatabaseManager:
    """
    Manages database connections and sessions

    Features:
    - Async SQLAlchemy (high performance)
    - Connection pooling
    - Transaction management
    - Migration support
    """

    def __init__(self, database_url: str):
        """
        Initialize database manager

        Args:
            database_url: PostgreSQL connection string
                         Example: "postgresql+asyncpg://user:pass@localhost/foot_scan"
        """
        self.engine = create_async_engine(
            database_url,
            echo=False,  # Set True for SQL query logging in development
            pool_size=20,  # Connection pool size
            max_overflow=10,  # Additional connections when pool exhausted
            pool_pre_ping=True,  # Test connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def create_tables(self):
        """Create all tables (for development/testing only)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all tables (for testing only)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_session(self) -> AsyncSession:
        """Get database session"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Singleton database manager
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get singleton database manager"""
    global _db_manager
    if _db_manager is None:
        from app.core.config import get_settings
        settings = get_settings()
        _db_manager = DatabaseManager(str(settings.DATABASE_URL))
    return _db_manager


# Export
__all__ = [
    "Base",
    "Patient",
    "Scan",
    "Diagnosis",
    "AuditLog",
    "HazardLog",
    "User",
    "DatabaseManager",
    "get_db_manager"
]
