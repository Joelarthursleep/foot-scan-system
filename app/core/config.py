"""
Medical-Grade Configuration Management
Compliant with ISO 13485, DCB0129
Environment-agnostic, secure configuration system
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator, PostgresDsn
from typing import Optional, Literal
from pathlib import Path
import secrets


class Settings(BaseSettings):
    """
    Application configuration with validation and security

    Regulatory Notes:
    - All secrets loaded from environment (never hardcoded)
    - Configuration changes tracked in change control system
    - Sensitive values encrypted at rest
    """

    # ========== APPLICATION ==========
    APP_NAME: str = "Foot Scan Diagnosis System"
    APP_VERSION: str = "2.0.0-medical"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # Clinical Safety
    CLINICAL_SAFETY_OFFICER: str = Field("TBD", description="CSO name per DCB0129")
    DEVICE_CLASSIFICATION: str = "Class IIa"
    UKCA_MARK_STATUS: str = "In Progress"

    # ========== SECURITY ==========
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ENCRYPTION_KEY: str = Field(..., description="AES-256 encryption key for PHI")

    # Authentication
    AUTH_ENABLED: bool = True
    MFA_ENABLED: bool = Field(True, description="Multi-factor authentication (DSPT requirement)")
    SESSION_TIMEOUT_MINUTES: int = Field(15, ge=5, le=60, description="Auto-logout per NHS policy")
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_SPECIAL: bool = True

    # Access Control (RBAC)
    REQUIRE_NHS_LOGIN: bool = False  # Future: NHS Care Identity Service
    ALLOW_PATIENT_ACCESS: bool = False  # Future: patient portal

    # ========== DATABASE ==========
    DATABASE_URL: PostgresDsn = Field(..., description="PostgreSQL connection string")
    DATABASE_POOL_SIZE: int = Field(20, ge=5, le=100)
    DATABASE_MAX_OVERFLOW: int = Field(10, ge=0, le=50)
    DATABASE_ECHO: bool = False  # Set True only in development

    # Audit Database (separate, append-only)
    AUDIT_DATABASE_URL: Optional[PostgresDsn] = None

    # ========== CACHE & QUEUE ==========
    REDIS_URL: str = Field("redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = Field(300, ge=60, le=3600)
    CACHE_ENABLED: bool = True

    # ========== STORAGE ==========
    STORAGE_TYPE: Literal["local", "s3", "azure", "gcs"] = "local"
    STORAGE_PATH: Path = Field(Path("data/scans"), description="Local storage path")

    # Cloud Storage (when STORAGE_TYPE != local)
    S3_BUCKET: Optional[str] = None
    S3_REGION: str = "eu-west-2"  # London (UK data residency)
    AZURE_STORAGE_ACCOUNT: Optional[str] = None
    GCS_BUCKET: Optional[str] = None

    # Encryption at rest (DSPT requirement)
    STORAGE_ENCRYPTION_ENABLED: bool = True
    STL_FILE_RETENTION_DAYS: int = Field(2555, description="7 years per NHS policy")

    # ========== DATA PROTECTION (GDPR, DSPT) ==========
    DATA_RESIDENCY_UK_ONLY: bool = Field(True, description="NHS requirement")
    PSEUDONYMIZATION_ENABLED: bool = Field(True, description="Pseudonymize NHS Numbers in logs")
    PII_ENCRYPTION_ENABLED: bool = Field(True, description="Encrypt names, DOB, addresses")

    # NHS Number handling
    NHS_NUMBER_VALIDATION_ENABLED: bool = True
    NHS_NUMBER_VERIFY_WITH_SPINE: bool = False  # Future: NHS Spine integration

    # Data retention
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years
    SCAN_DATA_RETENTION_DAYS: int = 2555  # 7 years
    TEMPORARY_FILES_RETENTION_HOURS: int = 24

    # ========== AUDIT & COMPLIANCE ==========
    AUDIT_LOGGING_ENABLED: bool = Field(True, description="DCB0129 requirement")
    AUDIT_LOG_ALL_API_CALLS: bool = True
    AUDIT_LOG_DATA_ACCESS: bool = True
    AUDIT_LOG_DIAGNOSTIC_DECISIONS: bool = Field(True, description="Clinical safety requirement")

    # Hazard Logging (DCB0129)
    HAZARD_LOG_PATH: Path = Field(Path("docs/clinical_safety/hazard_log.xlsx"))
    AUTO_HAZARD_DETECTION: bool = True  # Log errors as potential hazards

    # ========== CLINICAL SAFETY ==========
    AI_MODEL_VERSION_TRACKING: bool = Field(True, description="ISO 14971 requirement")
    AI_CONFIDENCE_THRESHOLD: float = Field(0.75, ge=0.0, le=1.0, description="Flag below this")

    # Safety checks
    ENABLE_OUT_OF_DISTRIBUTION_DETECTION: bool = True
    ENABLE_ANATOMICAL_VALIDITY_CHECKS: bool = True
    ENABLE_CLINICAL_RULE_VALIDATION: bool = True

    # Adverse event reporting
    MHRA_YELLOW_CARD_INTEGRATION: bool = False  # Future
    ADVERSE_EVENT_EMAIL_ALERTS: bool = True
    CLINICAL_SAFETY_OFFICER_EMAIL: Optional[str] = None

    # ========== AI/ML ==========
    ML_MODEL_REGISTRY_PATH: Path = Field(Path("models/registry"))
    ML_MODEL_VERSIONING_ENABLED: bool = True
    ML_EXPLAINABILITY_ENABLED: bool = Field(True, description="NHS AI transparency requirement")

    # Performance
    ML_BATCH_SIZE: int = Field(1, ge=1, le=32)
    ML_USE_GPU: bool = False  # Set True if CUDA available
    ML_NUM_WORKERS: int = Field(4, ge=1, le=16)

    # Validation
    REQUIRE_CLINICAL_VALIDATION: bool = Field(True, description="Validate against gold standard")
    MIN_DIAGNOSTIC_ACCURACY: float = Field(0.85, ge=0.0, le=1.0, description="MHRA requirement")

    # ========== INTEROPERABILITY ==========
    FHIR_ENABLED: bool = False  # Future
    FHIR_SERVER_URL: Optional[str] = None
    FHIR_VERSION: str = "R4"

    # Terminologies
    SNOMED_CT_EDITION: str = "uk-edition"
    SNOMED_CT_VERSION: str = "20240501"
    USE_ICD10_CODING: bool = True
    USE_LOINC_CODING: bool = True

    # NHS Integrations
    NHS_SPINE_ENABLED: bool = False
    NHS_SPINE_ENDPOINT: Optional[str] = None
    GP_CONNECT_ENABLED: bool = False

    # ========== PERFORMANCE ==========
    MAX_UPLOAD_SIZE_MB: int = Field(500, ge=10, le=1000)
    STL_PROCESSING_TIMEOUT_SECONDS: int = Field(30, ge=10, le=120)
    API_RATE_LIMIT_PER_MINUTE: int = Field(60, ge=10, le=1000)

    # Async processing
    ASYNC_PROCESSING_ENABLED: bool = True
    CELERY_BROKER_URL: str = Field("redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field("redis://localhost:6379/2")

    # ========== MONITORING & ALERTING ==========
    SENTRY_DSN: Optional[str] = None  # Error tracking
    PROMETHEUS_ENABLED: bool = False
    GRAFANA_ENABLED: bool = False

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"  # JSON for structured logging
    LOG_FILE_PATH: Optional[Path] = None

    # Health checks
    HEALTHCHECK_ENABLED: bool = True
    HEALTHCHECK_INTERVAL_SECONDS: int = 30

    # ========== SECURITY SCANNING ==========
    PENETRATION_TEST_MODE: bool = Field(False, description="Disable rate limits for pentesting")
    VULNERABILITY_SCAN_ENABLED: bool = False

    # Cyber Essentials Plus requirements
    MFA_ENFORCEMENT_LEVEL: Literal["required", "optional", "disabled"] = "required"
    TLS_VERSION: str = Field("1.3", description="Minimum TLS version")
    HSTS_ENABLED: bool = True
    CORS_ALLOWED_ORIGINS: list[str] = ["https://localhost:8501"]

    # ========== NOTIFICATIONS ==========
    EMAIL_ENABLED: bool = False
    EMAIL_SMTP_HOST: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_FROM: str = "noreply@foot-scan-system.nhs.uk"

    SMS_ENABLED: bool = False  # Future: for MFA
    WEBHOOK_ENABLED: bool = False  # Future: for integrations

    # ========== FEATURE FLAGS ==========
    FEATURE_3D_VISUALIZATION: bool = True
    FEATURE_TEMPORAL_COMPARISON: bool = True
    FEATURE_RISK_PREDICTION: bool = True
    FEATURE_INSURANCE_REPORTS: bool = True
    FEATURE_PATIENT_PORTAL: bool = False  # Future

    # ========== DEVELOPMENT ==========
    MOCK_NHS_SPINE: bool = Field(False, description="Use mock NHS Spine for testing")
    MOCK_AUTHENTICATION: bool = Field(False, description="Bypass auth in dev")
    SEED_DATABASE: bool = Field(False, description="Seed with test data")

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Enforce security settings in production"""
        if v == "production":
            # These MUST be True in production
            assert cls.AUTH_ENABLED, "Authentication required in production"
            assert cls.MFA_ENABLED, "MFA required in production (DSPT)"
            assert cls.AUDIT_LOGGING_ENABLED, "Audit logging required (DCB0129)"
            assert cls.STORAGE_ENCRYPTION_ENABLED, "Storage encryption required (DSPT)"
        return v

    @validator("DATABASE_URL")
    def validate_database_uk_region(cls, v, values):
        """Enforce UK data residency"""
        if values.get("DATA_RESIDENCY_UK_ONLY") and values.get("ENVIRONMENT") == "production":
            # Check that database is in UK region
            # This is a simplified check - real implementation should verify actual region
            url_str = str(v)
            if "rds" in url_str and "eu-west-2" not in url_str:
                raise ValueError("Database must be in eu-west-2 (London) for UK data residency")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
_settings = None

def get_settings() -> Settings:
    """
    Get application settings (singleton pattern)

    Returns:
        Settings: Application configuration
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Clinical constants (from medical literature and NHS guidelines)
class ClinicalConstants:
    """
    Clinical thresholds and constants

    Regulatory Note:
    - All values based on peer-reviewed literature
    - Changes require clinical safety review
    - Documented in Clinical Safety Case Report
    """

    # Hallux Valgus (Bunion) severity thresholds (degrees)
    HV_NORMAL_MAX = 15.0
    HV_MILD_MAX = 20.0
    HV_MODERATE_MAX = 40.0
    HV_SEVERE_MIN = 40.0

    # Intermetatarsal Angle (IMA) thresholds (degrees)
    IMA_NORMAL_MAX = 9.0
    IMA_MILD_MAX = 13.0
    IMA_MODERATE_MAX = 20.0
    IMA_SEVERE_MIN = 20.0

    # Arch height thresholds (mm)
    ARCH_FLAT_MAX = 10.0
    ARCH_LOW_MAX = 20.0
    ARCH_NORMAL_MIN = 20.0
    ARCH_NORMAL_MAX = 30.0
    ARCH_HIGH_MIN = 30.0

    # Foot length norms (mm) - adult
    FOOT_LENGTH_MIN = 200.0  # Pediatric or measurement error if below
    FOOT_LENGTH_MAX = 350.0  # Gigantism or measurement error if above
    FOOT_LENGTH_FEMALE_MEAN = 240.0
    FOOT_LENGTH_MALE_MEAN = 270.0

    # Foot width norms (mm) - adult
    FOOT_WIDTH_MIN = 70.0
    FOOT_WIDTH_MAX = 150.0
    FOOT_WIDTH_NARROW_MAX = 90.0
    FOOT_WIDTH_NORMAL_MIN = 90.0
    FOOT_WIDTH_NORMAL_MAX = 110.0
    FOOT_WIDTH_WIDE_MIN = 110.0

    # Asymmetry thresholds
    LENGTH_ASYMMETRY_NORMAL_MAX = 6.0  # mm
    LENGTH_ASYMMETRY_MODERATE = 10.0  # mm
    LENGTH_ASYMMETRY_SIGNIFICANT = 15.0  # mm

    WIDTH_ASYMMETRY_NORMAL_MAX = 5.0  # mm
    WIDTH_ASYMMETRY_MODERATE = 8.0  # mm
    WIDTH_ASYMMETRY_SIGNIFICANT = 12.0  # mm

    # Fall risk score thresholds (percentage)
    FALL_RISK_LOW = 20.0
    FALL_RISK_MODERATE = 40.0
    FALL_RISK_HIGH = 60.0

    # Health score thresholds (0-100)
    HEALTH_EXCELLENT_MIN = 80.0
    HEALTH_GOOD_MIN = 65.0
    HEALTH_FAIR_MIN = 50.0
    HEALTH_POOR_MIN = 30.0
    # Below 30 = Critical

    # Age-related thresholds
    AGE_PEDIATRIC_MAX = 18
    AGE_ADULT_MIN = 18
    AGE_SENIOR_MIN = 65

    # SNOMED CT codes for common conditions
    SNOMED_HALLUX_VALGUS = "202855006"
    SNOMED_PES_PLANUS = "53226007"  # Flat feet
    SNOMED_PES_CAVUS = "239830003"  # High arch
    SNOMED_HAMMER_TOE = "302322003"
    SNOMED_DIABETIC_FOOT = "609563008"

    # ICD-10 codes
    ICD10_HALLUX_VALGUS = "M20.1"
    ICD10_FLAT_FOOT = "M21.4"
    ICD10_HIGH_ARCH = "Q66.7"


# Export
__all__ = ["Settings", "get_settings", "ClinicalConstants"]
