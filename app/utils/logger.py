"""
Medical-Grade Structured Logging
GDPR-compliant logging with sensitive data masking

Compliance:
- GDPR: Pseudonymization of PII in logs
- DSPT: Secure logging practices
- DCB0129: Audit trail requirements
- NHS Digital: Clinical safety logging standards

Features:
- Structured JSON logging
- Automatic PII masking (NHS Numbers, names, DOB)
- Log rotation (7-year retention)
- Severity levels with clinical context
- Integration with audit service
"""

import logging
import logging.handlers
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

import structlog
from pythonjsonlogger import jsonlogger


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ClinicalLogLevel(Enum):
    """Clinical safety log levels (DCB0129)"""
    CLINICAL_SAFETY = "clinical_safety"
    CLINICAL_ERROR = "clinical_error"
    DIAGNOSTIC_DECISION = "diagnostic_decision"
    DATA_ACCESS = "data_access"
    SECURITY_EVENT = "security_event"


# Sensitive data patterns for masking
SENSITIVE_PATTERNS = {
    "nhs_number": re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+44|0)\s?(?:\d\s?){9,10}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b"),
    "date_of_birth": re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
}


def mask_sensitive_data(text: str, replacement: str = "***MASKED***") -> str:
    """
    Mask sensitive data in text

    Args:
        text: Input text that may contain sensitive data
        replacement: Replacement string for sensitive data

    Returns:
        Text with sensitive data masked

    Example:
        >>> mask_sensitive_data("NHS Number: 123-456-7890")
        "NHS Number: ***MASKED***"

        >>> mask_sensitive_data("Email: patient@example.com")
        "Email: ***MASKED***"
    """
    masked_text = text

    for pattern_name, pattern in SENSITIVE_PATTERNS.items():
        masked_text = pattern.sub(replacement, masked_text)

    return masked_text


def mask_nhs_number(nhs_number: str) -> str:
    """
    Mask NHS Number for logging (show only last 4 digits)

    Args:
        nhs_number: 10-digit NHS Number

    Returns:
        Masked NHS Number (e.g., "******7890")

    Example:
        >>> mask_nhs_number("1234567890")
        "******7890"
    """
    if not nhs_number:
        return ""

    # Remove spaces/hyphens
    nhs_clean = nhs_number.replace(" ", "").replace("-", "")

    if len(nhs_clean) == 10 and nhs_clean.isdigit():
        return f"******{nhs_clean[-4:]}"

    return "***INVALID***"


def mask_dict_values(data: Dict[str, Any], sensitive_keys: list = None) -> Dict[str, Any]:
    """
    Mask sensitive values in dictionary

    Args:
        data: Dictionary that may contain sensitive data
        sensitive_keys: List of keys to mask (default: common PII fields)

    Returns:
        Dictionary with sensitive values masked

    Example:
        >>> mask_dict_values({"name": "John Doe", "age": 45})
        {"name": "***MASKED***", "age": 45}
    """
    if sensitive_keys is None:
        sensitive_keys = [
            "name", "first_name", "last_name", "surname",
            "nhs_number", "dob", "date_of_birth",
            "email", "phone", "mobile", "address",
            "postcode", "postal_code", "password"
        ]

    masked_data = {}

    for key, value in data.items():
        if key.lower() in sensitive_keys:
            if "nhs" in key.lower() and isinstance(value, str):
                masked_data[key] = mask_nhs_number(value)
            else:
                masked_data[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked_data[key] = mask_dict_values(value, sensitive_keys)
        elif isinstance(value, str):
            masked_data[key] = mask_sensitive_data(value)
        else:
            masked_data[key] = value

    return masked_data


class MedicalJSONFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for medical-grade logging

    Adds:
    - Timestamp in ISO 8601 format
    - Severity level
    - Automatic PII masking
    - Clinical safety context
    """

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add severity
        log_record["severity"] = record.levelname

        # Add clinical context if available
        if hasattr(record, "clinical_context"):
            log_record["clinical_context"] = record.clinical_context

        # Add user context if available
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id

        # Add session context
        if hasattr(record, "session_id"):
            log_record["session_id"] = record.session_id

        # Mask sensitive data in message
        if "message" in log_record:
            log_record["message"] = mask_sensitive_data(str(log_record["message"]))

        # Mask any extra fields
        for key in list(log_record.keys()):
            if key not in ["timestamp", "severity", "message", "logger_name", "level"]:
                if isinstance(log_record[key], str):
                    log_record[key] = mask_sensitive_data(log_record[key])
                elif isinstance(log_record[key], dict):
                    log_record[key] = mask_dict_values(log_record[key])


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    app_name: str = "foot-scan-system",
    enable_console: bool = True,
    enable_file: bool = True,
    enable_audit: bool = True
) -> logging.Logger:
    """
    Setup medical-grade structured logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        app_name: Application name for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_audit: Enable separate audit log file

    Returns:
        Configured logger instance

    Example:
        logger = setup_logging(log_level="INFO", log_dir=Path("logs"))
        logger.info("System started", extra={"user_id": "USER-123"})
    """
    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()  # Clear existing handlers

    # JSON formatter
    formatter = MedicalJSONFormatter(
        fmt="%(timestamp)s %(severity)s %(name)s %(message)s"
    )

    # Console handler (for development)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation (7-year retention per NHS requirement)
    if enable_file:
        # Main application log
        app_log_file = log_dir / f"{app_name}.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=app_log_file,
            when="midnight",
            interval=1,
            backupCount=365 * 7,  # 7 years
            encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Error log (separate file for critical issues)
        error_log_file = log_dir / f"{app_name}_error.log"
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=error_log_file,
            when="midnight",
            interval=1,
            backupCount=365 * 7,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    # Audit log (DCB0129 requirement - separate from application logs)
    if enable_audit:
        audit_log_file = log_dir / f"{app_name}_audit.log"
        audit_handler = logging.handlers.TimedRotatingFileHandler(
            filename=audit_log_file,
            when="midnight",
            interval=1,
            backupCount=365 * 7,  # 7 years minimum retention
            encoding="utf-8"
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(formatter)

        # Create separate audit logger
        audit_logger = logging.getLogger(f"{app_name}.audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(audit_handler)

    return logger


def get_logger(name: str = "foot-scan-system") -> logging.Logger:
    """
    Get configured logger instance

    Args:
        name: Logger name (default: foot-scan-system)

    Returns:
        Configured logger

    Example:
        logger = get_logger(__name__)
        logger.info("Processing scan", extra={"scan_id": "SCAN-123"})
    """
    logger = logging.getLogger(name)

    # If logger not yet configured, set up with defaults
    if not logger.handlers:
        setup_logging(app_name=name)

    return logger


def log_clinical_event(
    logger: logging.Logger,
    event_type: ClinicalLogLevel,
    message: str,
    scan_id: Optional[str] = None,
    patient_id_pseudonym: Optional[str] = None,
    user_id: Optional[str] = None,
    diagnosis: Optional[str] = None,
    confidence: Optional[float] = None,
    ai_model_version: Optional[str] = None,
    **kwargs
):
    """
    Log clinical safety event (DCB0129 compliant)

    Args:
        logger: Logger instance
        event_type: Type of clinical event
        message: Log message
        scan_id: Scan identifier
        patient_id_pseudonym: Pseudonymized patient ID (never plain NHS Number)
        user_id: User who triggered event
        diagnosis: Diagnosis name (if applicable)
        confidence: Diagnostic confidence score
        ai_model_version: AI model version used
        **kwargs: Additional context

    Example:
        log_clinical_event(
            logger=logger,
            event_type=ClinicalLogLevel.DIAGNOSTIC_DECISION,
            message="Hallux valgus diagnosed",
            scan_id="SCAN-123",
            diagnosis="Hallux Valgus - Moderate",
            confidence=0.92,
            ai_model_version="2.0.0"
        )
    """
    extra = {
        "event_type": event_type.value,
        "clinical_context": True,
        "scan_id": scan_id,
        "patient_id_pseudonym": patient_id_pseudonym,
        "user_id": user_id,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "ai_model_version": ai_model_version,
        **kwargs
    }

    # Remove None values
    extra = {k: v for k, v in extra.items() if v is not None}

    # Log at appropriate level
    if event_type in [ClinicalLogLevel.CLINICAL_ERROR, ClinicalLogLevel.SECURITY_EVENT]:
        logger.error(message, extra=extra)
    elif event_type == ClinicalLogLevel.CLINICAL_SAFETY:
        logger.warning(message, extra=extra)
    else:
        logger.info(message, extra=extra)


# Setup structlog for enhanced structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


# Export
__all__ = [
    "setup_logging",
    "get_logger",
    "mask_sensitive_data",
    "mask_nhs_number",
    "mask_dict_values",
    "log_clinical_event",
    "LogLevel",
    "ClinicalLogLevel",
    "MedicalJSONFormatter"
]
