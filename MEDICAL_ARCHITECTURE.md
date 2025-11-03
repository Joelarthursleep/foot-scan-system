# Medical-Grade Foot Scan System Architecture

## Version 2.0 - NHS Compliance Ready
**Date**: November 3, 2025
**Status**: Development
**Regulatory Target**: MHRA Class IIa, UKCA Marking, NHS DTAC Compliant

---

## Architecture Principles

### 1. Regulatory Compliance (DCB0129, DCB0160, DTAC)
- **Audit Everything**: Every action, every data access, every diagnostic decision
- **Traceable**: Full version control of AI models, code, and clinical rules
- **Safe**: Clinical Safety Officer review points, hazard logging, risk management
- **Validated**: Clinical validation data tracked and reportable

### 2. Security by Design (DSPT, Cyber Essentials Plus)
- **Encryption**: At-rest (AES-256), in-transit (TLS 1.3+)
- **Authentication**: Multi-factor, role-based access control (RBAC)
- **Privacy**: Data pseudonymization, NHS Number handling, GDPR compliance
- **Monitoring**: Real-time security event detection and alerting

### 3. Performance & Scalability
- **Fast**: <5 second scan processing (vs 30+ seconds currently)
- **Concurrent**: Handle 100+ simultaneous scans
- **Optimized**: Async processing, caching, database indexing
- **Scalable**: Horizontal scaling ready for NHS-wide deployment

### 4. Interoperability (HL7 FHIR, SNOMED CT)
- **Standards-Based**: FHIR R4 DiagnosticReport, Observation resources
- **Coded**: SNOMED CT for conditions, ICD-10 for diagnoses
- **Integrable**: RESTful API, webhook support, NHS Spine connectivity

### 5. Clinical Safety (ISO 14971, IEC 62304)
- **Risk Managed**: Hazard analysis, risk mitigation, safety testing
- **Validated**: Clinical validation against gold standard
- **Monitored**: Post-market surveillance, adverse event reporting
- **Explainable**: AI transparency, clinical justification for all diagnoses

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Streamlit  │  │   REST API   │  │  FHIR API    │         │
│  │   Web UI     │  │   (FastAPI)  │  │   (HAPI)     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│                    APPLICATION LAYER                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │            Audit & Logging Middleware                   │    │
│  │  (Every request/response logged with user context)     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Clinical    │  │  Diagnostic  │  │  Patient     │         │
│  │  Workflow    │  │  Engine      │  │  Management  │         │
│  │  Service     │  │  (AI/ML)     │  │  Service     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│                      DOMAIN LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  STL         │  │  AI Model    │  │  Clinical    │         │
│  │  Processor   │  │  Repository  │  │  Rules       │         │
│  │              │  │  (Versioned) │  │  Engine      │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼────────────────┐
│                  INFRASTRUCTURE LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  PostgreSQL  │  │  Redis       │  │  Object      │         │
│  │  (Clinical   │  │  (Cache +    │  │  Storage     │         │
│  │   Records)   │  │   Queue)     │  │  (STL Files) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Audit Log   │  │  Security    │  │  Monitoring  │         │
│  │  Store       │  │  Vault       │  │  (Metrics)   │         │
│  │  (Append-    │  │  (Secrets)   │  │              │         │
│  │   only)      │  │              │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
foot-scan-system/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entry point
│   ├── streamlit_app.py            # Streamlit UI (refactored)
│   │
│   ├── api/                         # REST API Layer
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── scans.py            # Scan upload/processing endpoints
│   │   │   ├── patients.py         # Patient management
│   │   │   ├── reports.py          # Diagnostic reports
│   │   │   └── admin.py            # Admin/configuration
│   │   ├── middleware/
│   │   │   ├── audit.py            # Audit logging middleware
│   │   │   ├── auth.py             # Authentication/authorization
│   │   │   └── rate_limit.py       # Rate limiting
│   │   └── schemas/                # Pydantic models
│   │       ├── scan.py
│   │       ├── patient.py
│   │       └── report.py
│   │
│   ├── core/                        # Core Business Logic
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   ├── security.py             # Encryption, hashing, key management
│   │   ├── exceptions.py           # Custom exceptions
│   │   └── constants.py            # Clinical constants, thresholds
│   │
│   ├── services/                    # Application Services
│   │   ├── __init__.py
│   │   ├── scan_service.py         # Scan processing orchestration
│   │   ├── diagnostic_service.py   # AI diagnostic engine
│   │   ├── patient_service.py      # Patient data management
│   │   ├── report_service.py       # Report generation
│   │   ├── audit_service.py        # Audit logging
│   │   └── notification_service.py # Alerts/notifications
│   │
│   ├── domain/                      # Domain Models & Logic
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── patient.py          # Patient entity
│   │   │   ├── scan.py             # Scan entity
│   │   │   ├── diagnosis.py        # Diagnosis entity
│   │   │   └── audit.py            # Audit log entity
│   │   ├── repositories/
│   │   │   ├── patient_repo.py
│   │   │   ├── scan_repo.py
│   │   │   └── audit_repo.py
│   │   └── value_objects/
│   │       ├── nhs_number.py       # NHS Number validation
│   │       ├── snomed_code.py      # SNOMED CT code
│   │       └── icd10_code.py       # ICD-10 code
│   │
│   ├── ml/                          # Machine Learning Pipeline
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── condition_detector.py
│   │   │   ├── risk_predictor.py
│   │   │   └── model_registry.py   # Versioned model tracking
│   │   ├── preprocessing/
│   │   │   ├── stl_processor.py    # Optimized STL loading
│   │   │   └── feature_extractor.py
│   │   ├── validation/
│   │   │   ├── clinical_validator.py
│   │   │   └── performance_metrics.py
│   │   └── explainability/
│   │       ├── shap_explainer.py
│   │       └── feature_importance.py
│   │
│   ├── clinical/                    # Clinical Safety & Rules
│   │   ├── __init__.py
│   │   ├── safety/
│   │   │   ├── hazard_log.py       # DCB0129 hazard logging
│   │   │   ├── risk_assessment.py  # ISO 14971 risk management
│   │   │   └── incident_reporting.py
│   │   ├── rules/
│   │   │   ├── diagnostic_rules.py # Clinical decision rules
│   │   │   ├── safety_checks.py    # Safety constraints
│   │   │   └── contraindications.py
│   │   └── validation/
│   │       ├── clinical_validation.py
│   │       └── ground_truth.py     # Gold standard comparison
│   │
│   ├── interop/                     # Interoperability Layer
│   │   ├── __init__.py
│   │   ├── fhir/
│   │   │   ├── builders/
│   │   │   │   ├── diagnostic_report.py
│   │   │   │   ├── observation.py
│   │   │   │   └── patient.py
│   │   │   └── serializers/
│   │   │       └── fhir_serializer.py
│   │   ├── terminologies/
│   │   │   ├── snomed_mapper.py    # Map conditions to SNOMED CT
│   │   │   ├── icd10_mapper.py     # Map to ICD-10
│   │   │   └── loinc_mapper.py     # Map observations to LOINC
│   │   └── integrations/
│   │       ├── nhs_spine.py        # NHS Spine integration
│   │       └── gp_connect.py       # GP Connect integration
│   │
│   ├── quality/                     # Quality Management System
│   │   ├── __init__.py
│   │   ├── documents/
│   │   │   ├── clinical_safety_case.py
│   │   │   ├── risk_management_file.py
│   │   │   └── validation_report.py
│   │   ├── change_control/
│   │   │   ├── change_request.py
│   │   │   └── version_control.py
│   │   └── metrics/
│   │       ├── quality_metrics.py
│   │       └── kpi_tracker.py
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logger.py               # Structured logging
│       ├── validators.py           # Input validation
│       └── formatters.py           # Data formatting
│
├── infrastructure/                  # Infrastructure as Code
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.worker
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── database.tf
│   │   └── storage.tf
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana-dashboard.json
│
├── tests/                           # Comprehensive Testing
│   ├── unit/
│   ├── integration/
│   ├── clinical_validation/
│   └── security/
│
├── docs/                            # QMS Documentation
│   ├── clinical_safety/
│   │   ├── hazard_log.xlsx
│   │   ├── clinical_safety_case.pdf
│   │   └── risk_management_plan.pdf
│   ├── technical/
│   │   ├── architecture.md
│   │   ├── api_documentation.md
│   │   └── deployment_guide.md
│   └── regulatory/
│       ├── dcb0129_compliance.md
│       ├── dtac_evidence.md
│       └── mhra_submission.md
│
├── alembic/                         # Database Migrations
│   └── versions/
│
├── .env.example                     # Environment variables template
├── pyproject.toml                   # Poetry dependency management
├── pytest.ini                       # Test configuration
└── README.md                        # System overview
```

---

## Key Components Detail

### 1. Audit Logging System

**Every action logged with**:
- User ID and role
- NHS Number (pseudonymized)
- Action type (create, read, update, delete, diagnose)
- Timestamp (ISO 8601, UTC)
- IP address
- Session ID
- AI model version used
- Clinical decision rationale
- Data accessed/modified

**Retention**: 7 years minimum (NHS Records Management Code of Practice)

**Storage**: Write-once, append-only PostgreSQL table + periodic export to immutable object storage

---

### 2. Clinical Safety Features

**Hazard Detection**:
- Out-of-distribution scan detection (reject if not similar to training data)
- Confidence thresholds (flag low-confidence diagnoses for manual review)
- Anatomical validity checks (e.g., foot can't be 50cm long)
- Model version tracking (every diagnosis linked to specific AI model version)

**Safety Alerts**:
- Critical findings (e.g., diabetic foot ulcer) → immediate clinician notification
- System errors → logged to hazard log
- Adverse events → MHRA yellow card reporting workflow

---

### 3. Performance Optimizations

**STL Processing** (Target: <3 seconds):
```python
# Async loading with Rust-powered numpy-stl
# Parallel processing of left/right feet
# Cached feature extraction
# GPU-accelerated mesh operations (optional)
```

**Database**:
- Indexed queries (patient_id, nhs_number, scan_date)
- Connection pooling (SQLAlchemy async)
- Read replicas for reporting queries
- Partitioned tables by date (monthly partitions)

**Caching**:
- Redis for:
  - Session data
  - Recent scan results
  - Frequently accessed reports
  - AI model predictions (with cache invalidation on model update)

---

### 4. Security Implementation

**Encryption**:
```python
# At-rest: AES-256-GCM
# STL files encrypted in object storage
# Database column-level encryption for NHS Numbers, DOB, patient names

# In-transit: TLS 1.3
# All API calls over HTTPS
# Certificate pinning for NHS Spine connections
```

**Authentication**:
- OAuth 2.0 / OpenID Connect
- Multi-factor authentication (TOTP, SMS, biometric)
- NHS Care Identity Service integration (future)

**Authorization** (RBAC):
```python
Roles:
- Administrator (full access)
- Clinician (diagnose, view all patients)
- Radiographer (upload scans only)
- Audit_Viewer (read-only access to audit logs)
- Patient (view own records only)
```

---

### 5. FHIR Interoperability

**DiagnosticReport Resource**:
```json
{
  "resourceType": "DiagnosticReport",
  "id": "foot-scan-001",
  "meta": {
    "versionId": "1",
    "lastUpdated": "2025-11-03T10:00:00Z",
    "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"]
  },
  "identifier": [{
    "system": "https://foot-scan-system.nhs.uk/scan-id",
    "value": "FS-20251103-001"
  }],
  "status": "final",
  "category": [{
    "coding": [{
      "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
      "code": "RAD",
      "display": "Radiology"
    }]
  }],
  "code": {
    "coding": [{
      "system": "http://snomed.info/sct",
      "code": "241667001",
      "display": "3D foot scan"
    }]
  },
  "subject": {
    "reference": "Patient/nhs-9876543210",
    "identifier": {
      "system": "https://fhir.nhs.uk/Id/nhs-number",
      "value": "9876543210"
    }
  },
  "effectiveDateTime": "2025-11-03T09:45:00Z",
  "issued": "2025-11-03T10:00:00Z",
  "performer": [{
    "reference": "Practitioner/DR-12345",
    "display": "Dr. Jane Smith"
  }],
  "result": [
    {
      "reference": "Observation/hallux-valgus-left"
    },
    {
      "reference": "Observation/hallux-valgus-right"
    }
  ],
  "conclusion": "Bilateral severe hallux valgus identified. Surgical consultation recommended.",
  "conclusionCode": [{
    "coding": [{
      "system": "http://snomed.info/sct",
      "code": "202855006",
      "display": "Hallux valgus"
    }]
  }]
}
```

---

## Deployment Strategy

### Phase 1: Localhost Development (Current)
- SQLite → PostgreSQL migration
- Environment configuration system
- Comprehensive testing suite
- Security hardening

### Phase 2: Staging Environment (Pre-Production)
- Docker containerization
- Kubernetes deployment
- Load testing (1000+ concurrent users)
- Penetration testing
- Clinical validation study

### Phase 3: NHS Production (Cloud Hosted)
- UK data residency (AWS eu-west-2 or Azure UK South)
- HSCN connectivity
- Disaster recovery (RPO: 4 hours, RTO: 8 hours)
- 24/7 monitoring and alerting
- Post-market surveillance

---

## Compliance Checklist

### MHRA (UKCA Marking)
- [x] Device classification determined (Class IIa)
- [ ] UK Approved Body selected
- [ ] Technical documentation prepared
- [ ] Risk management file (ISO 14971)
- [ ] Clinical evaluation report
- [ ] UKCA mark affixed to software

### DCB0129 (Manufacturer)
- [ ] Clinical Safety Officer appointed
- [ ] Clinical Safety Case Report created
- [ ] Hazard Log maintained
- [ ] Safety incident reporting process
- [ ] Version control and change management

### DCB0160 (Deployment)
- [ ] Deployment safety case
- [ ] Local risk assessment
- [ ] Training materials for NHS staff

### DTAC (NHS Digital Technology Assessment Criteria)
- [ ] Clinical safety compliance evidenced
- [ ] Data protection (DSPT "Standards Met")
- [ ] Technical security (Cyber Essentials Plus)
- [ ] Interoperability (FHIR R4)
- [ ] Usability/accessibility (WCAG 2.1 AA)

### ISO 13485 (Quality Management)
- [ ] Quality Manual
- [ ] Document control procedures
- [ ] Design and development procedures
- [ ] Verification and validation protocols
- [ ] CAPA (Corrective and Preventive Actions)

---

## Performance Targets

| Metric | Current | Target | NHS Requirement |
|--------|---------|--------|----------------|
| Scan Processing Time | 30s | <5s | <10s |
| Diagnostic Accuracy | 85% | >90% | >85% |
| System Uptime | 95% | 99.9% | 99.5% |
| Concurrent Users | 10 | 1000+ | 100+ per trust |
| API Response Time | 2s | <500ms | <1s |
| Database Query Time | 500ms | <100ms | <200ms |
| STL File Upload | 30s | <10s | <20s |

---

## Security Standards

| Standard | Requirement | Status |
|----------|-------------|--------|
| Cyber Essentials Plus | Mandatory for NHS | To implement |
| ISO 27001 | Recommended | Future |
| DSPT | Standards Met | To implement |
| Penetration Testing | Annual | Not started |
| Vulnerability Scanning | Continuous | To implement |
| GDPR | Compliant | Partial |
| NHS Data Security Standards | 10 standards | To implement |

---

## Next Steps

1. **Immediate** (This Sprint):
   - Refactor current Streamlit app to new architecture
   - Implement audit logging system
   - Add encryption for sensitive data
   - Create configuration management system

2. **Short-term** (Next 30 days):
   - FastAPI REST API
   - PostgreSQL migration
   - Redis caching layer
   - Comprehensive test suite
   - Docker containerization

3. **Medium-term** (Next 90 days):
   - FHIR API implementation
   - Clinical safety features
   - Quality management documentation
   - Penetration testing
   - Performance optimization

4. **Long-term** (6-12 months):
   - Clinical validation study
   - UKCA marking process
   - DTAC compliance
   - NHS pilot deployment

---

**Document Version**: 1.0
**Last Updated**: 2025-11-03
**Owner**: Clinical Safety Officer (To be appointed)
**Review Date**: 2025-12-03
