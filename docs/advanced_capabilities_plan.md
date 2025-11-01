# Advanced Capabilities Implementation Plan

This document captures the concrete implementation pathway for the seven advanced capabilities requested for the Enhanced Foot Scan System. Each section summarises the current state, outlines phased engineering tasks, lists external assets that must be sourced, and highlights the groundwork we can complete with existing resources.

## Segmentation Fidelity (PointNet++ Anatomical Labelling)

- **Current state:** The preprocessing pipeline (`src/preprocessing/point_cloud_processor.py`) can ingest a segmentation model via `segmentation_model_path`, but the app currently defaults to heuristic segment mocks when no trained weights are provided.
- **Implementation steps:**
  1. Assemble a training corpus of bilateral STL/PLY scans with per-point anatomical labels covering diverse cohorts.
  2. Fine-tune a PointNet++ (or similar point-cloud network) to the 22-region schema already referenced in `features/comprehensive_condition_detection.py`.
  3. Export the trained network to ONNX, validate inference parity, and register the path in configuration.
  4. Extend `PointCloudProcessor.run_segmentation` to load metadata about model provenance (training cohort, IoU metrics) for audit logging.
  5. Add automated validation scripts that compare segmentation output against manual annotations on a held-out benchmark set and write IoU/F1 metrics to `output/quality_reports/segmentation.json`.
- **External assets required:** Annotated point-cloud dataset; GPU training environment; model weights packaged as ONNX; benchmark annotations.
- **Work we can do now:** Maintain the ONNX loading shim (`src/segmentation/pointnet_segmenter.py`) and ensure the pipeline gracefully falls back when assets are missing. No additional coding work is possible without the labelled dataset and trained weights.

## Research-Linked Rule Base (Evidence Normalisation)

- **Current state:** `data/foot_conditions_knowledge_base.json` stores raw literature extracts but lacks structured thresholds, cohort metadata, or citation-level scoring.
- **Implementation steps:**
  1. Design a normalised schema (e.g., SQLite or JSON Lines) that stores condition, metric, cohort descriptors, threshold ranges, confidence scores, and citation identifiers.
  2. Build an ETL script to convert the existing 44k-study corpus into the structured schema, extracting quantitative ranges via NLP templates.
  3. Implement a `RuleEvidenceStore` service that surfaces cohort-aware thresholds to diagnostic modules (`features/*.py`) with provenance metadata.
  4. Integrate citation rendering into Streamlit panels so clinicians can inspect the evidence trail per finding.
- **External assets required:** Structured extractions from the literature corpus (manual curation or semi-automated NLP tooling); citation quality ratings; storage for the normalised dataset.
- **Work we can do now:** Draft the schema and interface contracts inside `src/knowledge_base/` (new package). Population of the store awaits curated data.

## Early-Warning Analytics (Longitudinal Drift Detection)

- **Current state:** `src/analysis/early_warning.py` computes z-scores and trend signals using the processed scans database, but the UI does not expose these insights.
- **Implementation steps:**
  1. Extend the clinical database logging (already writing to `output/clinical_records.db`) to ensure all relevant metrics are captured per scan.
  2. Integrate `EarlyWarningAnalytics` into the Risk Assessment tab, displaying metric-level z-scores, percentile drift, and trend commentary for each patient with ≥2 scans.
  3. Add CSV/JSON export of early-warning signals for interoperability with electronic health records.
  4. Backfill tests in `tests/test_early_warning.py` to validate edge cases (insufficient data, constant values, high variance).
- **External assets required:** None; operates on existing processed scans.
- **Work we can do now:** Implement UI integration and exports (see code changes in this update).

## Clinical Reporting (SOAP + ICD-10 Packages)

- **Current state:** The system generates rich on-screen diagnostics but lacks a formal clinical summary export that mirrors podiatry documentation standards.
- **Implementation steps:**
  1. Define a report assembly pipeline that ingests processed scan payloads, condition findings, risk matrix results, and early-warning signals.
  2. Create templates for SOAP notes and ICD-10 summaries (likely stored under `templates/reports/`).
  3. Implement PDF/JSON export functions in `src/reporting/clinical_reporter.py`, embedding segmentation snapshots and evidence citations when available.
  4. Surface a “Download Clinical Report” action within the Enhanced Analysis tab, with audit logging to `output/reports/`.
- **External assets required:** ICD-10 mapping table for foot conditions; design template assets for PDF generation (fonts, logos).
- **Work we can do now:** Build the core report generator and JSON export; PDF layout will need approved branding assets.

## Model Architecture (Ensemble + Explainability)

- **Current state:** The ensemble predictor (`src/ml/ensemble_predictor.py`) is implemented and trains from tabular features but does not yet combine point-cloud CNN outputs or provide SHAP-based explanations.
- **Implementation steps:**
  1. Develop a point-cloud CNN (PointNet/EdgeConv) that ingests segmented geometry and outputs condition probabilities; save models alongside tabular ensemble weights.
  2. Implement an orchestration layer that blends tabular and point-cloud predictions via calibrated stacking.
  3. Integrate SHAP (or Captum) explainability, caching feature attributions for both model families.
  4. Expand training scripts to record calibration curves, decision thresholds, and uncertainty metrics.
- **External assets required:** Large, labelled point-cloud dataset; GPU resources for CNN training; SHAP background dataset derived from clinical scans.
- **Work we can do now:** Prepare the orchestration scaffolding and ensure the training agent (`src/ml/continuous_trainer.py`) can register new model artefacts when they become available.

## Temporal Modelling (Progression Forecasting)

- **Current state:** Longitudinal metrics are stored, but no sequence model is training against them.
- **Implementation steps:**
  1. Define feature engineering utilities that convert patient scan histories into time-series tensors with aligned measurement intervals.
  2. Implement an LSTM or Temporal Fusion Transformer pipeline under `src/ml/temporal/`, including training, evaluation, and serialization routines.
  3. Hook the temporal forecaster into the Enhanced Analysis dashboard to display projected decline trajectories and confidence intervals.
  4. Update the training agent to schedule temporal model retraining once sufficient longitudinal data per patient exists.
- **External assets required:** Longitudinal labelled datasets with ≥4 scans per patient; GPU acceleration for sequence model training.
- **Work we can do now:** Build feature engineering scaffolds and placeholder inference code that gracefully exits when trained weights are absent.

## Risk Matrix (Unified Scoring and Alerts)

- **Current state:** `src/analysis/risk_matrix.py` offers a baseline scoring algorithm that blends health scores, condition severity, and risk probabilities, but UI feedback is limited.
- **Implementation steps:**
  1. Expand the risk matrix logic to include uncertainty parameters from ensemble models and temporal forecasts.
  2. Introduce configurable policy thresholds via YAML/JSON so clinical teams can tune risk categorisation.
  3. Create dashboard components that visualise driver contributions and recommended interventions.
  4. Emit structured alerts (e.g., FHIR `RiskAssessment`) for integration with external systems.
- **External assets required:** Clinical sign-off on scoring rubric; integration specifications for downstream alerting systems.
- **Work we can do now:** Enhance the UI presentation and logging of risk matrix outputs (addressed in this update) and add configuration support.

