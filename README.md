# Foot Scan System Dashboard

A comprehensive medical-grade foot scanning and analysis system with AI-powered risk assessment, clinical recommendations, and insurance reporting.

## Live Demo

Deployed on Streamlit Cloud: [Coming Soon]

## System Architecture

```
foot-scan-system/
├── data/               # Data storage
│   ├── raw/           # Raw Volumental OBJ/JSON files
│   ├── processed/     # Processed point clouds
│   └── models/        # Trained ML models
├── src/               # Source code
│   ├── ingestion/     # Data loading modules
│   ├── preprocessing/ # Point cloud processing
│   ├── models/        # PointNet architecture
│   ├── features/      # Anatomical feature detection
│   ├── matching/      # Last matching algorithms
│   ├── printing/      # 3D print generation
│   └── api/          # REST API endpoints
├── tests/            # Test scripts
├── docs/             # Documentation
├── config/           # Configuration files
├── output/           # Generated results
└── web/             # Web interface
```

## Key Features

### Clinical Analysis
- **AI Risk Assessment**: Multi-model ensemble for predicting foot conditions
- **Medical Research Integration**: Evidence-based diagnosis using 100+ clinical studies
- **Condition Detection**: Identifies bunions, hammertoes, pes planus, pes cavus, plantar fasciitis, and more
- **Temporal Analysis**: Track progression of conditions over time
- **Early Warning System**: Predictive alerts for developing conditions

### Professional Reporting
- **Insurance Reports**: ICD-10 coded reports with cost estimates
- **Clinical Summaries**: Comprehensive patient reports with visualizations
- **Last Matching**: AI-powered shoe last recommendations with medical accommodations
- **3D Visualization**: Interactive foot scan viewing and measurements

### Data Management
- **Patient Records**: SQLite-based patient history tracking
- **Baseline Comparisons**: Compare against healthy population baselines
- **Export Capabilities**: PDF reports, CSV data exports
- **Secure Storage**: HIPAA-compliant data handling

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn (Random Forest, XGBoost, SVM ensemble)
- **3D Processing**: Trimesh
- **Computer Vision**: OpenCV
- **Database**: SQLite with SQLAlchemy
- **Data Science**: Pandas, NumPy, SciPy
- **Visualization**: Plotly

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/[your-username]/foot-scan-system.git
cd foot-scan-system

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Streamlit Cloud Deployment

This app is configured for one-click deployment to Streamlit Cloud:

1. Fork this repository
2. Sign up at https://share.streamlit.io
3. Connect your GitHub account
4. Select this repository
5. Deploy!

## System Requirements

- Python 3.10+
- 4GB RAM minimum
- Modern web browser (Chrome, Firefox, Safari recommended)

## Medical Disclaimer

This system is designed as a clinical decision support tool and should not replace professional medical diagnosis. Always consult with qualified healthcare providers for medical advice.