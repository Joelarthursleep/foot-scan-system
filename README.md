# Foot Scan to Custom Last System

An automated system for converting Volumental LiDAR foot scans into customized shoe lasts with 3D-printed composite material additions for anatomical variations.

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

- **Automatic Segmentation**: 22-region foot segmentation using modified PointNet
- **Anatomical Detection**: Identifies bunions, high insteps, arch variations
- **Last Matching**: Intelligent matching to base last inventory
- **3D Print Generation**: Creates TPU buildup instructions for customization
- **Complete Pipeline**: End-to-end from scan to production-ready files

## Technology Stack

- **Deep Learning**: TensorFlow/Keras with PointNet architecture
- **3D Processing**: Open3D, Trimesh, PyMesh
- **Database**: PostgreSQL with PostGIS
- **API**: FastAPI
- **3D Printing**: Custom G-code generation

## Performance Targets

- Scan processing: <30 seconds
- Segmentation accuracy: >85% IoU
- Size recommendation: 90% first-time fit success
- G-code generation: <2 minutes

## Installation

See `docs/installation.md` for detailed setup instructions.

## Usage

1. Upload Volumental scan files (OBJ + JSON)
2. System performs automatic analysis
3. Review recommendations and customizations
4. Generate 3D printing files
5. Print TPU modifications on base last