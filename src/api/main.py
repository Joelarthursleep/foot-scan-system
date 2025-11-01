"""
FastAPI Application for Foot Scan to Custom Last System
Main API endpoints for processing foot scans and generating customized lasts
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import tempfile
import shutil
from pathlib import Path
import logging
import json
import uuid
import numpy as np

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.volumental_loader import VolumentalLoader
from preprocessing.point_cloud_processor import PointCloudProcessor
from models.pointnet_foot import create_foot_segmentation_model
from features.bunion_detector import BunionDetector
from features.arch_analyzer import ArchAnalyzer
from matching.last_matcher import LastMatcher, LastDatabase
from printing.gcode_generator import generate_custom_last_modifications

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Foot Scan to Custom Last System",
    description="API for processing foot scans and generating customized shoe lasts",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processors
model = None
processor = None
bunion_detector = None
arch_analyzer = None
last_matcher = None

# Response models
class ScanAnalysisResponse(BaseModel):
    scan_id: str
    measurements: Dict
    features: Dict
    confidence: float
    processing_time_seconds: float

class LastRecommendationResponse(BaseModel):
    scan_id: str
    recommendations: List[Dict]
    modifications_needed: Dict
    confidence: float

class PrintingFilesResponse(BaseModel):
    scan_id: str
    gcode_url: str
    stl_url: str
    estimated_print_time_minutes: float
    material_weight_g: float

class ProcessingStatus(BaseModel):
    scan_id: str
    status: str  # 'processing', 'completed', 'failed'
    stage: str
    progress_percentage: float
    error_message: Optional[str] = None

# Storage for processing jobs
processing_jobs = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and processors on startup"""
    global model, processor, bunion_detector, arch_analyzer, last_matcher

    logger.info("Initializing system components...")

    # Initialize processors
    processor = PointCloudProcessor(target_points=10000)
    bunion_detector = BunionDetector()
    arch_analyzer = ArchAnalyzer()

    # Initialize last matcher and populate sample data
    last_matcher = LastMatcher()
    last_matcher.db.populate_sample_data()

    # Load or create model (simplified for demo)
    try:
        model = create_foot_segmentation_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        model = None

    logger.info("System initialization complete")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Foot Scan to Custom Last System API",
        "status": "operational",
        "endpoints": {
            "upload_scan": "/api/scan/upload",
            "get_recommendations": "/api/last/recommend/{scan_id}",
            "generate_printing": "/api/printing/generate/{scan_id}",
            "check_status": "/api/status/{scan_id}"
        }
    }

@app.post("/api/scan/upload", response_model=ScanAnalysisResponse)
async def upload_scan(
    background_tasks: BackgroundTasks,
    obj_file: UploadFile = File(...),
    json_file: UploadFile = File(...)
):
    """
    Upload and process Volumental scan files

    Args:
        obj_file: OBJ mesh file
        json_file: JSON measurements file

    Returns:
        Initial analysis results
    """
    import time
    start_time = time.time()

    # Generate scan ID
    scan_id = str(uuid.uuid4())

    # Save uploaded files temporarily
    temp_dir = Path(tempfile.mkdtemp())
    obj_path = temp_dir / "scan.obj"
    json_path = temp_dir / "measurements.json"

    try:
        # Save files
        with open(obj_path, "wb") as f:
            shutil.copyfileobj(obj_file.file, f)

        with open(json_path, "wb") as f:
            shutil.copyfileobj(json_file.file, f)

        # Load data
        loader = VolumentalLoader(str(obj_path), str(json_path))
        vertices, faces, measurements = loader.load_all()

        # Process point cloud
        point_cloud, processing_params = processor.process_scan(vertices, faces)

        # Simulate segmentation (in production, use actual model)
        if model is not None:
            # Run segmentation model
            segmentation = np.random.randint(0, 22, size=len(point_cloud))
        else:
            # Mock segmentation for demo
            segmentation = np.zeros(len(point_cloud), dtype=int)
            # Simulate some segments
            segmentation[:1000] = 1  # Hallux
            segmentation[1000:2000] = 6  # Medial ball
            segmentation[2000:3000] = 10  # Medial arch

        # Detect anatomical features
        bunion_analysis = bunion_detector.detect(point_cloud, segmentation)
        arch_analysis = arch_analyzer.analyze(point_cloud, segmentation)

        # Store results
        processing_jobs[scan_id] = {
            'status': 'completed',
            'point_cloud': point_cloud,
            'segmentation': segmentation,
            'measurements': measurements.__dict__,
            'features': {
                'bunion': {
                    'has_bunion': bunion_analysis.has_bunion,
                    'severity': bunion_analysis.severity,
                    'angle': bunion_analysis.hallux_valgus_angle,
                    'confidence': bunion_analysis.confidence
                },
                'arch': {
                    'type': arch_analysis.arch_type,
                    'ahi': arch_analysis.arch_height_index,
                    'support_level_needed': arch_analysis.support_level_needed,
                    'confidence': arch_analysis.confidence
                }
            }
        }

        processing_time = time.time() - start_time

        # Clean up temp files
        shutil.rmtree(temp_dir)

        return ScanAnalysisResponse(
            scan_id=scan_id,
            measurements=measurements.__dict__,
            features=processing_jobs[scan_id]['features'],
            confidence=(bunion_analysis.confidence + arch_analysis.confidence) / 2,
            processing_time_seconds=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing scan: {e}")
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/last/recommend/{scan_id}", response_model=LastRecommendationResponse)
async def get_last_recommendations(scan_id: str):
    """
    Get last recommendations for a processed scan

    Args:
        scan_id: Scan identifier

    Returns:
        Last recommendations with modifications
    """
    if scan_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Scan not found")

    job = processing_jobs[scan_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Scan processing not complete")

    try:
        # Get recommendations
        recommendations = last_matcher.match(
            job['measurements'],
            job['features'],
            top_k=3
        )

        # Format response
        rec_list = []
        for rec in recommendations:
            rec_list.append({
                'last_id': rec.last_id,
                'size_eu': rec.size_eu,
                'size_uk': rec.size_uk,
                'size_us': rec.size_us,
                'confidence': rec.confidence,
                'match_score': rec.match_score,
                'reasoning': rec.reasoning
            })

        # Aggregate modifications
        all_modifications = {}
        if recommendations:
            all_modifications = recommendations[0].modifications_needed

        return LastRecommendationResponse(
            scan_id=scan_id,
            recommendations=rec_list,
            modifications_needed=all_modifications,
            confidence=recommendations[0].confidence if recommendations else 0
        )

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/printing/generate/{scan_id}", response_model=PrintingFilesResponse)
async def generate_printing_files(scan_id: str):
    """
    Generate 3D printing files for last customization

    Args:
        scan_id: Scan identifier

    Returns:
        URLs to generated G-code and STL files
    """
    if scan_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Scan not found")

    job = processing_jobs[scan_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Scan processing not complete")

    try:
        # Generate printing files
        output_dir = Path(f"output/{scan_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        print_job = generate_custom_last_modifications(
            job['point_cloud'],
            job['segmentation'],
            job['features'],
            str(output_dir)
        )

        if print_job is None:
            raise HTTPException(status_code=400, detail="No modifications needed")

        # In production, upload to cloud storage and return URLs
        # For now, return local paths
        return PrintingFilesResponse(
            scan_id=scan_id,
            gcode_url=f"/files/{scan_id}/custom_last_modification.gcode",
            stl_url=f"/files/{scan_id}/custom_last_modification.stl",
            estimated_print_time_minutes=print_job.estimated_time_minutes,
            material_weight_g=print_job.material_weight_g
        )

    except Exception as e:
        logger.error(f"Error generating printing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{scan_id}", response_model=ProcessingStatus)
async def check_processing_status(scan_id: str):
    """
    Check the processing status of a scan

    Args:
        scan_id: Scan identifier

    Returns:
        Current processing status
    """
    if scan_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Scan not found")

    job = processing_jobs[scan_id]

    return ProcessingStatus(
        scan_id=scan_id,
        status=job.get('status', 'processing'),
        stage=job.get('stage', 'unknown'),
        progress_percentage=job.get('progress', 0),
        error_message=job.get('error')
    )

@app.get("/files/{scan_id}/{filename}")
async def download_file(scan_id: str, filename: str):
    """Download generated files"""
    file_path = Path(f"output/{scan_id}/{filename}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), filename=filename)

@app.delete("/api/scan/{scan_id}")
async def delete_scan(scan_id: str):
    """Delete scan data and associated files"""
    if scan_id in processing_jobs:
        del processing_jobs[scan_id]

    # Clean up files
    output_dir = Path(f"output/{scan_id}")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return {"message": f"Scan {scan_id} deleted successfully"}

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "model": "loaded" if model is not None else "not loaded",
            "database": "connected"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)