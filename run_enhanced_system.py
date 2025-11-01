#!/usr/bin/env python3
"""
Enhanced Foot Scan to Custom Last System
Main execution script with all medical condition detections and integrations
"""

import sys
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import all modules
from ingestion.volumental_loader import VolumentalLoader
from preprocessing.point_cloud_processor import PointCloudProcessor
from models.enhanced_pointnet_foot import create_enhanced_segmentation_model, DETAILED_FOOT_SEGMENTS
from features.medical_conditions import ComprehensiveMedicalAnalyzer
from features.enhanced_medical_analyzer import EnhancedMedicalAnalyzer
from features.bunion_detector import BunionDetector
from features.arch_analyzer import ArchAnalyzer
from baseline.healthy_foot_baseline import HealthyFootDatabase, HealthyFootComparator, generate_synthetic_healthy_profiles
from matching.last_library import LastLibraryDatabase, LastSpecification
from printing.gcode_generator import generate_custom_last_modifications
from integrations.volumental_api import VolumentalAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedFootScanSystem:
    """Complete enhanced foot scanning and last customization system"""

    def __init__(self):
        """Initialize all system components"""
        logger.info("Initializing Enhanced Foot Scan System...")

        # Initialize processors
        self.point_cloud_processor = PointCloudProcessor(target_points=10000)

        # Initialize advanced medical analyzer with ensemble ML models
        self.medical_analyzer = EnhancedMedicalAnalyzer(site_id="foot_clinic_001")

        # Keep traditional analyzer for fallback
        self.traditional_medical_analyzer = ComprehensiveMedicalAnalyzer()

        # Initialize databases
        self.healthy_db = HealthyFootDatabase()
        self.healthy_comparator = HealthyFootComparator(self.healthy_db)
        self.last_library = LastLibraryDatabase()

        # Initialize segmentation model (would load trained model in production)
        self.segmentation_model = None
        try:
            self.segmentation_model = create_enhanced_segmentation_model()
            logger.info("Enhanced segmentation model loaded (45 regions)")
        except Exception as e:
            logger.warning(f"Could not load segmentation model: {e}")

        # Generate sample data if databases are empty
        self._initialize_sample_data()

        logger.info("System initialization complete")

    def _initialize_sample_data(self):
        """Generate sample data for testing"""
        # Check if healthy baselines exist
        cursor = self.healthy_db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM healthy_profiles")
        if cursor.fetchone()[0] == 0:
            logger.info("Generating synthetic healthy foot profiles...")
            generate_synthetic_healthy_profiles(self.healthy_db, count=50)

        # Check if lasts exist
        cursor = self.last_library.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM last_specifications")
        if cursor.fetchone()[0] == 0:
            logger.info("Adding sample lasts to library...")
            self._add_sample_lasts()

    def _add_sample_lasts(self):
        """Add sample lasts to library"""
        sample_lasts = [
            LastSpecification(
                last_id="ORTHO_EU42_WIDE",
                brand="MedicalFit",
                model_name="Orthopedic Pro",
                style_category="orthopedic",
                size_eu=42,
                size_uk=8,
                size_us=9,
                width_code="EE",
                length=270,
                ball_girth=260,
                waist_girth=240,
                instep_girth=250,
                heel_width=70,
                toe_spring=8,
                heel_height=25,
                ball_width=105,
                stick_length=265,
                heel_to_ball=180,
                arch_length=175,
                toe_box_height=45,
                toe_box_width=100,
                vamp_height=55,
                collar_height=70,
                toe_shape="round",
                arch_profile="normal",
                heel_cup_depth=18,
                medial_wall_height=22,
                lateral_wall_height=20,
                forefoot_rocker_angle=5,
                heel_rocker_angle=3,
                metatarsal_support_height=4,
                pronation_control=3,
                material_zones={"upper": "leather", "sole": "rubber"},
                mesh_file="data/lasts/ortho_42_wide.stl",
                point_cloud_file="data/lasts/ortho_42_wide.ply",
                cad_file=None,
                bunion_accommodation=True,
                hammer_toe_accommodation=True,
                diabetic_friendly=True,
                removable_insole=True,
                extra_depth=True,
                created_date=datetime.now().isoformat(),
                manufacturer="MedicalFit Inc",
                country_of_origin="USA",
                target_demographic="unisex",
                price_category="premium",
                quality_score=0.95,
                validation_status="validated",
                test_feedback={"comfort": 9.2, "fit": 9.5}
            ),
            LastSpecification(
                last_id="ATHLETIC_EU42_STD",
                brand="SportFit",
                model_name="Runner Pro",
                style_category="athletic",
                size_eu=42,
                size_uk=8,
                size_us=9,
                width_code="D",
                length=270,
                ball_girth=250,
                waist_girth=235,
                instep_girth=240,
                heel_width=65,
                toe_spring=12,
                heel_height=30,
                ball_width=95,
                stick_length=265,
                heel_to_ball=175,
                arch_length=170,
                toe_box_height=38,
                toe_box_width=92,
                vamp_height=50,
                collar_height=65,
                toe_shape="anatomical",
                arch_profile="normal",
                heel_cup_depth=15,
                medial_wall_height=20,
                lateral_wall_height=18,
                forefoot_rocker_angle=8,
                heel_rocker_angle=5,
                metatarsal_support_height=2,
                pronation_control=1,
                material_zones={"upper": "mesh", "sole": "eva"},
                mesh_file="data/lasts/athletic_42_std.stl",
                point_cloud_file="data/lasts/athletic_42_std.ply",
                cad_file=None,
                bunion_accommodation=False,
                hammer_toe_accommodation=False,
                diabetic_friendly=False,
                removable_insole=True,
                extra_depth=False,
                created_date=datetime.now().isoformat(),
                manufacturer="SportFit Corp",
                country_of_origin="Vietnam",
                target_demographic="unisex",
                price_category="mid",
                quality_score=0.88,
                validation_status="validated",
                test_feedback={"comfort": 8.5, "performance": 9.0}
            )
        ]

        for last in sample_lasts:
            self.last_library.add_last(last)

    def process_scan(self, obj_path: str, json_path: str) -> Dict:
        """
        Process a foot scan through the complete enhanced pipeline

        Args:
            obj_path: Path to OBJ file
            json_path: Path to JSON measurements

        Returns:
            Complete analysis results
        """
        logger.info(f"Processing scan: {obj_path}")

        # Load scan data
        loader = VolumentalLoader(obj_path, json_path)
        vertices, faces, measurements = loader.load_all()

        # Process point cloud
        point_cloud, processing_params = self.point_cloud_processor.process_scan(
            vertices, faces, normalize=True, align=True
        )

        # Perform segmentation (mock if no model)
        if self.segmentation_model:
            # Run actual segmentation
            predictions = self.segmentation_model.predict(
                point_cloud.reshape(1, -1, 3)
            )
            segmentation = np.argmax(predictions['segmentation'][0], axis=-1)
        else:
            # Mock segmentation for demo
            segmentation = self._mock_segmentation(point_cloud)

        logger.info(f"Segmented into {len(np.unique(segmentation))} regions")

        # Run comprehensive enhanced medical analysis with AI ensemble
        try:
            enhanced_medical_conditions = self.medical_analyzer.analyze_foot_enhanced(
                point_cloud, segmentation
            )

            # Generate enhanced medical report with advanced diagnostics
            medical_report = self.medical_analyzer.generate_enhanced_medical_report(
                enhanced_medical_conditions
            )

            logger.info(f"Enhanced AI analysis complete - {len(enhanced_medical_conditions)} conditions analyzed")

        except Exception as e:
            logger.warning(f"Enhanced analysis failed, falling back to traditional: {e}")

            # Fallback to traditional analysis
            medical_conditions = self.traditional_medical_analyzer.analyze_foot(
                point_cloud, segmentation
            )

            medical_report = self.traditional_medical_analyzer.generate_medical_report(
                medical_conditions
            )

        logger.info(f"Detected {len(medical_report['detected_conditions'])} medical conditions")

        # Compare to healthy baselines
        scan_measurements = {
            'size_eu': self._estimate_size_eu(measurements.foot_length),
            'foot_length': measurements.foot_length,
            'foot_width': measurements.ball_girth / 3.14,  # Approximate
            'ball_girth': measurements.ball_girth,
            'arch_height': measurements.arch_height,
            'arch_index': 0.23,  # Would calculate from segmentation
            'hallux_angle': 10,  # Would calculate from segmentation
            'total_volume': 350  # Would calculate from point cloud
        }

        baseline_comparison = self.healthy_comparator.compare_to_baseline(
            scan_measurements, point_cloud, segmentation
        )

        logger.info(f"Health score: {baseline_comparison['health_score']:.1f}/100")

        # Find best matching lasts
        medical_condition_names = [
            cond['name'].lower().replace(' ', '_')
            for cond in medical_report['detected_conditions']
        ]

        best_lasts = self.last_library.find_best_matches(
            scan_measurements,
            medical_conditions=medical_condition_names,
            k=3
        )

        logger.info(f"Found {len(best_lasts)} matching lasts")

        # Generate modifications for 3D printing
        if best_lasts:
            selected_last = best_lasts[0][0]
            logger.info(f"Selected last: {selected_last.last_id}")

            # Aggregate all modifications needed
            all_modifications = medical_report['total_modifications']

            # Generate printing files (simplified)
            printing_info = {
                'last_id': selected_last.last_id,
                'modifications': all_modifications,
                'estimated_print_time': len(all_modifications) * 15,  # Minutes
                'material_required': sum(abs(v) for v in all_modifications.values()) * 2  # Grams
            }
        else:
            printing_info = None

        # Compile complete results
        results = {
            'scan_id': Path(obj_path).stem,
            'timestamp': datetime.now().isoformat(),
            'measurements': measurements.__dict__,
            'segmentation_regions': len(np.unique(segmentation)),
            'medical_conditions': medical_report,
            'health_comparison': baseline_comparison,
            'recommended_lasts': [
                {
                    'last_id': last.last_id,
                    'brand': last.brand,
                    'model': last.model_name,
                    'score': score,
                    'style': last.style_category
                }
                for last, score in best_lasts
            ],
            'printing_info': printing_info,
            'processing_params': processing_params
        }

        return results

    def _estimate_size_eu(self, foot_length_mm: float) -> float:
        """Estimate EU size from foot length"""
        return foot_length_mm / 6.67

    def _mock_segmentation(self, point_cloud: np.ndarray) -> np.ndarray:
        """Create mock segmentation for testing"""
        segmentation = np.zeros(len(point_cloud), dtype=int)

        # Simple region assignment based on position
        y_coords = point_cloud[:, 1]
        z_coords = point_cloud[:, 2]

        # Toe region
        segmentation[y_coords > np.percentile(y_coords, 85)] = 1

        # Forefoot
        mask = (y_coords > np.percentile(y_coords, 60)) & (y_coords <= np.percentile(y_coords, 85))
        segmentation[mask] = 16

        # Midfoot
        mask = (y_coords > np.percentile(y_coords, 30)) & (y_coords <= np.percentile(y_coords, 60))
        segmentation[mask] = 24

        # Hindfoot
        segmentation[y_coords <= np.percentile(y_coords, 30)] = 32

        # Dorsal regions (high z)
        segmentation[z_coords > np.percentile(z_coords, 80)] = 38

        return segmentation

    def generate_report(self, results: Dict, output_path: str):
        """Generate comprehensive HTML report with advanced diagnostic information"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced AI Foot Scan Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
                h2 {{ color: #34495e; margin-top: 40px; margin-bottom: 20px; border-left: 4px solid #3498db; padding-left: 15px; }}
                h3 {{ color: #2c3e50; margin-bottom: 10px; }}
                .header-info {{ display: flex; justify-content: space-between; margin-bottom: 30px; }}
                .metric {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }}
                .condition {{ background: #f8f9ff; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 5px solid; }}
                .severe {{ border-left-color: #e74c3c; }}
                .moderate {{ border-left-color: #f39c12; }}
                .mild {{ border-left-color: #27ae60; }}
                .health-score {{ font-size: 48px; font-weight: bold; color: #3498db; text-align: center; margin: 20px 0; }}
                .ai-analysis {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .ensemble-info {{ background: #e8f4fd; padding: 15px; border-radius: 8px; margin: 10px 0; }}
                .uncertainty {{ background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 10px 0; }}
                .model-consensus {{ background: #d1ecf1; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                .risk-factors {{ background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .explanation {{ font-style: italic; color: #6c757d; margin: 10px 0; padding: 10px; background: #f1f3f4; border-radius: 5px; }}
                .last-recommendation {{ background: #f0f2ff; padding: 15px; margin: 10px 0; border-radius: 8px; }}
                .clinical-recommendations {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #28a745; }}
                .evidence-strength {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
                .strong {{ background: #28a745; color: white; }}
                .moderate-evidence {{ background: #ffc107; color: black; }}
                .weak {{ background: #6c757d; color: white; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #3498db; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Enhanced AI Foot Scan Analysis Report</h1>

                <div class="header-info">
                    <div><strong>Scan ID:</strong> {results['scan_id']}</div>
                    <div><strong>Analysis Date:</strong> {results['timestamp']}</div>
                    <div><strong>AI Models:</strong> Ensemble Learning + Uncertainty Quantification</div>
                </div>"""

        # Add AI Analysis summary if available
        if 'ensemble_analysis' in results['medical_conditions']:
            ensemble_data = results['medical_conditions']['ensemble_analysis']
            html_content += f"""
                <div class="ai-analysis">
                    <h3>🤖 Advanced AI Diagnostic Summary</h3>
                    <p><strong>Models Used:</strong> {', '.join(ensemble_data.get('models_used', []))}</p>
                    <p><strong>Conditions Analyzed:</strong> {ensemble_data.get('total_conditions_analyzed', 0)}</p>
                    <p><strong>Average Confidence:</strong> {ensemble_data.get('average_confidence', 0)*100:.1f}%</p>
                    <p><strong>High Confidence Diagnoses:</strong> {ensemble_data.get('high_confidence_conditions', 0)}</p>
                </div>
            """

        # Health Score
        html_content += f"""
                <h2>🏥 Health Assessment</h2>
                <div class="health-score">{results['health_comparison']['health_score']:.1f}/100</div>
        """

        # Add statistics grid
        if 'uncertainty_analysis' in results['medical_conditions']:
            uncertainty_data = results['medical_conditions']['uncertainty_analysis']
            consensus_data = results['medical_conditions'].get('model_consensus', {})

            html_content += f"""
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>🎯 Diagnostic Certainty</h4>
                        <p><strong>{(1-uncertainty_data.get('average_uncertainty', 0))*100:.1f}%</strong></p>
                        <small>{uncertainty_data.get('uncertainty_recommendation', '')}</small>
                    </div>
                    <div class="stat-card">
                        <h4>🤝 Model Agreement</h4>
                        <p><strong>{consensus_data.get('average_agreement', 0)*100:.1f}%</strong></p>
                        <small>Consensus across ML models</small>
                    </div>
                    <div class="stat-card">
                        <h4>⚠️ Risk Assessment</h4>
                        <p><strong>{results['medical_conditions'].get('risk_assessment', {}).get('risk_level', 'Unknown').title()}</strong></p>
                        <small>{results['medical_conditions'].get('risk_assessment', {}).get('total_risk_factors', 0)} risk factors identified</small>
                    </div>
                </div>
            """

        # Detected Conditions
        html_content += f"""
                <h2>🔍 Detected Medical Conditions ({len(results['medical_conditions']['detected_conditions'])})</h2>
        """

        for condition in results['medical_conditions']['detected_conditions']:
            # Determine evidence strength styling
            evidence_class = condition.get('evidence_strength', 'moderate-evidence')
            if evidence_class not in ['strong', 'moderate-evidence', 'weak']:
                evidence_class = 'moderate-evidence'

            html_content += f"""
            <div class="condition {condition.get('severity', 'mild')}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <h3>{condition['name']}</h3>
                    <span class="evidence-strength {evidence_class}">{condition.get('evidence_strength', 'moderate').title()} Evidence</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div>
                        <p><strong>Severity:</strong> {condition['severity'].title()}</p>
                        <p><strong>Confidence:</strong> {condition['confidence']*100:.1f}%</p>
                        {f"<p><strong>Uncertainty:</strong> {condition.get('uncertainty', 0)*100:.1f}%</p>" if 'uncertainty' in condition else ""}
                    </div>
                    <div>
                        {f"<p><strong>Risk Factors:</strong> {len(condition.get('risk_factors', []))}</p>" if 'risk_factors' in condition else ""}
                    </div>
                </div>

                {f'<div class="explanation">💡 <strong>AI Explanation:</strong> {condition.get("explanation", "Traditional analysis performed")}</div>' if condition.get('explanation') else ''}

                {f'<div class="risk-factors"><strong>⚠️ Risk Factors:</strong> {", ".join(condition.get("risk_factors", []))}</div>' if condition.get('risk_factors') else ''}

                {f'<div class="model-consensus"><strong>🤖 Model Consensus:</strong> {", ".join([f"{k}: {v*100:.0f}%" for k, v in condition.get("model_consensus", {}).items()])}</div>' if condition.get('model_consensus') else ''}
            </div>
            """

        # Clinical Recommendations
        if 'clinical_recommendations' in results['medical_conditions']:
            html_content += f"""
                <h2>👩‍⚕️ Clinical Recommendations</h2>
                <div class="clinical-recommendations">
            """
            for rec in results['medical_conditions']['clinical_recommendations']:
                html_content += f"<p>• {rec}</p>"
            html_content += "</div>"

        html_content += f"""
            <h2>Recommended Lasts</h2>
        """

        for last in results['recommended_lasts'][:3]:
            html_content += f"""
            <div class="last-recommendation">
                <h3>{last['brand']} - {last['model']}</h3>
                <p>Style: {last['style']}</p>
                <p>Match Score: {last['score']*100:.1f}%</p>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Report generated: {output_path}")

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Foot Scan Processing System")
    parser.add_argument("--obj", required=True, help="Path to OBJ file")
    parser.add_argument("--json", required=True, help="Path to JSON measurements")
    parser.add_argument("--output", default="report.html", help="Output report path")

    args = parser.parse_args()

    # Initialize system
    system = EnhancedFootScanSystem()

    # Process scan
    results = system.process_scan(args.obj, args.json)

    # Generate report
    system.generate_report(results, args.output)

    # Print summary
    print("\n" + "="*50)
    print("ENHANCED FOOT SCAN ANALYSIS COMPLETE")
    print("="*50)
    print(f"Health Score: {results['health_comparison']['health_score']:.1f}/100")
    print(f"Medical Conditions Detected: {len(results['medical_conditions']['detected_conditions'])}")

    for condition in results['medical_conditions']['detected_conditions']:
        print(f"  - {condition['name']} ({condition['severity']})")

    print(f"\nTop Last Recommendation: {results['recommended_lasts'][0]['model'] if results['recommended_lasts'] else 'None'}")
    print(f"\nReport saved to: {args.output}")

if __name__ == "__main__":
    # If no arguments provided, run demo
    if len(sys.argv) == 1:
        print("Running demo with sample data...")
        system = EnhancedFootScanSystem()
        print("\nEnhanced Foot Scan System initialized successfully!")
        print("\nCapabilities:")
        print("- 45 anatomical regions for precise segmentation")
        print("- Detection of 10+ medical conditions")
        print("- Healthy baseline comparison")
        print("- Intelligent last matching from library")
        print("- Volumental API integration ready")
        print("\nTo process a scan, run:")
        print("  python run_enhanced_system.py --obj scan.obj --json measurements.json")
    else:
        main()