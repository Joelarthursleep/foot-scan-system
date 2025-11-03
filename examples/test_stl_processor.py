"""
Example: Test STL Processor
Demonstrates high-performance STL processing with quality validation
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.preprocessing import STLProcessor, ProcessingStatus


async def main():
    """Test STL processor with sample scan"""

    # Initialize processor
    processor = STLProcessor(
        validate_quality=True,
        extract_features=True,
        compute_curvature=True
    )

    # Example: Process single foot
    print("=" * 60)
    print("Testing STL Processor v2.0.0-medical")
    print("=" * 60)

    # You would replace this with actual STL file path
    stl_path = Path("data/scans/example_left.stl")

    if not stl_path.exists():
        print(f"\n⚠️  Example STL file not found: {stl_path}")
        print("To test with actual scan:")
        print(f"  python {__file__} /path/to/scan.stl")
        return

    print(f"\nProcessing: {stl_path}")
    print("Target: <5 seconds\n")

    # Process
    result = await processor.process_stl(
        file_path=stl_path,
        scan_id="DEMO-001",
        side="left"
    )

    # Display results
    print(f"Status: {result.status.value}")
    print(f"Processing time: {result.processing_duration_seconds:.2f}s")
    print(f"File checksum: {result.file_checksum[:16]}...")

    if result.quality_metrics:
        qm = result.quality_metrics
        print(f"\n{'Quality Metrics':-^60}")
        print(f"  Quality Score: {qm.quality_score:.2f}")
        print(f"  Vertices: {qm.vertex_count:,}")
        print(f"  Faces: {qm.face_count:,}")
        print(f"  Watertight: {'✓' if qm.is_watertight else '✗'}")
        print(f"  Manifold: {'✓' if qm.is_manifold else '✗'}")
        print(f"  Point Density: {qm.point_cloud_density:.1f} points/cm²")
        print(f"  Surface Area: {qm.surface_area/100:.1f} cm²")
        print(f"  Volume: {qm.volume/1000:.1f} cm³")
        print(f"  Dimensions: {qm.length_mm:.1f} x {qm.width_mm:.1f} x {qm.height_mm:.1f} mm")
        print(f"  Anatomically Plausible: {'✓' if qm.is_anatomically_plausible else '✗'}")

        if qm.quality_issues:
            print(f"\n  Quality Issues:")
            for issue in qm.quality_issues:
                print(f"    [{issue['severity'].upper()}] {issue['message']}")

    if result.morphological_features:
        mf = result.morphological_features
        print(f"\n{'Morphological Features':-^60}")
        print(f"  Length: {mf.length_mm:.1f} mm")
        print(f"  Width: {mf.width_mm:.1f} mm")
        print(f"  Height: {mf.height_mm:.1f} mm")
        print(f"  Arch Height: {mf.arch_height_mm:.1f} mm ({mf.arch_height_ratio:.3f})")
        print(f"  Arch Index: {mf.arch_index:.3f}")
        print(f"  Forefoot Width: {mf.forefoot_width_mm:.1f} mm")
        print(f"  Midfoot Width: {mf.midfoot_width_mm:.1f} mm")
        print(f"  Heel Width: {mf.heel_width_mm:.1f} mm")
        print(f"  Volume: {mf.total_volume_cm3:.1f} cm³")
        print(f"  Surface Area: {mf.total_surface_area_cm2:.1f} cm²")
        print(f"  Plantar Area: {mf.plantar_surface_area_cm2:.1f} cm²")

    if result.status == ProcessingStatus.SUCCESS:
        print(f"\n{'✓ Processing successful!':-^60}")
    else:
        print(f"\n{'✗ Processing failed or has warnings':-^60}")
        if result.error_message:
            print(f"  Error: {result.error_message}")

    print("\n" + "=" * 60)


async def test_foot_pair():
    """Test processing left and right foot pair"""

    processor = STLProcessor()

    left_path = Path("data/scans/example_left.stl")
    right_path = Path("data/scans/example_right.stl")

    if not (left_path.exists() and right_path.exists()):
        print("Foot pair test skipped - example files not found")
        return

    print("\nTesting foot pair processing...")

    left_result, right_result = await processor.process_foot_pair(
        left_stl_path=left_path,
        right_stl_path=right_path,
        scan_id="DEMO-PAIR-001"
    )

    print(f"Left foot: {left_result.status.value} ({left_result.processing_duration_seconds:.2f}s)")
    print(f"Right foot: {right_result.status.value} ({right_result.processing_duration_seconds:.2f}s)")

    if (left_result.morphological_features and right_result.morphological_features):
        left_mf = left_result.morphological_features
        right_mf = right_result.morphological_features

        print(f"\nAsymmetry Analysis:")
        print(f"  Length difference: {left_mf.length_asymmetry_mm:.1f} mm")
        print(f"  Width difference: {left_mf.width_asymmetry_mm:.1f} mm")
        print(f"  Volume difference: {left_mf.volume_asymmetry_percent:.1f}%")


if __name__ == "__main__":
    # Run test
    if len(sys.argv) > 1:
        # Custom STL file provided
        asyncio.run(main())
    else:
        # Run with example data
        asyncio.run(main())
