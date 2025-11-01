"""
Healthy Foot Baseline Management System
Maintains database of healthy foot profiles for comparison and anomaly detection
"""

import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from scipy import stats
from scipy.spatial import distance
import pickle

logger = logging.getLogger(__name__)

@dataclass
class HealthyFootProfile:
    """Healthy foot baseline profile"""
    profile_id: str
    size_eu: float
    size_uk: float
    size_us: float
    gender: str  # 'male', 'female', 'unisex'
    age_group: str  # 'child', 'teen', 'adult', 'senior'
    ethnicity: Optional[str]  # For population-specific variations

    # Measurements (all in mm)
    foot_length: float
    foot_width: float
    ball_girth: float
    waist_girth: float
    instep_girth: float
    heel_width: float
    arch_height: float

    # Anatomical measurements
    hallux_length: float
    hallux_angle: float  # Normal range
    metatarsal_width: float
    arch_index: float  # Cavanagh-Rodgers index
    heel_angle: float  # Calcaneal angle

    # Volume and surface metrics
    total_volume: float  # in ml
    surface_area: float  # in cm²

    # Point cloud data
    point_cloud_file: str  # Path to stored point cloud
    segmentation_file: str  # Path to segmentation data

    # Statistical measures
    measurement_std_dev: Dict[str, float]  # Standard deviations
    percentile_rank: Dict[str, float]  # Percentile rankings

    # Metadata
    scan_date: str
    scanner_model: str
    data_quality_score: float

class HealthyFootDatabase:
    """Database for healthy foot profiles"""

    def __init__(self, db_path: str = "data/healthy_baselines.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.initialize_database()

        # Cache for frequently accessed profiles
        self.profile_cache = {}

        # Statistical models
        self.population_stats = {}

    def initialize_database(self):
        """Create database schema"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Main profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS healthy_profiles (
                profile_id TEXT PRIMARY KEY,
                size_eu REAL,
                size_uk REAL,
                size_us REAL,
                gender TEXT,
                age_group TEXT,
                ethnicity TEXT,
                foot_length REAL,
                foot_width REAL,
                ball_girth REAL,
                waist_girth REAL,
                instep_girth REAL,
                heel_width REAL,
                arch_height REAL,
                hallux_length REAL,
                hallux_angle REAL,
                metatarsal_width REAL,
                arch_index REAL,
                heel_angle REAL,
                total_volume REAL,
                surface_area REAL,
                point_cloud_file TEXT,
                segmentation_file TEXT,
                measurement_std_dev TEXT,
                percentile_rank TEXT,
                scan_date TEXT,
                scanner_model TEXT,
                data_quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Population statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS population_statistics (
                stat_id TEXT PRIMARY KEY,
                size_group TEXT,
                gender TEXT,
                age_group TEXT,
                measurement_type TEXT,
                mean_value REAL,
                std_dev REAL,
                min_value REAL,
                max_value REAL,
                percentile_25 REAL,
                percentile_50 REAL,
                percentile_75 REAL,
                sample_size INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Anomaly thresholds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_thresholds (
                threshold_id TEXT PRIMARY KEY,
                condition_name TEXT,
                measurement_type TEXT,
                min_normal REAL,
                max_normal REAL,
                mild_threshold REAL,
                moderate_threshold REAL,
                severe_threshold REAL,
                confidence_required REAL
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_size ON healthy_profiles(size_eu)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gender_age ON healthy_profiles(gender, age_group)")

        self.conn.commit()
        logger.info(f"Healthy baseline database initialized at {self.db_path}")

    def add_healthy_profile(self, profile: HealthyFootProfile):
        """Add a healthy foot profile to database"""
        cursor = self.conn.cursor()

        # Convert dictionaries to JSON strings
        profile_dict = asdict(profile)
        profile_dict['measurement_std_dev'] = json.dumps(profile.measurement_std_dev)
        profile_dict['percentile_rank'] = json.dumps(profile.percentile_rank)

        columns = list(profile_dict.keys())
        placeholders = ','.join(['?' for _ in columns])

        query = f"""
            INSERT OR REPLACE INTO healthy_profiles ({','.join(columns)})
            VALUES ({placeholders})
        """

        cursor.execute(query, list(profile_dict.values()))
        self.conn.commit()

        # Update cache
        self.profile_cache[profile.profile_id] = profile

        # Update population statistics
        self._update_population_stats(profile)

        logger.info(f"Added healthy profile {profile.profile_id}")

    def get_matching_baselines(self,
                              size_eu: float,
                              gender: str = None,
                              age_group: str = None,
                              tolerance: float = 1.0) -> List[HealthyFootProfile]:
        """Get healthy baselines matching criteria"""

        cursor = self.conn.cursor()

        query = """
            SELECT * FROM healthy_profiles
            WHERE ABS(size_eu - ?) <= ?
        """
        params = [size_eu, tolerance]

        if gender:
            query += " AND (gender = ? OR gender = 'unisex')"
            params.append(gender)

        if age_group:
            query += " AND age_group = ?"
            params.append(age_group)

        query += " ORDER BY ABS(size_eu - ?) LIMIT 10"
        params.append(size_eu)

        cursor.execute(query, params)

        profiles = []
        for row in cursor.fetchall():
            profile = self._row_to_profile(row)
            profiles.append(profile)

        return profiles

    def _row_to_profile(self, row) -> HealthyFootProfile:
        """Convert database row to HealthyFootProfile"""
        columns = [desc[0] for desc in self.conn.cursor().description]
        data = dict(zip(columns, row))

        # Parse JSON fields
        data['measurement_std_dev'] = json.loads(data['measurement_std_dev'])
        data['percentile_rank'] = json.loads(data['percentile_rank'])

        # Remove non-profile fields
        data.pop('created_at', None)

        return HealthyFootProfile(**data)

    def _update_population_stats(self, profile: HealthyFootProfile):
        """Update population statistics with new profile"""

        cursor = self.conn.cursor()

        # Define measurement types to track
        measurements = [
            'foot_length', 'foot_width', 'ball_girth', 'arch_height',
            'arch_index', 'hallux_angle', 'total_volume'
        ]

        for measurement in measurements:
            # Get current statistics
            stat_id = f"{profile.gender}_{profile.age_group}_{profile.size_eu:.0f}_{measurement}"

            cursor.execute("""
                SELECT mean_value, std_dev, sample_size
                FROM population_statistics
                WHERE stat_id = ?
            """, (stat_id,))

            result = cursor.fetchone()

            if result:
                # Update existing statistics (running average)
                old_mean, old_std, old_n = result
                new_value = getattr(profile, measurement)

                # Calculate new mean and std dev
                new_n = old_n + 1
                new_mean = (old_mean * old_n + new_value) / new_n
                new_std = np.sqrt(((old_n - 1) * old_std**2 +
                                  (new_value - new_mean)**2) / old_n) if old_n > 0 else 0

                cursor.execute("""
                    UPDATE population_statistics
                    SET mean_value = ?, std_dev = ?, sample_size = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE stat_id = ?
                """, (new_mean, new_std, new_n, stat_id))
            else:
                # Create new statistic entry
                cursor.execute("""
                    INSERT INTO population_statistics (
                        stat_id, size_group, gender, age_group, measurement_type,
                        mean_value, std_dev, sample_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (stat_id, f"EU{profile.size_eu:.0f}", profile.gender,
                     profile.age_group, measurement, getattr(profile, measurement), 0, 1))

        self.conn.commit()

    def load_anomaly_thresholds(self):
        """Load medical condition thresholds"""

        thresholds = [
            # Bunion thresholds
            ('bunion_mild', 'bunion', 'hallux_angle', 0, 15, 15, 25, 35, 0.7),
            ('bunion_prominence', 'bunion', 'medial_prominence', 0, 2, 2, 5, 8, 0.6),

            # Flat feet thresholds
            ('flat_feet_arch', 'flat_feet', 'arch_index', 0.21, 0.26, 0.26, 0.31, 0.35, 0.8),
            ('flat_feet_height', 'flat_feet', 'arch_height', 10, 20, 10, 7, 4, 0.75),

            # Hammer toe thresholds
            ('hammer_toe', 'hammer_toe', 'toe_flexion_angle', 0, 20, 30, 40, 50, 0.65),

            # Swelling thresholds
            ('swelling_volume', 'edema', 'volume_increase_percent', 0, 5, 10, 20, 30, 0.7),
            ('swelling_girth', 'edema', 'girth_increase_percent', 0, 5, 10, 15, 25, 0.7),

            # Gout thresholds
            ('gout_swelling', 'gout', 'joint_swelling', 0, 2, 3, 5, 8, 0.6),
        ]

        cursor = self.conn.cursor()
        for threshold_data in thresholds:
            cursor.execute("""
                INSERT OR REPLACE INTO anomaly_thresholds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, threshold_data)

        self.conn.commit()
        logger.info("Loaded anomaly thresholds")

class HealthyFootComparator:
    """Compare scanned foot to healthy baselines"""

    def __init__(self, database: HealthyFootDatabase):
        self.database = database

    def compare_to_baseline(self,
                           scan_data: Dict,
                           point_cloud: np.ndarray,
                           segmentation: np.ndarray) -> Dict:
        """Compare foot scan to healthy baselines"""

        # Get matching baselines
        baselines = self.database.get_matching_baselines(
            size_eu=scan_data.get('size_eu', 42),
            gender=scan_data.get('gender'),
            age_group=scan_data.get('age_group', 'adult')
        )

        if not baselines:
            logger.warning("No matching healthy baselines found")
            return self._no_baseline_comparison()

        # Perform comparisons
        comparison_results = {
            'baseline_count': len(baselines),
            'deviations': {},
            'percentile_rankings': {},
            'anomaly_scores': {},
            'health_score': 0,
            'recommendations': []
        }

        # Calculate deviations from average baseline
        avg_baseline = self._calculate_average_baseline(baselines)

        # Measurement comparisons
        measurements_to_compare = [
            'foot_length', 'foot_width', 'ball_girth', 'arch_height',
            'arch_index', 'hallux_angle', 'total_volume'
        ]

        for measurement in measurements_to_compare:
            if measurement in scan_data and measurement in avg_baseline:
                deviation = scan_data[measurement] - avg_baseline[measurement]
                std_dev = avg_baseline.get(f'{measurement}_std', 1)
                z_score = deviation / std_dev if std_dev > 0 else 0

                comparison_results['deviations'][measurement] = {
                    'value': scan_data[measurement],
                    'baseline': avg_baseline[measurement],
                    'deviation': deviation,
                    'z_score': z_score,
                    'percentile': stats.norm.cdf(z_score) * 100
                }

                # Flag significant deviations
                if abs(z_score) > 2:
                    comparison_results['anomaly_scores'][measurement] = abs(z_score)

        # Shape comparison using point cloud
        if len(baselines) > 0 and baselines[0].point_cloud_file:
            shape_similarity = self._compare_shape(
                point_cloud, baselines[0].point_cloud_file
            )
            comparison_results['shape_similarity'] = shape_similarity

        # Calculate overall health score
        comparison_results['health_score'] = self._calculate_health_score(
            comparison_results['anomaly_scores']
        )

        # Generate recommendations
        comparison_results['recommendations'] = self._generate_recommendations(
            comparison_results['deviations'],
            comparison_results['anomaly_scores']
        )

        return comparison_results

    def _calculate_average_baseline(self, baselines: List[HealthyFootProfile]) -> Dict:
        """Calculate average measurements from baselines"""

        avg = {}
        measurements = [
            'foot_length', 'foot_width', 'ball_girth', 'waist_girth',
            'instep_girth', 'heel_width', 'arch_height', 'hallux_angle',
            'arch_index', 'total_volume'
        ]

        for measurement in measurements:
            values = [getattr(b, measurement) for b in baselines
                     if hasattr(b, measurement)]
            if values:
                avg[measurement] = np.mean(values)
                avg[f'{measurement}_std'] = np.std(values)

        return avg

    def _compare_shape(self, scan_cloud: np.ndarray, baseline_file: str) -> float:
        """Compare shape similarity between scan and baseline"""

        try:
            # Load baseline point cloud
            baseline_cloud = np.load(baseline_file) if Path(baseline_file).exists() else None

            if baseline_cloud is None:
                return 0.5

            # Use Hausdorff distance for shape comparison
            hausdorff_dist = distance.directed_hausdorff(scan_cloud, baseline_cloud)[0]

            # Convert to similarity score (0-1)
            similarity = 1.0 / (1.0 + hausdorff_dist / 100)

            return similarity
        except Exception as e:
            logger.error(f"Shape comparison failed: {e}")
            return 0.5

    def _calculate_health_score(self, anomaly_scores: Dict) -> float:
        """Calculate overall foot health score (0-100)"""

        if not anomaly_scores:
            return 100.0

        # Weight different anomalies
        weights = {
            'arch_height': 2.0,
            'arch_index': 2.0,
            'hallux_angle': 1.5,
            'foot_width': 1.0,
            'total_volume': 0.8
        }

        total_weighted_anomaly = 0
        total_weight = 0

        for measurement, z_score in anomaly_scores.items():
            weight = weights.get(measurement, 1.0)
            total_weighted_anomaly += abs(z_score) * weight
            total_weight += weight

        if total_weight > 0:
            avg_anomaly = total_weighted_anomaly / total_weight
            # Convert to 0-100 score (lower anomaly = higher health)
            health_score = max(0, 100 - (avg_anomaly * 20))
        else:
            health_score = 100.0

        return health_score

    def _generate_recommendations(self, deviations: Dict, anomalies: Dict) -> List[str]:
        """Generate recommendations based on deviations"""

        recommendations = []

        # Check arch height
        if 'arch_height' in anomalies:
            if deviations['arch_height']['deviation'] < -5:
                recommendations.append("Consider arch support due to lower than normal arch")
            elif deviations['arch_height']['deviation'] > 5:
                recommendations.append("May benefit from cushioning for high arch")

        # Check foot width
        if 'foot_width' in anomalies:
            if deviations['foot_width']['deviation'] > 5:
                recommendations.append("Recommend wide last option")

        # Check hallux angle
        if 'hallux_angle' in deviations:
            if deviations['hallux_angle']['value'] > 15:
                recommendations.append("Bunion accommodation may be needed")

        # Check volume
        if 'total_volume' in anomalies:
            if deviations['total_volume']['deviation'] > 50:
                recommendations.append("Consider adjustable closure for volume variation")

        if not recommendations:
            recommendations.append("Foot measurements within healthy range")

        return recommendations

    def _no_baseline_comparison(self) -> Dict:
        """Return when no baseline available"""
        return {
            'baseline_count': 0,
            'deviations': {},
            'percentile_rankings': {},
            'anomaly_scores': {},
            'health_score': -1,  # Unknown
            'recommendations': ["No baseline data available for comparison"]
        }

def generate_synthetic_healthy_profiles(database: HealthyFootDatabase, count: int = 100):
    """Generate synthetic healthy foot profiles for testing"""

    import uuid
    from datetime import datetime

    sizes_eu = np.arange(36, 48, 0.5)
    genders = ['male', 'female']
    age_groups = ['adult', 'senior']

    for _ in range(count):
        size_eu = np.random.choice(sizes_eu)
        gender = np.random.choice(genders)
        age_group = np.random.choice(age_groups)

        # Generate measurements based on size and gender
        base_length = size_eu * 6.67  # Approximate conversion

        if gender == 'male':
            width_ratio = 0.38
            volume_base = 400
        else:
            width_ratio = 0.36
            volume_base = 350

        profile = HealthyFootProfile(
            profile_id=str(uuid.uuid4()),
            size_eu=size_eu,
            size_uk=size_eu - 34,  # Approximate
            size_us=size_eu - 33,  # Approximate
            gender=gender,
            age_group=age_group,
            ethnicity=None,
            foot_length=base_length + np.random.normal(0, 3),
            foot_width=base_length * width_ratio + np.random.normal(0, 2),
            ball_girth=base_length * 0.9 + np.random.normal(0, 3),
            waist_girth=base_length * 0.85 + np.random.normal(0, 3),
            instep_girth=base_length * 0.88 + np.random.normal(0, 3),
            heel_width=base_length * 0.24 + np.random.normal(0, 2),
            arch_height=15 + np.random.normal(0, 2),  # Normal arch ~15mm
            hallux_length=base_length * 0.28 + np.random.normal(0, 2),
            hallux_angle=np.random.normal(8, 3),  # Normal <15 degrees
            metatarsal_width=base_length * 0.35 + np.random.normal(0, 2),
            arch_index=np.random.normal(0.23, 0.02),  # Normal 0.21-0.26
            heel_angle=np.random.normal(5, 2),  # Normal <10 degrees
            total_volume=volume_base + (size_eu - 40) * 15 + np.random.normal(0, 20),
            surface_area=200 + (size_eu - 40) * 5 + np.random.normal(0, 10),
            point_cloud_file=f"data/healthy/{profile_id}_cloud.npy",
            segmentation_file=f"data/healthy/{profile_id}_seg.npy",
            measurement_std_dev={},
            percentile_rank={},
            scan_date=datetime.now().isoformat(),
            scanner_model="Volumental",
            data_quality_score=np.random.uniform(0.8, 1.0)
        )

        database.add_healthy_profile(profile)

    logger.info(f"Generated {count} synthetic healthy profiles")