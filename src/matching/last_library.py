"""
Comprehensive Last Library Management System
Manages complete library of shoe lasts with 3D models, specifications,
and intelligent matching algorithms
"""

import sqlite3
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime
import pickle
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

@dataclass
class LastSpecification:
    """Complete specification for a shoe last"""

    # Identity
    last_id: str
    brand: str
    model_name: str
    style_category: str  # 'athletic', 'dress', 'casual', 'orthopedic', etc.

    # Sizes
    size_eu: float
    size_uk: float
    size_us: float
    width_code: str  # 'A', 'B', 'C', 'D', 'E', 'EE', 'EEE', 'EEEE'

    # Primary measurements (mm)
    length: float
    ball_girth: float
    waist_girth: float
    instep_girth: float
    heel_width: float
    toe_spring: float
    heel_height: float
    ball_width: float

    # Detailed measurements
    stick_length: float  # Inside length
    heel_to_ball: float
    arch_length: float
    toe_box_height: float
    toe_box_width: float
    vamp_height: float
    collar_height: float

    # Shape characteristics
    toe_shape: str  # 'round', 'square', 'pointed', 'oblique', 'anatomical'
    arch_profile: str  # 'low', 'normal', 'high', 'extra_high'
    heel_cup_depth: float
    medial_wall_height: float
    lateral_wall_height: float

    # Biomechanical features
    forefoot_rocker_angle: float
    heel_rocker_angle: float
    metatarsal_support_height: float
    pronation_control: float  # Degrees of medial posting

    # Material zones
    material_zones: Dict[str, str]  # Zone name -> material type

    # 3D model paths
    mesh_file: str
    point_cloud_file: str
    cad_file: Optional[str]

    # Medical accommodations
    bunion_accommodation: bool
    hammer_toe_accommodation: bool
    diabetic_friendly: bool
    removable_insole: bool
    extra_depth: bool

    # Metadata
    created_date: str
    manufacturer: str
    country_of_origin: str
    target_demographic: str  # 'men', 'women', 'unisex', 'children'
    price_category: str  # 'budget', 'mid', 'premium', 'luxury'

    # Quality and validation
    quality_score: float
    validation_status: str  # 'validated', 'pending', 'rejected'
    test_feedback: Dict[str, any]

class LastLibraryDatabase:
    """Database management for last library"""

    def __init__(self, db_path: str = "data/last_library.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.initialize_database()

        # Feature extraction cache
        self.feature_cache = {}

        # ML model for matching
        self.matching_model = None
        self._init_matching_model()

    def initialize_database(self):
        """Create comprehensive database schema"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Main last specifications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS last_specifications (
                last_id TEXT PRIMARY KEY,
                brand TEXT,
                model_name TEXT,
                style_category TEXT,
                size_eu REAL,
                size_uk REAL,
                size_us REAL,
                width_code TEXT,
                length REAL,
                ball_girth REAL,
                waist_girth REAL,
                instep_girth REAL,
                heel_width REAL,
                toe_spring REAL,
                heel_height REAL,
                ball_width REAL,
                stick_length REAL,
                heel_to_ball REAL,
                arch_length REAL,
                toe_box_height REAL,
                toe_box_width REAL,
                vamp_height REAL,
                collar_height REAL,
                toe_shape TEXT,
                arch_profile TEXT,
                heel_cup_depth REAL,
                medial_wall_height REAL,
                lateral_wall_height REAL,
                forefoot_rocker_angle REAL,
                heel_rocker_angle REAL,
                metatarsal_support_height REAL,
                pronation_control REAL,
                material_zones TEXT,
                mesh_file TEXT,
                point_cloud_file TEXT,
                cad_file TEXT,
                bunion_accommodation INTEGER,
                hammer_toe_accommodation INTEGER,
                diabetic_friendly INTEGER,
                removable_insole INTEGER,
                extra_depth INTEGER,
                created_date TEXT,
                manufacturer TEXT,
                country_of_origin TEXT,
                target_demographic TEXT,
                price_category TEXT,
                quality_score REAL,
                validation_status TEXT,
                test_feedback TEXT,
                feature_vector BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Last usage history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS last_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_id TEXT,
                scan_id TEXT,
                customer_id TEXT,
                usage_date TEXT,
                fit_score REAL,
                comfort_score REAL,
                modifications_applied TEXT,
                feedback TEXT,
                FOREIGN KEY (last_id) REFERENCES last_specifications(last_id)
            )
        """)

        # Last compatibility matrix
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS last_compatibility (
                compatibility_id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_id TEXT,
                condition_name TEXT,
                compatibility_score REAL,
                notes TEXT,
                FOREIGN KEY (last_id) REFERENCES last_specifications(last_id)
            )
        """)

        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_size_eu ON last_specifications(size_eu)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_style ON last_specifications(style_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_width ON last_specifications(width_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_medical ON last_specifications(bunion_accommodation, diabetic_friendly)")

        self.conn.commit()
        logger.info("Last library database initialized")

    def add_last(self, last_spec: LastSpecification):
        """Add a new last to the library"""

        # Extract features for ML matching
        feature_vector = self._extract_features(last_spec)

        cursor = self.conn.cursor()
        data = asdict(last_spec)

        # Convert complex types to JSON
        data['material_zones'] = json.dumps(data['material_zones'])
        data['test_feedback'] = json.dumps(data['test_feedback'])
        data['feature_vector'] = pickle.dumps(feature_vector)

        # Prepare insert statement
        columns = list(data.keys())
        placeholders = ','.join(['?' for _ in columns])

        query = f"""
            INSERT OR REPLACE INTO last_specifications ({','.join(columns)})
            VALUES ({placeholders})
        """

        cursor.execute(query, list(data.values()))

        # Add compatibility scores
        self._add_compatibility_scores(last_spec)

        self.conn.commit()
        logger.info(f"Added last {last_spec.last_id} to library")

    def _extract_features(self, last_spec: LastSpecification) -> np.ndarray:
        """Extract numerical feature vector for ML matching"""

        features = [
            last_spec.length,
            last_spec.ball_girth,
            last_spec.waist_girth,
            last_spec.instep_girth,
            last_spec.heel_width,
            last_spec.toe_spring,
            last_spec.heel_height,
            last_spec.ball_width,
            last_spec.toe_box_height,
            last_spec.toe_box_width,
            last_spec.arch_length,
            last_spec.heel_cup_depth,
            last_spec.forefoot_rocker_angle,
            last_spec.metatarsal_support_height,
            last_spec.pronation_control,

            # Categorical features as numerical
            {'round': 0, 'square': 1, 'pointed': 2, 'oblique': 3, 'anatomical': 4}.get(last_spec.toe_shape, 0),
            {'low': 0, 'normal': 1, 'high': 2, 'extra_high': 3}.get(last_spec.arch_profile, 1),
            {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'EE': 5, 'EEE': 6, 'EEEE': 7}.get(last_spec.width_code, 3),

            # Medical features
            int(last_spec.bunion_accommodation),
            int(last_spec.hammer_toe_accommodation),
            int(last_spec.diabetic_friendly),
            int(last_spec.extra_depth)
        ]

        return np.array(features, dtype=np.float32)

    def _add_compatibility_scores(self, last_spec: LastSpecification):
        """Add medical condition compatibility scores"""

        cursor = self.conn.cursor()

        # Define compatibility based on last features
        compatibilities = []

        # Bunion compatibility
        if last_spec.bunion_accommodation:
            compatibilities.append((last_spec.last_id, 'bunion', 0.9, 'Built-in bunion accommodation'))
        elif last_spec.toe_box_width > 100:  # Wide toe box
            compatibilities.append((last_spec.last_id, 'bunion', 0.6, 'Wide toe box may help'))

        # Plantar fasciitis compatibility
        if last_spec.arch_profile in ['normal', 'high'] and last_spec.heel_height > 10:
            compatibilities.append((last_spec.last_id, 'plantar_fasciitis', 0.8,
                                  'Good arch support and heel elevation'))

        # Hammer toe compatibility
        if last_spec.hammer_toe_accommodation or last_spec.toe_box_height > 40:
            compatibilities.append((last_spec.last_id, 'hammer_toe', 0.85,
                                  'Extra toe box depth'))

        # Flat feet compatibility
        if last_spec.arch_profile == 'low' and last_spec.pronation_control > 2:
            compatibilities.append((last_spec.last_id, 'flat_feet', 0.9,
                                  'Low arch with pronation control'))

        # Diabetes compatibility
        if last_spec.diabetic_friendly:
            compatibilities.append((last_spec.last_id, 'diabetes', 0.95,
                                  'Diabetic-friendly design'))

        # Swollen feet compatibility
        if last_spec.extra_depth and last_spec.width_code in ['E', 'EE', 'EEE', 'EEEE']:
            compatibilities.append((last_spec.last_id, 'edema', 0.8,
                                  'Extra depth and width'))

        # Insert compatibility scores
        for comp in compatibilities:
            cursor.execute("""
                INSERT OR REPLACE INTO last_compatibility
                (last_id, condition_name, compatibility_score, notes)
                VALUES (?, ?, ?, ?)
            """, comp)

    def _init_matching_model(self):
        """Initialize ML model for intelligent matching"""
        # Use KNN for similarity matching
        self.matching_model = NearestNeighbors(
            n_neighbors=10,
            metric='euclidean',
            algorithm='ball_tree'
        )

        # Load existing feature vectors
        self._update_matching_model()

    def _update_matching_model(self):
        """Update matching model with current database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT last_id, feature_vector FROM last_specifications
            WHERE feature_vector IS NOT NULL
        """)

        last_ids = []
        feature_vectors = []

        for row in cursor.fetchall():
            last_ids.append(row[0])
            feature_vectors.append(pickle.loads(row[1]))

        if feature_vectors:
            X = np.vstack(feature_vectors)
            self.matching_model.fit(X)
            self.feature_cache = dict(zip(last_ids, feature_vectors))
            logger.info(f"Updated matching model with {len(last_ids)} lasts")

    def find_best_matches(self,
                         measurements: Dict,
                         medical_conditions: List[str] = None,
                         style_preference: str = None,
                         k: int = 5) -> List[Tuple[LastSpecification, float]]:
        """
        Find best matching lasts using ML and rules

        Args:
            measurements: Foot measurements
            medical_conditions: List of medical conditions
            style_preference: Preferred style category
            k: Number of matches to return

        Returns:
            List of (LastSpecification, score) tuples
        """

        # Create feature vector from measurements
        query_features = self._measurements_to_features(measurements)

        # Find nearest neighbors
        if self.feature_cache:
            distances, indices = self.matching_model.kneighbors(
                query_features.reshape(1, -1),
                n_neighbors=min(k * 3, len(self.feature_cache))  # Get more for filtering
            )

            # Get last IDs
            last_ids = list(self.feature_cache.keys())
            candidate_ids = [last_ids[i] for i in indices[0]]

            # Score and filter candidates
            scored_matches = []
            for last_id, distance in zip(candidate_ids, distances[0]):
                last_spec = self.get_last(last_id)
                if last_spec:
                    # Calculate comprehensive score
                    score = self._calculate_match_score(
                        last_spec, measurements, medical_conditions, distance
                    )

                    # Apply style filter if specified
                    if style_preference and last_spec.style_category != style_preference:
                        score *= 0.7  # Penalty for style mismatch

                    scored_matches.append((last_spec, score))

            # Sort by score and return top k
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            return scored_matches[:k]

        else:
            logger.warning("No lasts in matching model")
            return []

    def _measurements_to_features(self, measurements: Dict) -> np.ndarray:
        """Convert foot measurements to feature vector"""

        # Map measurements to last features (with defaults)
        features = [
            measurements.get('foot_length', 270),
            measurements.get('ball_girth', 250),
            measurements.get('waist_girth', 230),
            measurements.get('instep_girth', 240),
            measurements.get('heel_width', 65),
            measurements.get('toe_spring', 10),
            measurements.get('heel_height', 20),
            measurements.get('ball_width', 95),
            measurements.get('toe_box_height', 35),
            measurements.get('toe_box_width', 90),
            measurements.get('arch_length', 180),
            measurements.get('heel_cup_depth', 15),
            measurements.get('forefoot_rocker', 5),
            measurements.get('metatarsal_support', 0),
            measurements.get('pronation_control', 0),

            # Toe shape preference (default round)
            measurements.get('toe_shape_pref', 0),
            # Arch type
            {'low': 0, 'normal': 1, 'high': 2}.get(measurements.get('arch_type', 'normal'), 1),
            # Width
            measurements.get('width_category', 3),  # D width default

            # Medical needs
            int(measurements.get('needs_bunion_accommodation', False)),
            int(measurements.get('needs_hammer_toe_space', False)),
            int(measurements.get('diabetic', False)),
            int(measurements.get('needs_extra_depth', False))
        ]

        return np.array(features, dtype=np.float32)

    def _calculate_match_score(self,
                              last_spec: LastSpecification,
                              measurements: Dict,
                              medical_conditions: List[str],
                              distance: float) -> float:
        """Calculate comprehensive match score"""

        # Base score from distance (inverse)
        base_score = 1.0 / (1.0 + distance / 100)

        # Size match score
        size_diff = abs(last_spec.length - measurements.get('foot_length', last_spec.length))
        size_score = max(0, 1.0 - size_diff / 20)

        # Width match score
        width_score = 1.0
        if 'foot_width' in measurements:
            width_diff = abs(last_spec.ball_width - measurements['foot_width'])
            width_score = max(0, 1.0 - width_diff / 15)

        # Medical compatibility score
        medical_score = 1.0
        if medical_conditions:
            cursor = self.conn.cursor()
            total_compatibility = 0
            for condition in medical_conditions:
                cursor.execute("""
                    SELECT compatibility_score FROM last_compatibility
                    WHERE last_id = ? AND condition_name = ?
                """, (last_spec.last_id, condition))
                result = cursor.fetchone()
                if result:
                    total_compatibility += result[0]
                else:
                    total_compatibility += 0.3  # Default low compatibility

            medical_score = total_compatibility / len(medical_conditions) if medical_conditions else 1.0

        # Combine scores with weights
        final_score = (
            base_score * 0.3 +
            size_score * 0.3 +
            width_score * 0.2 +
            medical_score * 0.2
        )

        return final_score

    def get_last(self, last_id: str) -> Optional[LastSpecification]:
        """Retrieve a specific last by ID"""

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM last_specifications WHERE last_id = ?", (last_id,))

        row = cursor.fetchone()
        if row:
            return self._row_to_last_spec(row)
        return None

    def _row_to_last_spec(self, row) -> LastSpecification:
        """Convert database row to LastSpecification"""

        columns = [desc[0] for desc in self.conn.cursor().description]
        data = dict(zip(columns, row))

        # Parse JSON fields
        data['material_zones'] = json.loads(data['material_zones']) if data['material_zones'] else {}
        data['test_feedback'] = json.loads(data['test_feedback']) if data['test_feedback'] else {}

        # Remove non-spec fields
        data.pop('feature_vector', None)
        data.pop('created_at', None)

        # Convert boolean fields
        bool_fields = ['bunion_accommodation', 'hammer_toe_accommodation',
                      'diabetic_friendly', 'removable_insole', 'extra_depth']
        for field in bool_fields:
            if field in data:
                data[field] = bool(data[field])

        return LastSpecification(**data)

    def search_lasts(self, criteria: Dict) -> List[LastSpecification]:
        """Search lasts with multiple criteria"""

        query = "SELECT * FROM last_specifications WHERE 1=1"
        params = []

        if 'size_eu' in criteria:
            query += " AND ABS(size_eu - ?) <= ?"
            params.extend([criteria['size_eu'], criteria.get('size_tolerance', 1)])

        if 'width_code' in criteria:
            query += " AND width_code = ?"
            params.append(criteria['width_code'])

        if 'style_category' in criteria:
            query += " AND style_category = ?"
            params.append(criteria['style_category'])

        if 'medical_needs' in criteria:
            for need in criteria['medical_needs']:
                if need == 'bunion':
                    query += " AND bunion_accommodation = 1"
                elif need == 'diabetic':
                    query += " AND diabetic_friendly = 1"
                elif need == 'extra_depth':
                    query += " AND extra_depth = 1"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        lasts = []
        for row in cursor.fetchall():
            lasts.append(self._row_to_last_spec(row))

        return lasts

    def load_last_3d_model(self, last_id: str) -> Optional[trimesh.Trimesh]:
        """Load 3D mesh model for a last"""

        last_spec = self.get_last(last_id)
        if not last_spec or not last_spec.mesh_file:
            return None

        mesh_path = Path(last_spec.mesh_file)
        if mesh_path.exists():
            try:
                mesh = trimesh.load(str(mesh_path))
                return mesh
            except Exception as e:
                logger.error(f"Failed to load mesh for {last_id}: {e}")
                return None

        return None

    def record_usage(self, last_id: str, scan_id: str,
                    fit_score: float, comfort_score: float,
                    modifications: Dict, feedback: str):
        """Record last usage and feedback"""

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO last_usage
            (last_id, scan_id, usage_date, fit_score, comfort_score,
             modifications_applied, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            last_id, scan_id, datetime.now().isoformat(),
            fit_score, comfort_score,
            json.dumps(modifications), feedback
        ))

        self.conn.commit()
        logger.info(f"Recorded usage of last {last_id}")

    def get_usage_statistics(self, last_id: str) -> Dict:
        """Get usage statistics for a last"""

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total_uses,
                AVG(fit_score) as avg_fit,
                AVG(comfort_score) as avg_comfort,
                MIN(fit_score) as min_fit,
                MAX(fit_score) as max_fit
            FROM last_usage
            WHERE last_id = ?
        """, (last_id,))

        row = cursor.fetchone()
        if row:
            return {
                'total_uses': row[0],
                'average_fit_score': row[1],
                'average_comfort_score': row[2],
                'min_fit_score': row[3],
                'max_fit_score': row[4]
            }

        return {}

    def export_last_data(self, last_id: str, output_dir: str):
        """Export all data for a last"""

        last_spec = self.get_last(last_id)
        if not last_spec:
            logger.error(f"Last {last_id} not found")
            return

        output_path = Path(output_dir) / last_id
        output_path.mkdir(parents=True, exist_ok=True)

        # Export specification as JSON
        with open(output_path / "specification.json", 'w') as f:
            json.dump(asdict(last_spec), f, indent=2)

        # Copy 3D files
        if last_spec.mesh_file and Path(last_spec.mesh_file).exists():
            shutil.copy(last_spec.mesh_file, output_path / "model.stl")

        if last_spec.point_cloud_file and Path(last_spec.point_cloud_file).exists():
            shutil.copy(last_spec.point_cloud_file, output_path / "points.ply")

        # Export usage data
        stats = self.get_usage_statistics(last_id)
        with open(output_path / "usage_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Exported last {last_id} to {output_path}")