"""
Last Matching Module
Matches foot scans to appropriate base lasts from inventory
"""

import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LastRecommendation:
    """Recommendation for base last selection"""
    last_id: str
    confidence: float
    size_eu: float
    size_uk: float
    size_us: float
    match_score: float
    modifications_needed: Dict[str, float]
    reasoning: str

class LastDatabase:
    """Manages last inventory database"""

    def __init__(self, db_path: str = "data/lasts.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        """Create database schema if not exists"""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Create lasts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS base_lasts (
                last_id TEXT PRIMARY KEY,
                size_eu REAL,
                size_uk REAL,
                size_us REAL,
                length_mm REAL,
                ball_girth_mm REAL,
                waist_girth_mm REAL,
                instep_girth_mm REAL,
                heel_width_mm REAL,
                last_type TEXT,
                toe_shape TEXT,
                heel_height_mm REAL,
                geometry_file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_measurements
            ON base_lasts(length_mm, ball_girth_mm)
        """)

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def add_last(self, last_data: Dict):
        """Add a new last to the database"""
        cursor = self.conn.cursor()

        columns = list(last_data.keys())
        values = list(last_data.values())
        placeholders = ','.join(['?' for _ in values])

        query = f"""
            INSERT OR REPLACE INTO base_lasts ({','.join(columns)})
            VALUES ({placeholders})
        """

        cursor.execute(query, values)
        self.conn.commit()

    def populate_sample_data(self):
        """Add sample lasts for testing"""
        sample_lasts = [
            {
                'last_id': 'STD_EU42_M',
                'size_eu': 42,
                'size_uk': 8,
                'size_us': 9,
                'length_mm': 270,
                'ball_girth_mm': 250,
                'waist_girth_mm': 230,
                'instep_girth_mm': 240,
                'heel_width_mm': 65,
                'last_type': 'standard',
                'toe_shape': 'round',
                'heel_height_mm': 25,
                'geometry_file_path': 'lasts/std_42_m.stl'
            },
            {
                'last_id': 'WIDE_EU42_M',
                'size_eu': 42,
                'size_uk': 8,
                'size_us': 9,
                'length_mm': 270,
                'ball_girth_mm': 260,
                'waist_girth_mm': 240,
                'instep_girth_mm': 250,
                'heel_width_mm': 70,
                'last_type': 'wide',
                'toe_shape': 'round',
                'heel_height_mm': 25,
                'geometry_file_path': 'lasts/wide_42_m.stl'
            },
            # Add more sample lasts as needed
        ]

        for last in sample_lasts:
            self.add_last(last)

        logger.info(f"Added {len(sample_lasts)} sample lasts to database")

    def query_lasts(self, length_mm: float, tolerance_mm: float = 10) -> List[Dict]:
        """Query lasts within length tolerance"""
        cursor = self.conn.cursor()

        query = """
            SELECT * FROM base_lasts
            WHERE ABS(length_mm - ?) <= ?
            ORDER BY ABS(length_mm - ?)
        """

        cursor.execute(query, (length_mm, tolerance_mm, length_mm))

        columns = [description[0] for description in cursor.description]
        results = []

        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))

        return results

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class LastMatcher:
    """Matches foot measurements to appropriate base lasts"""

    def __init__(self, database_path: str = "data/lasts.db"):
        """Initialize matcher with database"""
        self.db = LastDatabase(database_path)

        # Scoring weights
        self.weights = {
            'length': 0.3,
            'ball_girth': 0.25,
            'instep_girth': 0.2,
            'waist_girth': 0.15,
            'heel_width': 0.1
        }

    def match(self, measurements: Dict,
             features: Optional[Dict] = None,
             top_k: int = 3) -> List[LastRecommendation]:
        """
        Find best matching lasts for foot measurements

        Args:
            measurements: Foot measurements dictionary
            features: Optional anatomical features (bunion, arch type, etc.)
            top_k: Number of recommendations to return

        Returns:
            List of LastRecommendation objects
        """
        # Query candidate lasts
        candidates = self.db.query_lasts(
            measurements.get('foot_length', 0),
            tolerance_mm=15
        )

        if not candidates:
            logger.warning("No candidate lasts found")
            return []

        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_match_score(measurements, candidate)
            modifications = self._calculate_modifications(
                measurements, candidate, features
            )
            scored_candidates.append((score, candidate, modifications))

        # Sort by score and get top k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        for score, last, modifications in scored_candidates[:top_k]:
            reasoning = self._generate_reasoning(
                measurements, last, score, modifications
            )

            rec = LastRecommendation(
                last_id=last['last_id'],
                confidence=min(score, 1.0),
                size_eu=last['size_eu'],
                size_uk=last['size_uk'],
                size_us=last['size_us'],
                match_score=score,
                modifications_needed=modifications,
                reasoning=reasoning
            )
            recommendations.append(rec)

        return recommendations

    def _calculate_match_score(self, measurements: Dict, last: Dict) -> float:
        """Calculate match score between foot and last"""
        score = 0.0

        # Length match
        length_diff = abs(measurements.get('foot_length', 0) - last['length_mm'])
        length_score = max(0, 1 - length_diff / 10)  # 10mm = 0 score
        score += length_score * self.weights['length']

        # Ball girth match
        ball_diff = abs(measurements.get('ball_girth', 0) - last['ball_girth_mm'])
        ball_score = max(0, 1 - ball_diff / 15)  # 15mm = 0 score
        score += ball_score * self.weights['ball_girth']

        # Instep girth match
        instep_diff = abs(measurements.get('instep_girth', 0) - last['instep_girth_mm'])
        instep_score = max(0, 1 - instep_diff / 15)
        score += instep_score * self.weights['instep_girth']

        # Waist girth match
        waist_diff = abs(measurements.get('waist_girth', 0) - last['waist_girth_mm'])
        waist_score = max(0, 1 - waist_diff / 15)
        score += waist_score * self.weights['waist_girth']

        # Heel width match
        heel_diff = abs(measurements.get('heel_width', 0) - last['heel_width_mm'])
        heel_score = max(0, 1 - heel_diff / 10)
        score += heel_score * self.weights['heel_width']

        return score

    def _calculate_modifications(self, measurements: Dict,
                                last: Dict,
                                features: Optional[Dict]) -> Dict[str, float]:
        """Calculate required modifications to base last"""
        modifications = {}

        # Basic dimensional adjustments
        if measurements.get('ball_girth', 0) > last['ball_girth_mm']:
            modifications['ball_girth_expansion'] = \
                measurements['ball_girth'] - last['ball_girth_mm']

        if measurements.get('instep_girth', 0) > last['instep_girth_mm']:
            modifications['instep_expansion'] = \
                measurements['instep_girth'] - last['instep_girth_mm']

        # Feature-based modifications
        if features:
            if features.get('has_bunion'):
                severity = features.get('bunion_severity', 'mild')
                if severity == 'mild':
                    modifications['bunion_pocket_depth'] = -2.0
                elif severity == 'moderate':
                    modifications['bunion_pocket_depth'] = -3.5
                else:
                    modifications['bunion_pocket_depth'] = -5.0

            if features.get('arch_type') == 'high':
                modifications['arch_support_height'] = 5.0
            elif features.get('arch_type') == 'low':
                modifications['arch_support_height'] = 3.0

            if features.get('high_instep'):
                modifications['instep_relief'] = features.get('instep_excess', 3.0)

        return modifications

    def _generate_reasoning(self, measurements: Dict, last: Dict,
                          score: float, modifications: Dict) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasons = []

        # Size match
        length_diff = measurements.get('foot_length', 0) - last['length_mm']
        if abs(length_diff) < 3:
            reasons.append(f"Excellent length match (within {abs(length_diff):.1f}mm)")
        elif abs(length_diff) < 8:
            reasons.append(f"Good length match (difference: {length_diff:.1f}mm)")
        else:
            reasons.append(f"Length requires adjustment ({length_diff:.1f}mm difference)")

        # Width considerations
        if last['last_type'] == 'wide':
            reasons.append("Wide last selected for better fit")
        elif last['last_type'] == 'narrow':
            reasons.append("Narrow last selected for snug fit")

        # Modifications needed
        if modifications:
            mod_list = []
            for mod_type, value in modifications.items():
                if 'bunion' in mod_type:
                    mod_list.append(f"bunion accommodation ({abs(value):.1f}mm relief)")
                elif 'arch' in mod_type:
                    mod_list.append(f"arch support ({value:.1f}mm)")
                elif 'instep' in mod_type:
                    mod_list.append(f"instep adjustment ({value:.1f}mm)")
                else:
                    mod_list.append(f"{mod_type.replace('_', ' ')} ({value:.1f}mm)")

            reasons.append(f"Modifications: {', '.join(mod_list)}")

        # Confidence
        if score > 0.9:
            reasons.append("Very high confidence match")
        elif score > 0.75:
            reasons.append("High confidence match")
        elif score > 0.6:
            reasons.append("Moderate confidence match")
        else:
            reasons.append("Low confidence - manual review recommended")

        return "; ".join(reasons)