#!/usr/bin/env python3
"""
Medical Research Data Loader
Loads and provides access to evidence-based medical research data collected from PubMed
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

class MedicalResearchLoader:
    """Load and query medical research data from the research agent exports"""

    def __init__(self, research_data_path: str = None):
        """
        Initialize the medical research loader

        Args:
            research_data_path: Path to the medical research agent exports directory
        """
        if research_data_path is None:
            # Default to system data directory
            # Try system data directory first, fall back to Desktop if not found
            system_data_path = os.path.join(os.path.dirname(__file__), '../../data')
            desktop_path = os.path.expanduser("~/Desktop/medical_research_agent/exports")

            if os.path.exists(system_data_path) and os.path.exists(os.path.join(system_data_path, 'pipeline_report.json')):
                research_data_path = system_data_path
            else:
                research_data_path = desktop_path

        self.research_path = Path(research_data_path)
        self.conditions_data = {}
        self.ml_training_data = []
        self.condition_summary = {}
        self.pipeline_report = {}

        # Load all data files
        self._load_data()

    def _load_data(self):
        """Load all medical research data files"""
        try:
            # Load condition summary (lightweight, for quick access)
            summary_file = self.research_path / "condition_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.condition_summary = json.load(f)
                print(f"✓ Loaded {len(self.condition_summary)} conditions from research data")

            # Load ML training data
            ml_file = self.research_path / "ml_training_data.json"
            if ml_file.exists():
                with open(ml_file, 'r') as f:
                    self.ml_training_data = json.load(f)
                print(f"✓ Loaded ML training data with {len(self.ml_training_data)} condition profiles")

            # Load pipeline report
            report_file = self.research_path / "pipeline_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    self.pipeline_report = json.load(f)
                print(f"✓ Loaded pipeline report: {self.pipeline_report.get('collection_stats', {})}")

            # Create indexed conditions data for quick lookup
            self._index_conditions()

        except Exception as e:
            print(f"Warning: Could not load medical research data: {e}")
            print("Continuing with empty research data")

    def _index_conditions(self):
        """Create indexed lookup for conditions"""
        self.conditions_data = {}

        # Index from condition summary
        for condition_name, data in self.condition_summary.items():
            self.conditions_data[condition_name.lower()] = {
                'name': condition_name,
                'icd10_code': data.get('icd10_code', ''),
                'category': data.get('category', 'unknown'),
                'evidence_strength': data.get('evidence_base', {}).get('study_count', 0),
                'source_articles': data.get('evidence_base', {}).get('source_articles', 0),
                'symptoms': data.get('clinical_presentation', {}).get('primary_symptoms', []),
                'treatments': data.get('treatment_options', {}).get('primary_treatments', []),
                'symptom_count': data.get('clinical_presentation', {}).get('symptom_count', 0),
                'treatment_count': data.get('treatment_options', {}).get('treatment_count', 0)
            }

    def get_condition_by_name(self, condition_name: str) -> Optional[Dict]:
        """
        Get condition data by name

        Args:
            condition_name: Name of the condition (case-insensitive)

        Returns:
            Dictionary with condition data or None if not found
        """
        return self.conditions_data.get(condition_name.lower())

    def get_all_conditions(self) -> List[Dict]:
        """Get all conditions with their data"""
        return list(self.conditions_data.values())

    def get_conditions_by_category(self, category: str) -> List[Dict]:
        """
        Get all conditions in a specific category

        Args:
            category: Category name (structural, metabolic, neurological, etc.)

        Returns:
            List of conditions in that category
        """
        return [
            cond for cond in self.conditions_data.values()
            if cond['category'].lower() == category.lower()
        ]

    def get_top_conditions(self, limit: int = 10) -> List[Dict]:
        """
        Get top conditions by evidence strength (number of studies)

        Args:
            limit: Number of top conditions to return

        Returns:
            List of conditions sorted by evidence strength
        """
        conditions = sorted(
            self.conditions_data.values(),
            key=lambda x: x['evidence_strength'],
            reverse=True
        )
        return conditions[:limit]

    def search_conditions_by_symptom(self, symptom: str) -> List[Dict]:
        """
        Find conditions associated with a specific symptom

        Args:
            symptom: Symptom to search for

        Returns:
            List of conditions that include this symptom
        """
        matching_conditions = []
        symptom_lower = symptom.lower()

        for condition in self.conditions_data.values():
            if any(symptom_lower in s.lower() for s in condition['symptoms']):
                matching_conditions.append(condition)

        return matching_conditions

    def get_treatment_recommendations(self, condition_name: str) -> List[str]:
        """
        Get evidence-based treatment recommendations for a condition

        Args:
            condition_name: Name of the condition

        Returns:
            List of treatment options
        """
        condition = self.get_condition_by_name(condition_name)
        if condition:
            return condition['treatments']
        return []

    def get_statistics(self) -> Dict:
        """Get overall statistics about the research data"""
        return {
            'total_conditions': len(self.conditions_data),
            'total_articles': self.pipeline_report.get('collection_stats', {}).get('total_articles', 0),
            'total_symptoms': self.pipeline_report.get('collection_stats', {}).get('unique_symptoms', 0),
            'total_treatments': self.pipeline_report.get('collection_stats', {}).get('unique_treatments', 0),
            'total_relationships': self.pipeline_report.get('collection_stats', {}).get('relationships', 0),
            'categories': self._get_categories()
        }

    def _get_categories(self) -> Dict[str, int]:
        """Get condition count by category"""
        categories = {}
        for condition in self.conditions_data.values():
            cat = condition['category']
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def get_icd10_code(self, condition_name: str) -> Optional[str]:
        """Get ICD-10 code for a condition"""
        condition = self.get_condition_by_name(condition_name)
        return condition['icd10_code'] if condition else None

    def get_evidence_strength(self, condition_name: str) -> int:
        """Get evidence strength (number of supporting studies) for a condition"""
        condition = self.get_condition_by_name(condition_name)
        return condition['evidence_strength'] if condition else 0

    def is_data_loaded(self) -> bool:
        """Check if research data was successfully loaded"""
        return len(self.conditions_data) > 0


# Global singleton instance
_research_loader = None

def get_research_loader(research_data_path: str = None) -> MedicalResearchLoader:
    """Get or create the global research loader instance"""
    global _research_loader
    if _research_loader is None:
        _research_loader = MedicalResearchLoader(research_data_path)
    return _research_loader
