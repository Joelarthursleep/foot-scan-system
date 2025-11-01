#!/usr/bin/env python3
"""
Research-Based ML Training Pipeline
Train ML models using evidence-based medical research data
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os


class ResearchBasedMLTrainer:
    """Train ML models for foot condition classification using research data"""

    def __init__(self, research_data_path: str = None):
        """
        Initialize the ML trainer

        Args:
            research_data_path: Path to medical research exports directory
        """
        if research_data_path is None:
            research_data_path = os.path.expanduser("~/Desktop/medical_research_agent/exports")

        self.research_path = Path(research_data_path)
        self.models = {}
        self.scalers = {}
        self.condition_labels = []
        self.training_data = None

        # Load research data
        self._load_research_data()

    def _load_research_data(self):
        """Load medical research training data"""
        try:
            ml_file = self.research_path / "ml_training_data.json"
            if ml_file.exists():
                with open(ml_file, 'r') as f:
                    self.training_data = json.load(f)
                print(f"✓ Loaded training data for {len(self.training_data)} conditions")

                # Extract condition labels
                self.condition_labels = [item['condition'] for item in self.training_data]
                print(f"✓ Condition labels: {len(set(self.condition_labels))} unique")
            else:
                print(f"Warning: Training data not found at {ml_file}")
        except Exception as e:
            print(f"Error loading research data: {e}")

    def prepare_synthetic_training_data(
        self,
        n_samples_per_condition: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic training data based on research evidence

        Since we don't have actual 3D scan data yet, we create synthetic examples
        based on the clinical characteristics from research

        Args:
            n_samples_per_condition: Number of synthetic samples per condition

        Returns:
            Tuple of (X_features, y_labels, label_names)
        """
        if not self.training_data:
            raise ValueError("No training data loaded")

        X_samples = []
        y_labels = []
        label_names = []

        print("\n" + "="*70)
        print("GENERATING SYNTHETIC TRAINING DATA")
        print("="*70)

        for idx, condition_data in enumerate(self.training_data):
            condition_name = condition_data['condition']
            category = condition_data['category']
            evidence_strength = condition_data['evidence_strength']

            print(f"\n[{idx+1}/{len(self.training_data)}] {condition_name} ({category})")
            print(f"  Evidence strength: {evidence_strength} studies")

            # Generate synthetic features based on condition type
            for _ in range(n_samples_per_condition):
                features = self._generate_synthetic_features(condition_name, category)
                X_samples.append(features)
                y_labels.append(idx)

            label_names.append(condition_name)
            print(f"  ✓ Generated {n_samples_per_condition} synthetic samples")

        X = np.array(X_samples)
        y = np.array(y_labels)

        print("\n" + "="*70)
        print(f"✓ Total training samples: {len(X)}")
        print(f"✓ Feature dimensions: {X.shape[1]}")
        print(f"✓ Condition classes: {len(label_names)}")
        print("="*70 + "\n")

        return X, y, label_names

    def _generate_synthetic_features(self, condition_name: str, category: str) -> np.ndarray:
        """
        Generate synthetic 3D scan features for a condition

        Features (15 total):
        0: arch_height (mm)
        1: arch_index (ratio)
        2: hallux_angle (degrees)
        3: first_metatarsal_angle (degrees)
        4: toe_angle_avg (degrees)
        5: heel_prominence (mm)
        6: heel_width_ratio
        7: heel_angle (degrees)
        8: forefoot_pressure_ratio
        9: metatarsal_space_avg (mm)
        10: left_right_asymmetry (ratio)
        11: arch_strain_index
        12: foot_length (mm)
        13: foot_width (mm)
        14: volume (cm³)
        """
        # Base healthy foot values
        features = np.array([
            22.0,  # arch_height
            0.30,  # arch_index
            10.0,  # hallux_angle
            7.0,   # first_metatarsal_angle
            5.0,   # toe_angle_avg
            1.5,   # heel_prominence
            1.0,   # heel_width_ratio
            85.0,  # heel_angle
            1.0,   # forefoot_pressure_ratio
            10.0,  # metatarsal_space_avg
            0.05,  # left_right_asymmetry
            0.10,  # arch_strain_index
            250.0, # foot_length
            95.0,  # foot_width
            800.0  # volume
        ])

        # Add condition-specific variations with noise
        condition_lower = condition_name.lower()

        if 'hallux valgus' in condition_lower or 'bunion' in condition_lower:
            features[2] = np.random.normal(25, 8)  # hallux_angle
            features[3] = np.random.normal(13, 3)  # first_metatarsal_angle
            features[13] = np.random.normal(100, 5)  # wider forefoot

        elif 'pes planus' in condition_lower or 'flat foot' in condition_lower:
            features[0] = np.random.normal(10, 3)  # low arch_height
            features[1] = np.random.normal(0.20, 0.05)  # low arch_index
            features[11] = np.random.normal(0.25, 0.08)  # high strain

        elif 'pes cavus' in condition_lower or 'high arch' in condition_lower:
            features[0] = np.random.normal(35, 5)  # high arch_height
            features[1] = np.random.normal(0.42, 0.08)  # high arch_index
            features[8] = np.random.normal(1.4, 0.2)  # higher forefoot pressure

        elif 'plantar fasciitis' in condition_lower:
            features[6] = np.random.normal(1.15, 0.10)  # heel_width_ratio
            features[11] = np.random.normal(0.30, 0.10)  # arch_strain
            features[5] = np.random.normal(2.5, 0.8)  # heel_prominence

        elif 'metatarsalgia' in condition_lower:
            features[8] = np.random.normal(1.5, 0.3)  # forefoot_pressure_ratio
            features[9] = np.random.normal(7, 1.5)  # reduced metatarsal_space

        elif 'morton' in condition_lower:
            features[9] = np.random.normal(6, 1.0)  # narrow metatarsal_space
            features[8] = np.random.normal(1.3, 0.2)  # increased pressure

        elif 'hammer' in condition_lower or 'claw toe' in condition_lower:
            features[4] = np.random.normal(35, 10)  # high toe_angle_avg

        elif 'achilles' in condition_lower:
            features[7] = np.random.normal(75, 5)  # reduced heel_angle
            features[5] = np.random.normal(3, 1)  # heel_prominence

        elif 'spur' in condition_lower:
            features[5] = np.random.normal(4, 1.2)  # high heel_prominence

        elif 'diabetic foot' in condition_lower:
            features[10] = np.random.normal(0.15, 0.08)  # asymmetry
            features[11] = np.random.normal(0.20, 0.08)  # strain
            features[14] = np.random.normal(850, 50)  # increased volume

        elif 'stress fracture' in condition_lower:
            features[10] = np.random.normal(0.20, 0.10)  # high asymmetry

        elif 'gout' in condition_lower:
            features[2] = np.random.normal(15, 5)  # hallux involvement
            features[14] = np.random.normal(820, 40)  # swelling

        # Add random noise to all features (±5%)
        noise = np.random.normal(0, 0.05, size=features.shape)
        features = features * (1 + noise)

        # Ensure realistic bounds
        features = np.clip(features, [0, 0, 0, 0, 0, 0, 0.8, 60, 0.5, 0, 0, 0, 200, 70, 500],
                                     [50, 0.6, 60, 40, 90, 15, 2.0, 95, 3.0, 15, 0.5, 0.8, 300, 120, 1200])

        return features

    def train_models(
        self,
        n_samples_per_condition: int = 100,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train ML models on synthetic research-based data

        Args:
            n_samples_per_condition: Number of samples to generate per condition
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training results and metrics
        """
        print("\n" + "="*70)
        print("TRAINING ML MODELS")
        print("="*70 + "\n")

        # Generate training data
        X, y, label_names = self.prepare_synthetic_training_data(n_samples_per_condition)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples\n")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        # Train Random Forest
        print("Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model

        # Evaluate
        rf_train_score = rf_model.score(X_train_scaled, y_train)
        rf_test_score = rf_model.score(X_test_scaled, y_test)
        print(f"  ✓ Train accuracy: {rf_train_score:.3f}")
        print(f"  ✓ Test accuracy: {rf_test_score:.3f}\n")

        # Train Gradient Boosting
        print("Training Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gradient_boosting'] = gb_model

        # Evaluate
        gb_train_score = gb_model.score(X_train_scaled, y_train)
        gb_test_score = gb_model.score(X_test_scaled, y_test)
        print(f"  ✓ Train accuracy: {gb_train_score:.3f}")
        print(f"  ✓ Test accuracy: {gb_test_score:.3f}\n")

        # Save models
        self.label_names = label_names
        self._save_models()

        print("="*70)
        print("✓ MODEL TRAINING COMPLETE")
        print("="*70 + "\n")

        return {
            'label_names': label_names,
            'n_conditions': len(label_names),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'rf_train_accuracy': rf_train_score,
            'rf_test_accuracy': rf_test_score,
            'gb_train_accuracy': gb_train_score,
            'gb_test_accuracy': gb_test_score,
            'feature_names': self._get_feature_names()
        }

    def _get_feature_names(self) -> List[str]:
        """Get feature names"""
        return [
            'arch_height', 'arch_index', 'hallux_angle', 'first_metatarsal_angle',
            'toe_angle_avg', 'heel_prominence', 'heel_width_ratio', 'heel_angle',
            'forefoot_pressure_ratio', 'metatarsal_space_avg', 'left_right_asymmetry',
            'arch_strain_index', 'foot_length', 'foot_width', 'volume'
        ]

    def _save_models(self):
        """Save trained models to disk"""
        model_dir = Path(__file__).parent.parent.parent / "models"
        model_dir.mkdir(exist_ok=True)

        # Save models
        for name, model in self.models.items():
            model_path = model_dir / f"{name}_condition_classifier.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {name} model to {model_path}")

        # Save scaler
        scaler_path = model_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['main'], f)
        print(f"✓ Saved feature scaler to {scaler_path}")

        # Save label names
        labels_path = model_dir / "condition_labels.json"
        with open(labels_path, 'w') as f:
            json.dump(self.label_names, f, indent=2)
        print(f"✓ Saved condition labels to {labels_path}")

    def predict_condition(
        self,
        features: np.ndarray,
        model_name: str = 'random_forest'
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict condition from features

        Args:
            features: Array of 15 foot measurement features
            model_name: Which model to use ('random_forest' or 'gradient_boosting')

        Returns:
            Tuple of (predicted_condition, confidence, top_5_predictions)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        # Scale features
        features_scaled = self.scalers['main'].transform(features.reshape(1, -1))

        # Get prediction
        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Get top 5 predictions
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_predictions = [
            (self.label_names[idx], float(probabilities[idx]))
            for idx in top_5_idx
        ]

        predicted_condition = self.label_names[prediction]
        confidence = float(probabilities[prediction])

        return predicted_condition, confidence, top_5_predictions


def train_research_models(n_samples: int = 100) -> Dict:
    """
    Convenience function to train all models

    Args:
        n_samples: Number of samples per condition

    Returns:
        Training results dictionary
    """
    trainer = ResearchBasedMLTrainer()
    results = trainer.train_models(n_samples_per_condition=n_samples)
    return results
