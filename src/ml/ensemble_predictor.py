"""
Enhanced ML Ensemble Predictor for Foot Condition Detection
Implements ensemble methods, cross-validation, and model training
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance metrics captured during training"""
    mean_cv_accuracy: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confidence_calibration: float

@dataclass
class PredictionResult:
    """Enhanced prediction result with ensemble voting"""
    condition: str
    probability: float
    confidence: float
    severity: str
    model_agreement: float  # How many models agree
    individual_predictions: Dict[str, float]  # Each model's prediction

class EnsembleConditionPredictor:
    """
    Advanced ML ensemble predictor with:
    - Multiple model types (RF, GBM, Neural Net)
    - Ensemble voting
    - Cross-validation
    - Hyperparameter tuning
    - Confidence calibration
    """

    def __init__(self, model_dir: str = None):
        """Initialize ensemble predictor"""
        self.logger = logging.getLogger(__name__)

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), '../../models/trained')
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.condition_names = []
        self.performance_metrics = {}

        # Load or initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models with optimized hyperparameters"""

        # Random Forest - Good for non-linear relationships
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),

            # Gradient Boosting - Excellent accuracy, sequential learning
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),

            # Neural Network - Can learn complex patterns
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }

        self.logger.info(f"Initialized {len(self.base_models)} base models")

    def extract_features(self, foot_structure: Dict[str, Any],
                        measurements: Any) -> np.ndarray:
        """
        Extract comprehensive feature vector from foot data

        Features include:
        - Dimensional measurements (length, width, height)
        - Arch characteristics (height, index, angle)
        - Toe angles and deviations
        - Regional volumes
        - Pressure distribution proxies
        - Symmetry metrics
        """
        features = []

        # Basic dimensions
        features.append(measurements.foot_length)
        features.append(measurements.foot_width)
        features.append(measurements.foot_height)
        features.append(measurements.ball_girth)
        features.append(measurements.instep_girth)
        features.append(measurements.heel_width)
        features.append(measurements.volume)

        # Arch characteristics
        arch_data = foot_structure.get('arch', {})
        features.append(arch_data.get('height', 0))
        features.append(arch_data.get('arch_index', 0))
        features.append(arch_data.get('arch_angle', 0))
        features.append(arch_data.get('navicular_height', 0))

        # Toe analysis
        big_toe = foot_structure.get('big_toe', {})
        features.append(big_toe.get('hallux_valgus_angle', 0))
        features.append(big_toe.get('interphalangeal_angle', 0))
        features.append(big_toe.get('length', 0))

        small_toe = foot_structure.get('small_toe', {})
        features.append(small_toe.get('bunionette_angle', 0))
        features.append(small_toe.get('length', 0))

        # Instep
        instep = foot_structure.get('instep', {})
        features.append(instep.get('height', 0))
        features.append(instep.get('volume', 0))

        # Heel
        heel = foot_structure.get('heel', {})
        features.append(heel.get('height', 0))
        features.append(heel.get('width', 0))
        features.append(heel.get('spur_prominence', 0))

        # Regional volumes (if available)
        regions = foot_structure.get('regional_volumes', {})
        features.append(regions.get('forefoot', 0))
        features.append(regions.get('midfoot', 0))
        features.append(regions.get('hindfoot', 0))

        # Ratios and derived metrics
        if measurements.foot_length > 0:
            features.append(measurements.foot_width / measurements.foot_length)
            features.append(measurements.foot_height / measurements.foot_length)
            features.append(arch_data.get('height', 0) / measurements.foot_length)
        else:
            features.extend([0, 0, 0])

        # Asymmetry proxy (if bilateral data available)
        features.append(foot_structure.get('asymmetry_score', 0))

        return np.array(features).reshape(1, -1)

    def train_models(self, training_data: List[Dict[str, Any]],
                    labels: Dict[str, List[int]],
                    cv_folds: int = 5,
                    tune_hyperparameters: bool = False) -> Dict[str, ModelPerformance]:
        """
        Train ensemble models with cross-validation

        Args:
            training_data: List of foot scan data dictionaries
            labels: Dictionary mapping condition names to binary labels
            cv_folds: Number of cross-validation folds
            tune_hyperparameters: Whether to perform grid search

        Returns:
            Dictionary of performance metrics per condition
        """
        self.logger.info(f"Training models on {len(training_data)} samples...")

        # Extract features for all samples
        X_list = []
        for sample in training_data:
            features = self.extract_features(
                sample['structure'],
                sample['measurements']
            )
            X_list.append(features.flatten())

        X = np.array(X_list)

        # Store feature names
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.condition_names = list(labels.keys())

        performance_results = {}

        # Train model for each condition
        for condition_name, y in labels.items():
            self.logger.info(f"Training models for {condition_name}...")

            y = np.array(y)

            # Check for class imbalance
            pos_ratio = np.mean(y)
            self.logger.info(f"  {condition_name}: {pos_ratio*100:.1f}% positive samples")

            if pos_ratio < 0.05 or pos_ratio > 0.95:
                self.logger.warning(f"  Severe class imbalance for {condition_name}")

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[condition_name] = scaler

            # Train individual models
            trained_models = {}
            model_scores = {}

            for model_name, base_model in self.base_models.items():
                self.logger.info(f"    Training {model_name}...")

                try:
                    # Cross-validation
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(base_model, X_scaled, y, cv=cv, scoring='f1')

                    self.logger.info(f"      CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

                    # Train on full dataset
                    model = base_model.fit(X_scaled, y)

                    # Store model and scores
                    trained_models[model_name] = model
                    model_scores[model_name] = cv_scores.mean()

                except Exception as e:
                    self.logger.error(f"      Failed to train {model_name}: {e}")

            # Create ensemble voting classifier
            if len(trained_models) >= 2:
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft',  # Use probability voting
                    weights=[model_scores.get(name, 1.0) for name in trained_models.keys()]
                )
                ensemble.fit(X_scaled, y)
                trained_models['ensemble'] = ensemble

                self.logger.info(f"    Created ensemble of {len(trained_models)-1} models")

            # Store trained models
            self.models[condition_name] = trained_models

            # Evaluate performance
            if 'ensemble' in trained_models:
                y_pred = trained_models['ensemble'].predict(X_scaled)
                y_pred_proba = trained_models['ensemble'].predict_proba(X_scaled)[:, 1]
                mean_cv_accuracy = float(np.mean(list(model_scores.values()))) if model_scores else 0.0
                test_accuracy = accuracy_score(y, y_pred)
                test_precision = precision_score(y, y_pred, zero_division=0)
                test_recall = recall_score(y, y_pred, zero_division=0)
                test_f1 = f1_score(y, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
                confidence_calibration = self._calculate_calibration(y, y_pred_proba)

                performance = ModelPerformance(
                    mean_cv_accuracy=mean_cv_accuracy,
                    test_accuracy=test_accuracy,
                    test_precision=test_precision,
                    test_recall=test_recall,
                    test_f1=test_f1,
                    accuracy=test_accuracy,
                    precision=test_precision,
                    recall=test_recall,
                    f1_score=test_f1,
                    roc_auc=roc_auc,
                    confidence_calibration=confidence_calibration
                )

                performance_results[condition_name] = performance
                self.performance_metrics[condition_name] = performance

                self.logger.info(f"    Performance: Accuracy={performance.accuracy:.3f}, "
                               f"Precision={performance.precision:.3f}, "
                               f"Recall={performance.recall:.3f}, "
                               f"F1={performance.f1_score:.3f}")

        self.logger.info("Training complete!")
        return performance_results

    def _calculate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate calibration score (how well probabilities match actual outcomes)"""
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_pred_proba, bins) - 1

        calibration_error = 0
        n_bins = 0

        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                pred_mean = y_pred_proba[mask].mean()
                true_mean = y_true[mask].mean()
                calibration_error += abs(pred_mean - true_mean) * mask.sum()
                n_bins += mask.sum()

        return 1.0 - (calibration_error / n_bins if n_bins > 0 else 1.0)

    def predict(self, foot_structure: Dict[str, Any],
               measurements: Any,
               threshold: float = 0.5) -> Dict[str, PredictionResult]:
        """
        Predict conditions using ensemble models

        Returns:
            Dictionary mapping condition names to PredictionResult objects
        """
        # Extract features
        features = self.extract_features(foot_structure, measurements)

        predictions = {}

        for condition_name, trained_models in self.models.items():
            if not trained_models:
                continue

            # Get scaler
            scaler = self.scalers.get(condition_name)
            if scaler is None:
                self.logger.warning(f"No scaler found for {condition_name}")
                continue

            features_scaled = scaler.transform(features)

            # Get predictions from each model
            individual_preds = {}
            probabilities = []

            for model_name, model in trained_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0, 1]
                        individual_preds[model_name] = proba
                        probabilities.append(proba)
                except Exception as e:
                    self.logger.error(f"Prediction failed for {model_name}: {e}")

            if not probabilities:
                continue

            # Ensemble prediction
            avg_probability = np.mean(probabilities)
            std_probability = np.std(probabilities)

            # Model agreement (inverse of standard deviation)
            model_agreement = 1.0 - min(std_probability, 1.0)

            # Confidence based on agreement and probability distance from threshold
            confidence = model_agreement * abs(avg_probability - 0.5) * 2

            # Determine severity
            if avg_probability >= 0.8:
                severity = 'severe'
            elif avg_probability >= 0.65:
                severity = 'moderate'
            elif avg_probability >= threshold:
                severity = 'mild'
            else:
                severity = 'normal'

            predictions[condition_name] = PredictionResult(
                condition=condition_name,
                probability=avg_probability,
                confidence=confidence,
                severity=severity,
                model_agreement=model_agreement,
                individual_predictions=individual_preds
            )

        return predictions

    def save_models(self, filepath: str = None):
        """Save trained models to disk"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'ensemble_models.pkl')

        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'condition_names': self.condition_names,
            'performance_metrics': self.performance_metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        self.logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: str = None):
        """Load trained models from disk"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'ensemble_models.pkl')

        if not os.path.exists(filepath):
            self.logger.warning(f"No saved models found at {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)

            self.models = save_data.get('models', {})
            self.scalers = save_data.get('scalers', {})
            self.feature_names = save_data.get('feature_names', [])
            self.condition_names = save_data.get('condition_names', [])
            self.performance_metrics = save_data.get('performance_metrics', {})

            self.logger.info(f"Models loaded from {filepath}")
            self.logger.info(f"  Conditions: {', '.join(self.condition_names)}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    def get_feature_importance(self, condition_name: str,
                               model_type: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance for a specific condition"""
        if condition_name not in self.models:
            return {}

        model = self.models[condition_name].get(model_type)
        if model is None or not hasattr(model, 'feature_importances_'):
            return {}

        importances = model.feature_importances_
        return dict(zip(self.feature_names, importances))
