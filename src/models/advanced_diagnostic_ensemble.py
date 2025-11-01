"""
Advanced Diagnostic Models Module
Implements ensemble learning, federated learning, uncertainty quantification,
and explainable AI for enhanced medical diagnosis accuracy
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class DiagnosticPrediction:
    """Enhanced prediction with uncertainty and explainability"""
    condition_name: str
    probability: float
    confidence_interval: Tuple[float, float]
    uncertainty: float
    feature_importance: Dict[str, float]
    explanation: str
    model_consensus: Dict[str, float]
    risk_factors: List[str]
    evidence_strength: str  # 'strong', 'moderate', 'weak'

@dataclass
class EnsembleResult:
    """Result from ensemble prediction"""
    predictions: List[DiagnosticPrediction]
    overall_confidence: float
    model_agreement: float
    conflicting_diagnoses: List[Tuple[str, str]]
    recommended_actions: List[str]
    uncertainty_flags: List[str]

class BaseClassifier(ABC):
    """Base class for medical condition classifiers"""

    def __init__(self, condition_name: str):
        self.condition_name = condition_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_history = []

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the classifier"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass

class RandomForestMedicalClassifier(BaseClassifier):
    """Random Forest classifier for medical conditions"""

    def __init__(self, condition_name: str, **kwargs):
        super().__init__(condition_name)
        self.model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=42
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train Random Forest model"""
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True

        # Store training metrics
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X)
        })

        logger.info(f"RF trained for {self.condition_name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with Random Forest"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            return {}

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))

class GradientBoostingMedicalClassifier(BaseClassifier):
    """Gradient Boosting classifier for medical conditions"""

    def __init__(self, condition_name: str, **kwargs):
        super().__init__(condition_name)
        self.model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 6),
            random_state=42
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train Gradient Boosting model"""
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True

        # Store training metrics
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X)
        })

        logger.info(f"GB trained for {self.condition_name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with Gradient Boosting"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Gradient Boosting"""
        if not self.is_trained:
            return {}

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))

class NeuralNetworkMedicalClassifier(BaseClassifier):
    """Neural Network classifier for medical conditions"""

    def __init__(self, condition_name: str, **kwargs):
        super().__init__(condition_name)
        self.model = MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (64, 32)),
            activation=kwargs.get('activation', 'relu'),
            solver=kwargs.get('solver', 'adam'),
            alpha=kwargs.get('alpha', 0.001),
            learning_rate=kwargs.get('learning_rate', 'adaptive'),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=42
        )

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train Neural Network model"""
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.is_trained = True

        # Store training metrics
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X)
        })

        logger.info(f"NN trained for {self.condition_name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with Neural Network"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Approximate feature importance for Neural Network using permutation"""
        if not self.is_trained:
            return {}

        # Simplified feature importance (would use SHAP in production)
        # For now, return uniform importance
        uniform_importance = 1.0 / len(self.feature_names)
        return {name: uniform_importance for name in self.feature_names}

class UncertaintyQuantifier:
    """Quantifies prediction uncertainty using multiple methods"""

    def __init__(self):
        self.methods = ['ensemble_variance', 'prediction_entropy', 'confidence_interval']

    def calculate_uncertainty(self, predictions: List[np.ndarray],
                            method: str = 'ensemble_variance') -> Dict[str, float]:
        """Calculate uncertainty using specified method"""

        if method == 'ensemble_variance':
            return self._ensemble_variance(predictions)
        elif method == 'prediction_entropy':
            return self._prediction_entropy(predictions)
        elif method == 'confidence_interval':
            return self._confidence_interval(predictions)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

    def _ensemble_variance(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """Calculate uncertainty from ensemble prediction variance"""
        if not predictions:
            return {'uncertainty': 1.0, 'method': 'ensemble_variance'}

        # Stack predictions and calculate variance
        pred_stack = np.stack(predictions, axis=0)
        variance = np.var(pred_stack, axis=0)

        # For binary classification, take variance of positive class
        if pred_stack.shape[-1] == 2:
            uncertainty = variance[:, 1]  # Variance in positive class
        else:
            uncertainty = np.mean(variance, axis=1)  # Mean variance across classes

        return {
            'uncertainty': float(np.mean(uncertainty)),
            'uncertainty_std': float(np.std(uncertainty)),
            'method': 'ensemble_variance'
        }

    def _prediction_entropy(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """Calculate uncertainty from prediction entropy"""
        if not predictions:
            return {'uncertainty': 1.0, 'method': 'prediction_entropy'}

        # Average predictions across ensemble
        mean_pred = np.mean(np.stack(predictions, axis=0), axis=0)

        # Calculate entropy
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)

        return {
            'uncertainty': float(np.mean(entropy)),
            'uncertainty_std': float(np.std(entropy)),
            'method': 'prediction_entropy'
        }

    def _confidence_interval(self, predictions: List[np.ndarray]) -> Dict[str, float]:
        """Calculate confidence intervals from ensemble predictions"""
        if not predictions:
            return {'uncertainty': 1.0, 'method': 'confidence_interval'}

        pred_stack = np.stack(predictions, axis=0)

        # Calculate confidence intervals (95%)
        ci_lower = np.percentile(pred_stack, 2.5, axis=0)
        ci_upper = np.percentile(pred_stack, 97.5, axis=0)

        # Uncertainty is width of confidence interval
        ci_width = ci_upper - ci_lower

        return {
            'uncertainty': float(np.mean(ci_width)),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'method': 'confidence_interval'
        }

class ExplainableAI:
    """Provides explanations for medical diagnoses"""

    def __init__(self):
        self.explanation_templates = {
            'high_confidence': "Strong evidence for {condition} based on {key_features}. Model confidence: {confidence:.1%}",
            'moderate_confidence': "Moderate evidence for {condition}. Key indicators: {key_features}. Consider additional testing.",
            'low_confidence': "Weak evidence for {condition}. Insufficient reliable indicators detected.",
            'conflicting': "Conflicting evidence detected. Models disagree on {condition}. Manual review recommended."
        }

    def generate_explanation(self, prediction: DiagnosticPrediction,
                           feature_values: Dict[str, float]) -> str:
        """Generate human-readable explanation for diagnosis"""

        # Determine explanation type
        if prediction.probability > 0.8 and prediction.uncertainty < 0.2:
            template_key = 'high_confidence'
        elif prediction.probability > 0.6:
            template_key = 'moderate_confidence'
        elif prediction.probability > 0.4:
            template_key = 'low_confidence'
        else:
            template_key = 'conflicting'

        # Get top contributing features
        top_features = sorted(
            prediction.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        key_features = []
        for feature_name, importance in top_features:
            if feature_name in feature_values:
                value = feature_values[feature_name]
                key_features.append(f"{feature_name} ({value:.1f})")
            else:
                key_features.append(feature_name)

        # Format explanation
        explanation = self.explanation_templates[template_key].format(
            condition=prediction.condition_name,
            key_features=", ".join(key_features),
            confidence=prediction.probability
        )

        # Add uncertainty information
        if prediction.uncertainty > 0.3:
            explanation += f" High uncertainty detected ({prediction.uncertainty:.1%}). "
            explanation += "Consider multiple diagnostic approaches."

        return explanation

    def generate_risk_factors(self, feature_importance: Dict[str, float],
                            feature_values: Dict[str, float]) -> List[str]:
        """Generate list of risk factors"""

        risk_factors = []
        risk_thresholds = {
            'arch_height_mm': (8, 'low', 'Collapsed arch increases risk'),
            'ankle_girth_mm': (250, 'high', 'Swelling indicates inflammation'),
            'flexion_angle': (30, 'high', 'Toe deformity present'),
            'inflammation_score': (0.5, 'high', 'Active inflammation detected')
        }

        for feature, (threshold, direction, description) in risk_thresholds.items():
            if feature in feature_values and feature in feature_importance:
                value = feature_values[feature]
                importance = feature_importance[feature]

                # Only include if feature is important and threshold is met
                if importance > 0.1:  # Minimum importance threshold
                    if direction == 'low' and value < threshold:
                        risk_factors.append(description)
                    elif direction == 'high' and value > threshold:
                        risk_factors.append(description)

        return risk_factors

class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple sites"""

    def __init__(self, site_id: str):
        self.site_id = site_id
        self.local_models = {}
        self.global_model_weights = {}
        self.update_history = []

    def add_local_model(self, condition_name: str, model: BaseClassifier):
        """Add a local model for federated learning"""
        self.local_models[condition_name] = model

    def prepare_model_update(self, condition_name: str) -> Dict:
        """Prepare model weights for federated aggregation"""
        if condition_name not in self.local_models:
            raise ValueError(f"No local model for {condition_name}")

        model = self.local_models[condition_name]

        # Extract model parameters (simplified for demo)
        if hasattr(model.model, 'feature_importances_'):
            weights = model.model.feature_importances_.tolist()
        elif hasattr(model.model, 'coefs_'):
            weights = [layer.tolist() for layer in model.model.coefs_]
        else:
            weights = []

        update = {
            'site_id': self.site_id,
            'condition_name': condition_name,
            'weights': weights,
            'n_samples': len(model.training_history),
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__
        }

        self.update_history.append(update)
        return update

    def apply_global_update(self, condition_name: str, global_weights: Dict):
        """Apply global model update from federated aggregation"""
        self.global_model_weights[condition_name] = global_weights

        logger.info(f"Applied global update for {condition_name} from {len(global_weights.get('contributing_sites', []))} sites")

    def save_federated_state(self, filepath: str):
        """Save federated learning state"""
        state = {
            'site_id': self.site_id,
            'global_model_weights': self.global_model_weights,
            'update_history': self.update_history,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_federated_state(self, filepath: str):
        """Load federated learning state"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
                self.global_model_weights = state.get('global_model_weights', {})
                self.update_history = state.get('update_history', [])

class AdvancedDiagnosticEnsemble:
    """Main ensemble system combining multiple advanced diagnostic approaches"""

    def __init__(self, site_id: str = "default_site"):
        self.classifiers = {}
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.explainable_ai = ExplainableAI()
        self.federated_coordinator = FederatedLearningCoordinator(site_id)
        self.ensemble_weights = {}

    def add_classifier(self, condition_name: str, classifier_type: str, **kwargs):
        """Add a classifier for a specific condition"""

        if condition_name not in self.classifiers:
            self.classifiers[condition_name] = []

        if classifier_type == 'random_forest':
            classifier = RandomForestMedicalClassifier(condition_name, **kwargs)
        elif classifier_type == 'gradient_boosting':
            classifier = GradientBoostingMedicalClassifier(condition_name, **kwargs)
        elif classifier_type == 'neural_network':
            classifier = NeuralNetworkMedicalClassifier(condition_name, **kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.classifiers[condition_name].append(classifier)

        # Add to federated learning
        self.federated_coordinator.add_local_model(condition_name, classifier)

        logger.info(f"Added {classifier_type} classifier for {condition_name}")

    def train_ensemble(self, condition_name: str, X: np.ndarray, y: np.ndarray,
                      feature_names: List[str]):
        """Train all classifiers for a condition"""

        if condition_name not in self.classifiers:
            raise ValueError(f"No classifiers registered for {condition_name}")

        for classifier in self.classifiers[condition_name]:
            classifier.train(X, y, feature_names)

        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(condition_name, X, y)

        logger.info(f"Trained ensemble for {condition_name} with {len(self.classifiers[condition_name])} classifiers")

    def predict_condition(self, condition_name: str, X: np.ndarray,
                         feature_values: Dict[str, float]) -> DiagnosticPrediction:
        """Make ensemble prediction for a specific condition"""

        if condition_name not in self.classifiers:
            raise ValueError(f"No trained classifiers for {condition_name}")

        classifiers = self.classifiers[condition_name]
        predictions = []
        model_consensus = {}

        # Get predictions from all models
        for classifier in classifiers:
            if classifier.is_trained:
                pred_proba = classifier.predict_proba(X)
                predictions.append(pred_proba)

                # Store individual model prediction
                model_type = type(classifier).__name__
                model_consensus[model_type] = float(pred_proba[0, 1] if pred_proba.shape[1] > 1 else pred_proba[0, 0])

        if not predictions:
            raise ValueError(f"No trained classifiers available for {condition_name}")

        # Calculate ensemble prediction
        weights = self.ensemble_weights.get(condition_name, [1.0] * len(predictions))
        weighted_pred = np.average(np.stack(predictions, axis=0), axis=0, weights=weights)

        # Get probability for positive class
        probability = float(weighted_pred[0, 1] if weighted_pred.shape[1] > 1 else weighted_pred[0, 0])

        # Calculate uncertainty
        uncertainty_info = self.uncertainty_quantifier.calculate_uncertainty(predictions)
        uncertainty = uncertainty_info['uncertainty']

        # Calculate confidence interval
        ci_info = self.uncertainty_quantifier.calculate_uncertainty(predictions, 'confidence_interval')
        confidence_interval = (
            float(ci_info['ci_lower'][0][1] if len(ci_info['ci_lower'][0]) > 1 else ci_info['ci_lower'][0][0]),
            float(ci_info['ci_upper'][0][1] if len(ci_info['ci_upper'][0]) > 1 else ci_info['ci_upper'][0][0])
        )

        # Get feature importance (average across models)
        feature_importance = self._aggregate_feature_importance(condition_name)

        # Generate explanation
        prediction_obj = DiagnosticPrediction(
            condition_name=condition_name,
            probability=probability,
            confidence_interval=confidence_interval,
            uncertainty=uncertainty,
            feature_importance=feature_importance,
            explanation="",  # Will be filled by explainable AI
            model_consensus=model_consensus,
            risk_factors=[],  # Will be filled by explainable AI
            evidence_strength=self._determine_evidence_strength(probability, uncertainty)
        )

        # Generate explanation and risk factors
        prediction_obj.explanation = self.explainable_ai.generate_explanation(
            prediction_obj, feature_values
        )
        prediction_obj.risk_factors = self.explainable_ai.generate_risk_factors(
            feature_importance, feature_values
        )

        return prediction_obj

    def predict_all_conditions(self, X: np.ndarray,
                             feature_values: Dict[str, float]) -> EnsembleResult:
        """Predict all available conditions"""

        predictions = []
        conflicting_diagnoses = []
        uncertainty_flags = []

        for condition_name in self.classifiers.keys():
            try:
                prediction = self.predict_condition(condition_name, X, feature_values)
                predictions.append(prediction)

                # Check for high uncertainty
                if prediction.uncertainty > 0.4:
                    uncertainty_flags.append(f"High uncertainty for {condition_name}")

                # Check for model disagreement
                consensus_values = list(prediction.model_consensus.values())
                if len(consensus_values) > 1:
                    consensus_std = np.std(consensus_values)
                    if consensus_std > 0.3:
                        conflicting_diagnoses.append((condition_name, "Model disagreement detected"))

            except Exception as e:
                logger.warning(f"Failed to predict {condition_name}: {e}")

        # Calculate overall confidence and agreement
        overall_confidence = self._calculate_overall_confidence(predictions)
        model_agreement = self._calculate_model_agreement(predictions)

        # Generate recommendations
        recommended_actions = self._generate_recommendations(predictions, uncertainty_flags)

        return EnsembleResult(
            predictions=predictions,
            overall_confidence=overall_confidence,
            model_agreement=model_agreement,
            conflicting_diagnoses=conflicting_diagnoses,
            recommended_actions=recommended_actions,
            uncertainty_flags=uncertainty_flags
        )

    def _calculate_ensemble_weights(self, condition_name: str, X: np.ndarray, y: np.ndarray):
        """Calculate weights for ensemble based on cross-validation performance"""

        classifiers = self.classifiers[condition_name]
        weights = []

        for classifier in classifiers:
            if classifier.is_trained:
                # Use cross-validation score as weight
                cv_scores = cross_val_score(classifier.model, X, y, cv=3)
                weight = cv_scores.mean()
                weights.append(max(0.1, weight))  # Minimum weight of 0.1
            else:
                weights.append(0.1)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(classifiers)] * len(classifiers)

        self.ensemble_weights[condition_name] = weights

    def _aggregate_feature_importance(self, condition_name: str) -> Dict[str, float]:
        """Aggregate feature importance across all models for a condition"""

        classifiers = self.classifiers[condition_name]
        aggregated_importance = {}

        for classifier in classifiers:
            if classifier.is_trained:
                importance = classifier.get_feature_importance()

                for feature, score in importance.items():
                    if feature not in aggregated_importance:
                        aggregated_importance[feature] = []
                    aggregated_importance[feature].append(score)

        # Average importance scores
        final_importance = {}
        for feature, scores in aggregated_importance.items():
            final_importance[feature] = np.mean(scores)

        return final_importance

    def _determine_evidence_strength(self, probability: float, uncertainty: float) -> str:
        """Determine strength of evidence based on probability and uncertainty"""

        if probability > 0.8 and uncertainty < 0.2:
            return 'strong'
        elif probability > 0.6 and uncertainty < 0.4:
            return 'moderate'
        else:
            return 'weak'

    def _calculate_overall_confidence(self, predictions: List[DiagnosticPrediction]) -> float:
        """Calculate overall confidence across all predictions"""

        if not predictions:
            return 0.0

        # Weight confidence by evidence strength
        weighted_confidences = []
        for pred in predictions:
            if pred.evidence_strength == 'strong':
                weight = 1.0
            elif pred.evidence_strength == 'moderate':
                weight = 0.7
            else:
                weight = 0.4

            confidence = pred.probability * (1 - pred.uncertainty)
            weighted_confidences.append(confidence * weight)

        return np.mean(weighted_confidences)

    def _calculate_model_agreement(self, predictions: List[DiagnosticPrediction]) -> float:
        """Calculate how much models agree across predictions"""

        if not predictions:
            return 0.0

        agreement_scores = []
        for pred in predictions:
            if pred.model_consensus:
                consensus_values = list(pred.model_consensus.values())
                if len(consensus_values) > 1:
                    # Calculate coefficient of variation (lower = more agreement)
                    std = np.std(consensus_values)
                    mean = np.mean(consensus_values)
                    cv = std / (mean + 1e-8)
                    agreement = 1.0 - min(1.0, cv)  # Convert to agreement score
                    agreement_scores.append(agreement)

        return np.mean(agreement_scores) if agreement_scores else 1.0

    def _generate_recommendations(self, predictions: List[DiagnosticPrediction],
                                uncertainty_flags: List[str]) -> List[str]:
        """Generate actionable recommendations based on predictions"""

        recommendations = []

        # Check for high-confidence positive predictions
        high_confidence_conditions = [
            pred for pred in predictions
            if pred.probability > 0.7 and pred.evidence_strength == 'strong'
        ]

        if high_confidence_conditions:
            recommendations.append("Strong diagnostic evidence detected. Proceed with recommended treatment protocols.")

        # Check for uncertainty flags
        if uncertainty_flags:
            recommendations.append("High uncertainty detected in some diagnoses. Consider additional diagnostic testing.")

        # Check for conflicting evidence
        conflicting = [pred for pred in predictions if pred.evidence_strength == 'weak' and pred.probability > 0.5]
        if conflicting:
            recommendations.append("Conflicting evidence for some conditions. Manual clinical review recommended.")

        # Default recommendation
        if not recommendations:
            recommendations.append("Standard diagnostic protocols applied. Monitor for changes.")

        return recommendations

    def save_ensemble(self, filepath: str):
        """Save entire ensemble to disk"""

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        ensemble_data = {
            'classifiers': {},
            'ensemble_weights': self.ensemble_weights,
            'timestamp': datetime.now().isoformat()
        }

        # Save individual classifiers
        for condition_name, classifiers in self.classifiers.items():
            ensemble_data['classifiers'][condition_name] = []
            for i, classifier in enumerate(classifiers):
                classifier_file = filepath.replace('.pkl', f'_{condition_name}_{i}.pkl')
                with open(classifier_file, 'wb') as f:
                    pickle.dump(classifier, f)
                ensemble_data['classifiers'][condition_name].append(classifier_file)

        # Save ensemble metadata
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)

        # Save federated learning state
        federated_file = filepath.replace('.pkl', '_federated.json')
        self.federated_coordinator.save_federated_state(federated_file)

        logger.info(f"Ensemble saved to {filepath}")

    def load_ensemble(self, filepath: str):
        """Load ensemble from disk"""

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ensemble file not found: {filepath}")

        # Load ensemble metadata
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.ensemble_weights = ensemble_data.get('ensemble_weights', {})

        # Load individual classifiers
        self.classifiers = {}
        for condition_name, classifier_files in ensemble_data.get('classifiers', {}).items():
            self.classifiers[condition_name] = []
            for classifier_file in classifier_files:
                if os.path.exists(classifier_file):
                    with open(classifier_file, 'rb') as f:
                        classifier = pickle.load(f)
                        self.classifiers[condition_name].append(classifier)

        # Load federated learning state
        federated_file = filepath.replace('.pkl', '_federated.json')
        if os.path.exists(federated_file):
            self.federated_coordinator.load_federated_state(federated_file)

        logger.info(f"Ensemble loaded from {filepath}")