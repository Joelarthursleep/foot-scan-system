"""
Continuous ML Model Training Agent
Monitors scan data and automatically retrains models when needed
"""

import os
import sys
import time
import json
import pickle
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.ensemble_predictor import EnsembleConditionPredictor


class ContinuousTrainingAgent:
    """
    Autonomous agent for continuous ML model training

    Features:
    - Monitors new scan data
    - Triggers retraining based on data thresholds
    - Performance-based retraining triggers
    - Scheduled periodic retraining
    - Model versioning and rollback
    """

    def __init__(
        self,
        model_dir: str = 'models/trained',
        training_data_dir: str = 'data/scans',
        min_samples_for_retraining: int = 100,
        performance_threshold: float = 0.75,
        check_interval_minutes: int = 60
    ):
        self.model_dir = Path(model_dir)
        self.training_data_dir = Path(training_data_dir)
        self.min_samples = min_samples_for_retraining
        self.performance_threshold = performance_threshold
        self.check_interval = check_interval_minutes * 60  # Convert to seconds

        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger('ContinuousTrainer')
        self.logger.setLevel(logging.INFO)

        # Training state
        self.is_running = False
        self.training_thread = None
        self.last_training_time = None
        self.training_history = []

        # Load training metadata
        self.metadata_file = self.model_dir / 'training_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load training metadata from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'last_training_date': None,
            'total_training_runs': 0,
            'model_versions': [],
            'performance_history': {},
            'total_samples_trained': 0
        }

    def _save_metadata(self):
        """Save training metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_available_training_data(self) -> tuple:
        """
        Collect training data from scan database

        Returns:
            (training_data, labels) tuple
        """
        training_data = []
        labels = {
            'hallux_valgus': [],
            'pes_planus': [],
            'pes_cavus': [],
            'plantar_fasciitis': [],
            'hammertoe': [],
            'metatarsalgia': []
        }

        # Check for collected scan data
        scan_files = list(self.training_data_dir.glob('scan_*.pkl'))

        if not scan_files:
            self.logger.info("No training data found in database")
            return [], labels

        for scan_file in scan_files:
            try:
                with open(scan_file, 'rb') as f:
                    scan_data = pickle.load(f)

                # Validate scan data has required fields
                if 'structure' in scan_data and 'measurements' in scan_data and 'conditions' in scan_data:
                    training_data.append(scan_data)

                    # Extract labels for each condition
                    detected_conditions = scan_data.get('conditions', [])
                    for condition in labels.keys():
                        # Label as 1 if condition detected, 0 otherwise
                        labels[condition].append(1 if condition in detected_conditions else 0)

            except Exception as e:
                self.logger.warning(f"Error loading scan file {scan_file}: {e}")
                continue

        self.logger.info(f"Loaded {len(training_data)} scans from database")
        return training_data, labels

    def should_trigger_retraining(self) -> tuple:
        """
        Check if retraining should be triggered

        Returns:
            (should_train, reason) tuple
        """
        # Check 1: Minimum time since last training (daily at most)
        if self.metadata['last_training_date']:
            last_training = datetime.fromisoformat(self.metadata['last_training_date'])
            if datetime.now() - last_training < timedelta(hours=24):
                return False, "Too soon since last training"

        # Check 2: Enough new samples
        training_data, labels = self.get_available_training_data()
        current_samples = len(training_data)
        previous_samples = self.metadata['total_samples_trained']

        new_samples = current_samples - previous_samples

        if new_samples >= self.min_samples:
            return True, f"New samples threshold reached: {new_samples} new samples"

        # Check 3: Scheduled weekly retraining
        if self.metadata['last_training_date']:
            last_training = datetime.fromisoformat(self.metadata['last_training_date'])
            if datetime.now() - last_training >= timedelta(days=7):
                return True, "Scheduled weekly retraining"

        # Check 4: First training ever
        if self.metadata['total_training_runs'] == 0:
            return True, "Initial model training"

        return False, "No retraining triggers met"

    def train_models(self, training_data: List, labels: Dict) -> Dict:
        """
        Train all condition models

        Returns:
            Dictionary of performance metrics per condition
        """
        self.logger.info("Starting model training...")

        conditions = list(labels.keys())
        performance_results = {}

        for condition in conditions:
            self.logger.info(f"Training {condition} models...")

            try:
                # Create predictor
                predictor = EnsembleConditionPredictor()

                # Train with cross-validation
                condition_labels = {condition: labels[condition]}
                performance = predictor.train_models(
                    training_data=training_data,
                    labels=condition_labels,
                    cv_folds=5
                )

                # Save model with version
                version = self.metadata['total_training_runs'] + 1
                model_path = self.model_dir / f'{condition}_models_v{version}.pkl'
                predictor.save_models(str(model_path))

                # Also save as current version
                current_path = self.model_dir / f'{condition}_models.pkl'
                predictor.save_models(str(current_path))

                perf = performance[condition]
                performance_results[condition] = {
                    'mean_cv_accuracy': perf.mean_cv_accuracy,
                    'test_accuracy': perf.test_accuracy,
                    'test_precision': perf.test_precision,
                    'test_recall': perf.test_recall,
                    'test_f1': perf.test_f1,
                    'model_path': str(model_path),
                    'version': version
                }

                self.logger.info(f"✅ {condition}: Test Accuracy = {perf.test_accuracy:.3f}, F1 = {perf.test_f1:.3f}")

            except Exception as e:
                self.logger.error(f"❌ Error training {condition}: {e}")
                import traceback
                traceback.print_exc()
                performance_results[condition] = {'error': str(e)}

        return performance_results

    def create_ensemble_package(self):
        """Create combined ensemble package from current models"""
        self.logger.info("Creating ensemble package...")

        ensemble_package = {}

        for filename in os.listdir(self.model_dir):
            if not filename.endswith('_models.pkl') or filename.startswith('ensemble'):
                continue

            if '_models_v' in filename:
                # Skip versioned snapshots; rely on latest *_models.pkl copies
                continue

            model_path = self.model_dir / filename
            if not model_path.is_file():
                continue

            condition = filename.replace('_models.pkl', '')
            predictor = EnsembleConditionPredictor()
            predictor.load_models(str(model_path))
            ensemble_package[condition] = predictor

        # Save ensemble package
        package_path = self.model_dir / 'ensemble_package.pkl'
        with open(package_path, 'wb') as f:
            pickle.dump(ensemble_package, f)

        self.logger.info(f"✅ Ensemble package saved with {len(ensemble_package)} models")

    def run_training_cycle(self):
        """Execute a complete training cycle"""
        self.logger.info("="*60)
        self.logger.info("CONTINUOUS TRAINING CYCLE STARTED")
        self.logger.info("="*60)

        start_time = time.time()

        try:
            # Get training data
            training_data, labels = self.get_available_training_data()

            if len(training_data) < 10:
                self.logger.error("Not enough training data. Need at least 10 samples.")
                raise RuntimeError("Not enough training data. Need at least 10 samples.")

            # Train models
            performance = self.train_models(training_data, labels)

            # Create ensemble package
            self.create_ensemble_package()

            # Update metadata
            self.metadata['last_training_date'] = datetime.now().isoformat()
            self.metadata['total_training_runs'] += 1
            self.metadata['total_samples_trained'] = len(training_data)
            self.metadata['performance_history'][datetime.now().isoformat()] = performance
            self.metadata['model_versions'].append({
                'version': self.metadata['total_training_runs'],
                'date': datetime.now().isoformat(),
                'samples': len(training_data),
                'performance': performance
            })
            self._save_metadata()

            elapsed = time.time() - start_time
            self.logger.info(f"✅ Training cycle completed in {elapsed:.1f} seconds")

            # Log to training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'samples': len(training_data),
                'performance': performance,
                'duration': elapsed
            })

        except Exception as e:
            self.logger.error(f"❌ Training cycle failed: {e}")
            import traceback
            traceback.print_exc()

    def monitoring_loop(self):
        """Main monitoring loop that runs in background thread"""
        self.logger.info("Continuous training agent started")

        while self.is_running:
            try:
                # Check if retraining should be triggered
                should_train, reason = self.should_trigger_retraining()

                if should_train:
                    self.logger.info(f"Retraining triggered: {reason}")
                    self.run_training_cycle()
                else:
                    self.logger.debug(f"No retraining needed: {reason}")

                # Wait for next check
                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def start(self):
        """Start the continuous training agent"""
        if self.is_running:
            self.logger.warning("Agent is already running")
            return

        self.is_running = True
        self.training_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.training_thread.start()

        self.logger.info("✅ Continuous training agent started")

    def stop(self):
        """Stop the continuous training agent"""
        if not self.is_running:
            return

        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=5)

        self.logger.info("Continuous training agent stopped")

    def force_retrain(self):
        """Force an immediate retraining cycle"""
        self.logger.info("Manual retraining triggered")
        self.run_training_cycle()

    def reload_metadata(self):
        """Reload metadata from disk (useful after external changes)"""
        self.metadata = self._load_metadata()

    def get_status(self) -> Dict:
        """Get current agent status (reloads metadata for fresh data)"""
        # Reload metadata to get latest state
        self.reload_metadata()

        return {
            'is_running': self.is_running,
            'last_training': self.metadata.get('last_training_date'),
            'total_runs': self.metadata.get('total_training_runs', 0),
            'total_samples': self.metadata.get('total_samples_trained', 0),
            'check_interval_minutes': self.check_interval / 60,
            'training_history_count': len(self.training_history)
        }

    def get_training_history(self) -> List[Dict]:
        """Get recent training history"""
        return self.training_history[-10:]  # Last 10 runs


# Global agent instance
_agent_instance: Optional[ContinuousTrainingAgent] = None


def get_training_agent() -> ContinuousTrainingAgent:
    """Get or create the global training agent instance"""
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = ContinuousTrainingAgent(
            model_dir='models/trained',
            training_data_dir='data/scans',
            min_samples_for_retraining=100,
            performance_threshold=0.75,
            check_interval_minutes=60  # Check every hour
        )

    return _agent_instance


if __name__ == '__main__':
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and start agent
    agent = get_training_agent()

    print("="*60)
    print("CONTINUOUS ML TRAINING AGENT")
    print("="*60)
    print(f"\nAgent started. Monitoring every {agent.check_interval/60:.0f} minutes.")
    print("Press Ctrl+C to stop.\n")

    agent.start()

    try:
        # Keep running
        while True:
            time.sleep(10)
            status = agent.get_status()
            print(f"Status: Running={status['is_running']}, "
                  f"Total Runs={status['total_runs']}, "
                  f"Samples={status['total_samples']}")
    except KeyboardInterrupt:
        print("\nStopping agent...")
        agent.stop()
        print("Agent stopped.")
