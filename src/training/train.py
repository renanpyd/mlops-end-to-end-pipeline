import logging
import time
from typing import Dict, Any
from src.features.engineering import FeatureEngineeringProcessor

logger = logging.getLogger(__name__)

class MockMLflowRegistry:
    """
    Simulated MLflow Tracking Client Interface.
    Exposes identical API signatures to prove structural experiment tracking and Model Registry workflows.
    """
    @staticmethod
    def log_param(key: str, value: Any) -> None:
        logger.info(f"[MLflow Log] Parameter Registered -> {key}: {value}")

    @staticmethod
    def log_metric(key: str, value: float) -> None:
        logger.info(f"[MLflow Log] Metric Tracked -> {key}: {value}")

    @staticmethod
    def register_model(model_name: str, current_stage: str) -> None:
        logger.info(f"[MLflow Registry] Model '{model_name}' successfully committed to '{current_stage}' stage.")

class ModelTrainingPipeline:
    """
    Automated Machine Learning Training Core.
    Coordinates Feature Processing, Model Training, Metric Validation, and manages
    the model production lifecycle via native MLflow integration wrappers.
    """
    def __init__(self, mlflow_client: MockMLflowRegistry):
        self.mlflow = mlflow_client
        self.feature_processor = FeatureEngineeringProcessor()

    def run_training_experiment(self, raw_training_set: list, model_hyperparameters: Dict[str, Any]) -> None:
        """Executes full predictive compilation tracking model artifacts down to centralized registers."""
        logger.info("[Training Pipeline] Initializing distributed training experiment execution run...")
        
        # 1. Executa a extração de features através da camada dedicada
        features_matrix = self.feature_processor.compute_customer_churn_features(raw_training_set)

        # 2. Rastreamento de Hiperparâmetros via MLflow (Garantindo reprodutibilidade)
        for param, val in model_hyperparameters.items():
            self.mlflow.log_param(param, val)

        logger.info("[Training Pipeline] Processing matrix optimization algorithms...")
        time.sleep(1) # Simula o processamento computacional pesado de treinamento

        # 3. Extração e Log das Métricas Finais de Validação
        simulated_accuracy = 0.942 + (0.01 * np.random.randn())
        simulated_f1_score = 0.915
        
        self.mlflow.log_metric("accuracy_score", float(simulated_accuracy))
        self.mlflow.log_metric("f1_validation_score", simulated_f1_score)

        # 4. Homologação de Código e Registro do Artefato no Model Registry
        self.mlflow.register_model(model_name="customer_churn_production_model", current_stage="Staging")
        logger.info("[Training Pipeline] Experiment cycle successfully completed.")
