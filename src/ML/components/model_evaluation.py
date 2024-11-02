import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from ML.entity.config_entity import ModelEvaluationConfig
from ML.utils.common import save_json
from src.ML import logger
import dagshub
from dotenv import load_dotenv  



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        dagshub.init(repo_owner='omaar25',
                     repo_name='Predictive-Maintenance-Web-Application',
                     mlflow=True
                    )

    def load_test_data(self):
        """Load the test data and split into features and target."""
        self.test_data = pd.read_csv(self.config.test_data_path)
        self.X_test = self.test_data.drop(columns=[self.config.target_column])
        self.y_test = self.test_data[self.config.target_column]
        logger.info("Test data loaded successfully")

    def evaluate_model(self, model):
        """Evaluate a single model and return its metrics as a dictionary."""
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro')
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

    def get_model_params(self, model_name):
        """Retrieve model-specific parameters from the config."""
        return self.config.parms.get(model_name, {})

    def log_to_mlflow(self, model_name, metrics):
        """Log model metrics and parameters to MLflow."""
        with mlflow.start_run(run_name=model_name):
            model_params = self.get_model_params(model_name)
            for param, value in model_params.items():
                mlflow.log_param(param, value)
                
            for metric_name, metric_value in metrics.items():
                try:
                    mlflow.log_metric(metric_name, float(metric_value))
                except ValueError:
                    logger.error(f"Metric {metric_name} with value {metric_value} is not numeric.")
                    
            logger.info(f"Logged {model_name} metrics to MLflow")
    
    def evaluate_best_model(self):
        """Evaluate all models, log each to MLflow, and save the best model locally."""
        model_files = [os.path.join(self.config.model_path, model_file) for model_file in os.listdir(self.config.model_path)]
        metrics_list = []
        best_model = None
        best_model_name = ""
        best_f1_score = 0

        for model_path in model_files:
            model_name = os.path.basename(model_path).split('.')[0]
            model = joblib.load(model_path)
            logger.info(f"Evaluating model: {model_name}")
            metrics = self.evaluate_model(model)
            metrics['Model'] = model_name
            metrics_list.append(metrics)
            self.log_to_mlflow(model_name, metrics)

            if metrics["F1 Score"] > best_f1_score:
                best_f1_score = metrics["F1 Score"]
                best_model = model
                best_model_name = model_name
                model_artifact_path = os.path.join(self.config.root_dir, f'{model_name}.joblib')
        joblib.dump(best_model, model_artifact_path)
        logger.info(f"Best model is {best_model_name}")


        save_json(Path(self.config.metric_file_name), {"metrics": metrics_list})
