import joblib
from pathlib import Path
from src.ML.config.configuration import ConfigurationManager
from pathlib import Path
import joblib

class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        best_model_path = Path(config.get_model_evaluation_config().root_dir)
        
        try:
            self.best_model = joblib.load(next(best_model_path.glob("*.joblib")))
        except StopIteration:
            raise FileNotFoundError(f"No .joblib model file found in {best_model_path}")
    
    def predict(self, data):
        prediction = self.best_model.predict(data)
        return prediction
