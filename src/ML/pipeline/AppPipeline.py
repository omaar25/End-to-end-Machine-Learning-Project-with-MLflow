import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ML.config.configuration import ConfigurationManager
from src.ML.components.data_processing import DataProcessing

class AppPipelinePrediction:
    def __init__(self):
        self.config = ConfigurationManager()
        self.scaler = self.load_scaler()
        self.best_model = self.load_model()

    def load_scaler(self):
        scaler_path = os.path.join(self.config.get_data_processing_config().root_dir, "scaler.pkl")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = joblib.load(f)
            return scaler
        except FileNotFoundError:
            raise FileNotFoundError(f"No scaler.pkl file found at {self.scaler_path}")

    def load_model(self):
        model_path = Path(self.config.get_model_evaluation_config().root_dir)
        try:
            model_file = next(model_path.glob("*.joblib"))
            model = joblib.load(model_file)
            return model
        except StopIteration:
            raise FileNotFoundError(f"No .joblib model file found in {self.model_path}")

    def preprocess_input(self, form_data):
        Type = form_data['Type']
        Air_temperature_K = float(form_data['Air_temperature_K'])
        Process_temperature_K = float(form_data['Process_temperature_K'])
        Rotational_Speed_RPM = float(form_data['Rotational_Speed_RPM'])
        Torque_Nm = float(form_data['Torque_Nm'])
        Tool_Wear_min = float(form_data['Tool_Wear_min'])
        data = np.array([Type, Air_temperature_K, Process_temperature_K, 
                         Rotational_Speed_RPM, Torque_Nm, Tool_Wear_min]).reshape(1, -1)
        columns = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                   'Torque [Nm]', 'Tool wear [min]']
        df = pd.DataFrame(data, columns=columns)
        df['Air temperature [K]'] = df['Air temperature [K]'].astype(float)
        df['Process temperature [K]'] = df['Air temperature [K]'].astype(float)
        df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].astype(float)
        df['Torque [Nm]'] = df['Torque [Nm]'].astype(float)
        df['Tool wear [min]'] = df['Tool wear [min]'].astype(float)
        data_processor = DataProcessing(self.config.get_data_processing_config())
        data_processor.df = df
        data_processor.rename_columns()
        data_processor.convert_temperature()
        col_to_scale = ['Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_min', 'Air_temperature_C', 'Process_temperature_C']
        data_processor.df[col_to_scale] = self.scaler.transform(data_processor.df[col_to_scale])
        data_processor.encode_features()
        return data_processor.df

    def predict(self, form_data):
        data_processor = DataProcessing(self.config.get_data_processing_config())
        processed_df = self.preprocess_input(form_data)
        prediction = self.best_model.predict(processed_df.values)
        return data_processor.decode_failure_type(prediction[0])
