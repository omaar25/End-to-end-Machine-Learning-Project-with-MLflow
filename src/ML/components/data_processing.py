import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from src.ML.entity.config_entity import DataProcessingConfig
from src.ML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle


class DataProcessing:
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.df = None
        

    def load_data(self):
        """Load data from CSV file."""
        self.df = pd.read_csv(self.config.data_path)
        logger.info("Data loaded successfully")
    
    def drop_columns(self):
        """drop specified columns."""
        self.df.drop(['UDI', 'Product ID', 'Target'], axis=1, inplace=True)
        logger.info("dropped columns")

    def rename_columns(self):
        """Rename specified columns."""
        rename_dict = {
            'Air temperature [K]': 'Air_temperature_C',
            'Process temperature [K]': 'Process_temperature_C',
            'Rotational speed [rpm]': 'Rotational_Speed_RPM',
            'Torque [Nm]': 'Torque_Nm',
            'Tool wear [min]': 'Tool_Wear_min',
        }
        self.df.rename(columns=rename_dict, inplace=True)
        logger.info("Renamed columns")

    def convert_temperature(self):
        """Convert temperature from Kelvin to Celsius."""
        self.df['Air_temperature_C'] = self.df['Air_temperature_C'] - 273.15
        self.df['Process_temperature_C'] = self.df['Process_temperature_C'] - 273.15
        logger.info("Converted temperatures to Celsius")

    def get_type_mapping(self):
        """Return the mapping of failure types to numerical values."""
        return {
            'L': 0,
            'M': 1,
            'H': 2,
        }

    def get_failure_type_mapping(self):
        """Return the mapping of failure types to numerical values."""
        return {
            'No Failure': 0,
            'Heat Dissipation Failure': 1,
            'Power Failure': 2,
            'Overstrain Failure': 3,
            'Tool Wear Failure': 4,
            'Random Failures': 5
        }
    
    def encode_features(self):
        """Encode categorical features with OrdinalEncoder and LabelEncoder."""

        #The if jsut to make sure in the app to encode Type as we will not have Failure Type because it is the targer
        if 'Type' in self.df.columns:
            self.df['Type'] = self.df['Type'].map(self.get_type_mapping())
        if 'Failure Type' in self.df.columns:
            self.df['Failure Type'] = self.df['Failure Type'].map(self.get_failure_type_mapping())

    
    def decode_failure_type(self, encoded_value):
        """Decode an encoded failure type value using the failure type mapping."""
        inverse_mapping = {v: k for k, v in self.get_failure_type_mapping().items()}
        return inverse_mapping.get(encoded_value, "Unknown")


    def scale_features(self):
        """Scale specified numerical features with MinMaxScaler."""
        col_to_scale = ['Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_min', 'Air_temperature_C', 'Process_temperature_C']
        scaler = MinMaxScaler()
        self.df[col_to_scale] = scaler.fit_transform(self.df[col_to_scale])
        scaler_path = os.path.join(self.config.root_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("Scaled numerical features")

    def balance_data(self):
        """Balance the dataset by undersampling 'No Failure' and oversampling other classes."""
        # Separate features and target variable
        X = self.df.drop('Failure Type', axis=1)
        y = self.df['Failure Type']

        # Undersample the "No Failure" class
        rus = RandomUnderSampler(sampling_strategy={'No Failure': 2500})
        X_under, y_under = rus.fit_resample(X, y)

        # Now, oversample the other classes to specified counts
        ros = RandomOverSampler(sampling_strategy={
            'Heat Dissipation Failure': 210,
            'Power Failure': 190,
            'Overstrain Failure': 140,
            'Tool Wear Failure': 80,
            'Random Failures': 40
        })

        X_balanced, y_balanced = ros.fit_resample(X_under, y_under)

        # Combine balanced features and target variable back into a DataFrame
        self.df = pd.concat([X_balanced, y_balanced], axis=1)
        logger.info("Balanced the dataset")

    def train_test_split(self):
        """Split the data into training and test sets and save to CSV."""
        y = self.df['Failure Type']
        train, test = train_test_split(self.df, test_size=0.2, random_state=42, stratify=y)
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        logger.info("Data split into training and test sets")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
