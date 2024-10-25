import os

from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from src.ML.entity.config_entity import DataProcessingConfig
from src.ML import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


class DataProcessing:
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.df = None  
    
    def load_data(self):
        """Load data from CSV file."""
        self.df = pd.read_csv(self.config.data_path)
        logger.info("Data loaded successfully")

    def rename_and_drop_columns(self):
        """Rename and drop specified columns."""
        self.df.drop(['UDI', 'Product ID', 'Target'], axis=1, inplace=True)
        logger.info("Renamed and dropped columns")

    def convert_temperature(self):
        """Convert temperature from Kelvin to Celsius and drop original columns."""
        self.df['Air temperature [c]'] = self.df['Air temperature [K]'] - 273.15
        self.df['Process temperature [c]'] = self.df['Process temperature [K]'] - 273.15
        self.df.drop(['Air temperature [K]', 'Process temperature [K]'], axis=1, inplace=True)
        logger.info("Converted temperatures to Celsius and dropped original columns")

    def encode_features(self):
        """Encode categorical features with OrdinalEncoder and LabelEncoder."""
        ordinal_encoder = OrdinalEncoder(categories=[['L', 'M', 'H']])
        self.df['Type'] = ordinal_encoder.fit_transform(self.df[['Type']])
        self.df['Type'] = pd.to_numeric(self.df['Type'], errors='coerce')
        label_encoder = LabelEncoder()
        self.df['Failure Type'] = label_encoder.fit_transform(self.df['Failure Type'])
        self.df['Failure Type'] = pd.to_numeric(self.df['Failure Type'], errors='coerce')


    def scale_features(self):
        """Scale specified numerical features with MinMaxScaler."""
        col_to_scale = ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Air temperature [c]', 'Process temperature [c]']
        scaler = MinMaxScaler()
        self.df[col_to_scale] = scaler.fit_transform(self.df[col_to_scale])
        logger.info("Scaled numerical features")

    def oversample_data(self):
        """Apply RandomOverSampler to balance classes in the target variable."""
        X = self.df.drop('Failure Type', axis=1)
        y = self.df['Failure Type']
        ros = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = ros.fit_resample(X, y)
        self.df = pd.concat([X_resampled, y_resampled], axis=1)
        logger.info("Applied random oversampling to balance classes")

    def train_test_split(self):
        train, test = train_test_split(self.df, test_size=0.2, random_state=42)
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        logger.info("Data split into training and test sets")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")