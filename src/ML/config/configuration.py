from src.ML.constants import *
from ML.utils.common import read_yaml, create_directories
from ML.entity.config_entity import (DataIngestionConfig,
                                     DataValidationConfig,
                                     DataProcessingConfig,
                                     ModelTrainConfig,
                                     ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(self,
                config_filepath=CONFIG_FILE_PATH,
                params_filepath=PARAMS_FILE_PATH,
                schema_filepath=SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config
    

    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            unzip_dir = config.unzip_dir,
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            all_schema = schema
        )
        return data_validation_config
    


    def get_data_processing_config(self)-> DataProcessingConfig:
        config=self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            root_dir =config.root_dir,
            data_path=config.data_path
        )
        
        return data_processing_config 
    
    def get_model_train_config(self) -> ModelTrainConfig:

        config = self.config.model_train
        parms = self.params
        target_column = list(self.schema.TARGET_COLUMN.keys())[0]

        create_directories([config.root_dir])

        data_train_config = ModelTrainConfig(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            parms = parms,
            target_column = target_column
        )
        
        return data_train_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        parms = {model_name: self.params[model_name] for model_name in self.params.keys()}
        target_column = list(self.schema.TARGET_COLUMN.keys())[0]


        create_directories([config.root_dir])

        model_evaluation_config= ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            parms=parms,
            metric_file_name=config.metric_file_name,
            target_column = target_column,
            mlflow_uri="https://dagshub.com/omaar25/End-to-end-Machine-Learning-Project-with-MLflow.mlflow",
        )

        return model_evaluation_config