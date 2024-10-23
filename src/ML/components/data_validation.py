import os
import pandas as pd 
from src.ML import logger
from src.ML.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config:DataValidationConfig):
        self.config = config
    
    def validate_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_dir)
            columns = list(data.columns)

            schema = self.config.all_schema.keys()


            for column in columns:
                if column not in schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status:{validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status:{validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
