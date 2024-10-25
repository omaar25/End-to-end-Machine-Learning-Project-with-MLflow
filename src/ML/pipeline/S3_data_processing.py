from pathlib import Path
from ML import logger
from ML.components.data_processing import DataProcessing
from ML.config.configuration import ConfigurationManager

STAGE_NAME = "Data Processing"


class DataProcessingPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"),"r") as f:
                status = f.read().split(":")[-1]
            
            if status == "True":
                config = ConfigurationManager()
                data_processing_config = config.get_data_processing_config()
                data_processing= DataProcessing(config=data_processing_config)
                data_processing.load_data()
                data_processing.rename_and_drop_columns()
                data_processing.convert_temperature()
                data_processing.encode_features()
                data_processing.scale_features()
                data_processing.oversample_data()
                data_processing.train_test_split()
            
            else:
                raise Exception("schema not valid")
        
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f"******* stage {STAGE_NAME} started ********")
        obj = DataProcessingPipeline()
        obj.main()
        logger.info(f"****** stage {STAGE_NAME} completed *******")
    except Exception as e:
        logger.exception(e)
        raise e