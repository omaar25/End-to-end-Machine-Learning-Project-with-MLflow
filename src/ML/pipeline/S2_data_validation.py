from src.ML.config.configuration import ConfigurationManager
from src.ML.components.data_validation import DataValidation
from src.ML import logger

STAGE_NAME = "Data Validation"

class DataValidationPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_columns()



if __name__ == '__main__' :
    try:
        logger.info(f"******* stage {STAGE_NAME} started *******")
        object= DataValidationPipeline()
        object.main()
        logger.info(f"******* stage {STAGE_NAME} completed *******")
    except Exception as e:
        logger.exception(e)
        raise e