from ML.components.model_train import ModelTrain
from ML.config.configuration import ConfigurationManager
from src.ML import logger

STAGE_NAME = "Model Training"

class ModelTrainPipeline:
    def __init__(self):
        pass

    def main(self):
        config= ConfigurationManager()
        model_train_config = config.get_model_train_config()
        model_train_config = ModelTrain(config=model_train_config)
        model_train_config.train()


if __name__== '__main__':
    try:
        logger.info(f"******** stage {STAGE_NAME} started **********")
        obj=ModelTrainPipeline()
        obj.main()
        logger.info(f"******* stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e