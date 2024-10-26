from ML.components.model_evaluation import ModelEvaluation
from ML.config.configuration import ConfigurationManager
from src.ML import logger

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.load_test_data()
        model_evaluation.evaluate_best_model() 




if __name__== '__main__':
    try:
        logger.info(f"******** stage {STAGE_NAME} started **********")
        obj=ModelEvaluationPipeline()
        obj.main()
        logger.info(f"******* stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e