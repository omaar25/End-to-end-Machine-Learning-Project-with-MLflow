from src.ML.pipeline.S1_data_ingestion import DataIngestionPipeline
from src.ML import logger



STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"******* stage {STAGE_NAME} started *******")
    object = DataIngestionPipeline()
    object.main()
    logger.info(f"******* stage {STAGE_NAME} completed *******")
except Exception as e:
    logger.exception(e)
    raise e


