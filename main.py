from src.ML.pipeline.S1_data_ingestion import DataIngestionPipeline
from src.ML.pipeline.S2_data_validation import DataValidationPipeline
from src.ML.pipeline.S3_data_processing import DataProcessingPipeline
from src.ML import logger



STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"******* stage {STAGE_NAME} started *******")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f"******* stage {STAGE_NAME} completed  *******\n\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation"

try:
    logger.info(f"******* stage {STAGE_NAME} started *******")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f"******* stage {STAGE_NAME} completed *******\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Processing"

try:
    logger.info(f"******* stage {STAGE_NAME} started *******")
    data_processing = DataProcessingPipeline()
    data_processing.main()
    logger.info(f"******* stage {STAGE_NAME} completed *******\n\n")
except Exception as e:
    logger.exception(e)
    raise e



