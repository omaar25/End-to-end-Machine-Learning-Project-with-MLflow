from ML import logger
from ML.components.data_ingestion import DataIngestion
from ML.config.configuration import ConfigurationManager


STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:

    def __init__(self):
        pass

    def main(self):
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip()



if __name__ =='__main__':

    try:
        logger.info(f"******* stage {STAGE_NAME} started *******")
        object = DataIngestionPipeline()
        object.main()
        logger.info(f"******* stage {STAGE_NAME} completed *******")
    except Exception as e:
        logger.exception(e)
        raise e
