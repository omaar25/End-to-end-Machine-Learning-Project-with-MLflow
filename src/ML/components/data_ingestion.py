import os
import urllib.request as request
import zipfile
from ML import logger
from ML.entity.config_entity import DataIngestionConfig
from ML.utils.common import get_size
from pathlib import Path



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url= self.config.source_URL,
                filename= self.config.local_data_file
            )
            logger.info(f"{filename} download with info : \n{headers}")
        else:
            logger.info(f"File already exist , size : {get_size(Path(self.config.local_data_file))}")

    
    def extract_zip(self):
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip:
            zip.extractall(unzip_path) 