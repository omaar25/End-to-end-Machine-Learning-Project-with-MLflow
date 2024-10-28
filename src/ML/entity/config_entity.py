from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_URL : str
    local_data_file : Path
    unzip_dir : Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir : Path
    STATUS_FILE : str
    unzip_dir : Path
    all_schema : dict


@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path 
    data_path: Path


@dataclass(frozen=True)
class ModelTrainConfig:
    root_dir: Path 
    train_data_path: Path
    test_data_path : Path
    model_name : str
    parms : Path
    target_column : str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    parms: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str