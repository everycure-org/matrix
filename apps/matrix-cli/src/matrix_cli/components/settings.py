import warnings
from typing import List

from pydantic_settings import BaseSettings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class MatrixCliSettings(BaseSettings):
    base_model: str = "gemini-1.5-flash-002"
    power_model: str = "gemini-1.5-pro-002"
    workers: int = 8
    gcs_base_uri: str = "gs://mtrx-us-central1-hub-dev-storage"
    cli_base_path: str = "apps/matrix-cli"
    inclusion_patterns: list[str] = [
        "*.md",
        "*.py",
        "*.yaml",
        "*.tf",
        "*.Makefile",
        "*.Dockerfile",
        "*.sh",
        "*.toml",
        "*.yml",
        "*.txt",
        "*.hcl",
        "*.git",
        ":!**/matrix/packages/*",
    ]
    raw_paths: List[str] = [
        "data/01_RAW",
        "kedro/data/01_raw",
    ]


settings = MatrixCliSettings()
