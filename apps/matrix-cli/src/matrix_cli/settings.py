import warnings

from pydantic_settings import BaseSettings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class MatrixCliSettings(BaseSettings):
    base_model: str = "gemini-1.5-flash-002"
    power_model: str = "gemini-1.5-pro-002"


settings = MatrixCliSettings()
