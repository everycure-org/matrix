from pydantic_settings import BaseSettings


class MatrixCliSettings(BaseSettings):
    base_model: str = "gemini-1.5-flash-002"
    power_model: str = "gemini-1.5-pro-002"


settings = MatrixCliSettings()
