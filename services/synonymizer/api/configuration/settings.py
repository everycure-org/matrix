from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    NEO4J_HOST: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str
