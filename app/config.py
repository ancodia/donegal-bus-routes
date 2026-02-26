# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: str = "./data"
    graphml_path: str = "./graph/graphml"
    rp_graphml_path: str = "./route_planning/graphml"

    class Config:
        env_file = ".env"


settings = Settings()
