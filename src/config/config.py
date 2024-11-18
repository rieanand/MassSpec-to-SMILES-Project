'''Configures the settings to make running different models more efficient.'''

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    data_file_name: FilePath
    model_path: DirectoryPath
    model_name: str

settings = Settings()

DATA_FILE_NAME = filtered_smiles.csv
MODEL_PATH = models
MODEL_NAME = name