from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    google_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    gemini_temp: float = 0.7
    mysql_url: str
    jwt_secret: str
    jwt_exp_minutes: int
    jwt_algorithm: str = "HS256"

    model_config = SettingsConfigDict(
        env_file=("backend/.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()