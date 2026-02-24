"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "SpO2 Modelling"
    debug: bool = False

    # Database
    db_path: Path = Path(__file__).parent.parent.parent / "data" / "spo2.db"

    # CORS (for local development)
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"

    model_config = {"env_prefix": "SPO2_"}


settings = Settings()
