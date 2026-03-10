from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Project Management API"
    debug: bool = False

    database_url: str = "sqlite:///./app.db"

    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    upload_dir: str = "storage/uploads"
    max_upload_size_mb: int = 10
    allowed_content_types: list[str] = [
        "image/jpeg",
        "image/png",
        "application/pdf",
        "text/plain",
    ]

    class Config:
        env_file = ".env"


settings = Settings()
