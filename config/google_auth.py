from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class _GoogleOAuthSettings(BaseSettings):
    CLIENT_ID: str | None = None
    CLIENT_SECRET: str | None = None

    PROJECT_ID: str = "uiet-platform-459805-f5"
    AUTH_URI: str = "https://accounts.google.com/o/oauth2/auth"
    TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    REDIRECT_URIS: list[str] = [
        "urn:ietf:wg:oauth:2.0:oob",
        "http://localhost",
    ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


_google = _GoogleOAuthSettings()

SCOPES: list[str] = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]

TOKEN_PATH: str = "token.json"

CLIENT_CONFIG: dict = {
    "installed": {
        "project_id": _google.PROJECT_ID,
        "client_id": _google.CLIENT_ID,
        "client_secret": _google.CLIENT_SECRET,
        "auth_uri": _google.AUTH_URI,
        "token_uri": _google.TOKEN_URI,
        "redirect_uris": _google.REDIRECT_URIS,
    }
}
