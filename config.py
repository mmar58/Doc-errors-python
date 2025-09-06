from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import ClassVar
import os

class Settings(BaseSettings):
    # Service
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)

    # RouteLLM / Abacus
    ABACUS_API_KEY: str | None = Field(default="s2_8fe4b2ba5e984913af78aea198072d70")
    ABACUS_BASE_URL: str = Field(default="https://routellm.abacus.ai/v1")
    ROUTELLM_MODEL_GPT5: str = Field(default="gpt-5")
    ROUTELLM_MODEL_CLAUDE: str = Field(default="claude-sonnet-4-20250514")

    # Data / CSV
    SEED_CSV_PATH: str = Field(default="Tortured_Phrases_Lexicon_2.csv")
    DEFAULT_CSV_SEVERITY: str = Field(default="High")  # storage-time default; matching-time default is High
    DEFAULT_WEIGHT: int = Field(default=0)

    # Prompt caps
    MAX_PROMPT_MATCHES: int = Field(default=300)
    MAX_PROMPT_CHARS: int = Field(default=24000)

    # Reporting / Output
    REPORTS_DIR: str = Field(default="reports")

    # Timeouts
    LLM_TIMEOUT_SECS: float = Field(default=18.0)

    # Optional enhancements (off by default)
    ENABLE_AUTOMATON_CACHE: bool = Field(default=False)
    ENABLE_PAGE_PARALLEL: bool = Field(default=False)

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # LLM discovery settings (ClassVar so pydantic ignores them as model fields)
    DISCOVERY_ENABLED: bool = Field(default=True)  # always on per your request
    DISCOVERY_SCAN_FIRST_N_PAGES_IF_NO_MATCHES: ClassVar[int] = int(
        os.getenv("DISCOVERY_SCAN_FIRST_N_PAGES_IF_NO_MATCHES", "100")
    )
    DISCOVERY_MAX_PAGES_PER_DOC: ClassVar[int] = int(
        os.getenv("DISCOVERY_MAX_PAGES_PER_DOC", "100")
    )
    DISCOVERY_MAX_FINDINGS_PER_PAGE: ClassVar[int] = int(
        os.getenv("DISCOVERY_MAX_FINDINGS_PER_PAGE", "20")
    )
    DISCOVERY_MODEL_NAME: ClassVar[str] = os.getenv("DISCOVERY_MODEL_NAME", "gpt-5")

settings = Settings()