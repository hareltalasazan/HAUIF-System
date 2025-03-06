cat > hauif_system/core/config.py << 'EOL'
import asyncio
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError, validator
from watchfiles import awatch
import structlog
import json

structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger("HAUIFSystem")

class HAUIFSettings(BaseSettings):
    infura_url: str = Field(..., examples=["https://mainnet.infura.io/v3/..."])
    contract_address: str = Field(..., regex=r"^0x[a-fA-F0-9]{40}$")
    private_key: str = Field(..., min_length=64, max_length=64)
    difficulty: int = Field(2, ge=1)
    ledger_path: str = Field("enhanced_ledger.json")
    model_path: str = Field("hauif_model.pkl")
    corruption_threshold: float = Field(0.7, ge=0.0, le=1.0)
    evidence_weights: dict[str, float] = Field(
        default_factory=lambda: {"financial": 0.7, "property": 0.8, "court": 0.75, "personal": 0.6,
                                "foia": 0.85, "survey": 0.8, "registry": 0.7, "tax": 0.65}
    )

    @validator("evidence_weights")
    def validate_weights(cls, v):
        if not all(0 <= w <= 1 for w in v.values()):
            raise ValueError("Evidence weights must be between 0 and 1")
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="forbid")

async def watch_config():
    async for changes in awatch(".env"):
        try:
            global settings
            settings = HAUIFSettings()
            logger.info("Configuration reloaded", changes=changes)
        except ValidationError as e:
            logger.error("Failed to reload configuration", errors=e.errors())

settings = HAUIFSettings()
EOL
