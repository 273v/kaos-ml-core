"""Module settings for kaos-ml-core.

Standard KAOS ``ModuleSettings`` pattern: env_prefix ``KAOS_ML_CORE_``,
``mode="before"`` legacy fallback, ``extra="ignore"``, ``KAOS_PROFILE``
global fallback for the profile field.
"""

from __future__ import annotations

import os
from typing import Any

from kaos_core.config.module_settings import ModuleSettings
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict


class KaosMLCoreSettings(ModuleSettings):
    """Typed settings for kaos-ml-core.

    Resolution order (highest priority first):
        1. ``from_context(**overrides)``
        2. ``KaosContext._config`` (per-request MCP overrides)
        3. ``KAOS_ML_CORE_*`` environment variables
        4. Legacy ``KAOS_PROFILE`` for the profile field
        5. ``.env`` file
        6. Field defaults below
    """

    default_embed_model: str = "BAAI/bge-small-en-v1.5"
    default_threshold: float = 0.5
    recall_target: float = 0.80
    recall_confidence: float = 0.95
    profile: str = "default"

    model_config = SettingsConfigDict(
        env_prefix="KAOS_ML_CORE_",
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _profile_legacy_fallback(cls, values: Any) -> Any:
        """Fall back to ``KAOS_PROFILE`` when ``KAOS_ML_CORE_PROFILE`` unset."""
        if isinstance(values, dict) and not values.get("profile"):
            legacy = os.environ.get("KAOS_PROFILE")
            if legacy:
                values["profile"] = legacy
        return values


__all__ = ["KaosMLCoreSettings"]
