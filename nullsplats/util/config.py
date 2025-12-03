"""Application configuration for NullSplats.

The configuration is intentionally explicit and does not read any environment
variables. Callers construct an :class:`AppConfig` with concrete values so
behavior is predictable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Simple container for application-wide settings."""

    cache_root: Path = Path("cache")
    window_title: str = "NullSplats"

    def with_cache_root(self, cache_root: str | Path) -> "AppConfig":
        """Return a copy with an updated cache root."""
        return AppConfig(cache_root=Path(cache_root), window_title=self.window_title)


__all__ = ["AppConfig"]
