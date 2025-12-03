"""SceneId helper for validating and reusing scene identifiers.

A Scene ID is a short string used to group all cached artifacts for a dataset.
Valid characters are restricted to ASCII alphanumerics plus `_` and `-` to
ensure directory names are portable across platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_SCENE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass(frozen=True)
class SceneId:
    """Lightweight wrapper around a validated scene identifier."""

    value: str

    def __post_init__(self) -> None:
        cleaned = self.value.strip()
        if not cleaned:
            raise ValueError("Scene ID cannot be empty.")
        if not _SCENE_ID_PATTERN.match(cleaned):
            raise ValueError(
                "Scene ID must contain only letters, numbers, underscores, or hyphens."
            )
        object.__setattr__(self, "value", cleaned)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"SceneId({self.value!r})"
