from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageMatcher(ABC):
    @abstractmethod
    def build(self, reference_images: list[Path], reference_places: list[int]) -> None:
        """Extract/index features for the reference images."""

    @abstractmethod
    def match(self, query_image: Path) -> int:
        """Return the predicted place for a query image."""
