from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import cv2 as cv


class ImageMatcher(ABC):
    @abstractmethod
    def __init__(self, config) -> None:
        """Initialize the matcher."""
        pass

    @abstractmethod
    def match(self, query_image: Path, potential_places: list[int]) -> int:
        """Return the predicted place for a query image."""
        pass

    @abstractmethod
    def query(self, query_image: Path, top_k: int = 5) -> list[tuple[int, int, float]]:
        """Return ranked DBoW2 matches for a query image."""
        pass

    @abstractmethod
    def set_reference_database(self, reference_images: list[Path], reference_places: list[int]) -> list:
        """Extract/index features for the reference images."""
        pass

class LocalFeatureMatcher(ImageMatcher):
    @abstractmethod
    def extract_features_descriptors(self, image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        """Extract keypoints and descriptors from an image."""
        pass

class GlobalDescriptorMatcher(ImageMatcher):
    @abstractmethod
    def extract_features_descriptors(self, image: Path) -> tuple[np.ndarray]:
        """Extract a global descriptor from an image."""
        pass