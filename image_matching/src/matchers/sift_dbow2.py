from pathlib import Path

import numpy as np
import cv2 as cv
import torch

from .dbow2_base import DBoW2MatcherBase
from bindings import dbow2_cpp

# import sift algorithm from hloc
from hloc.extractors.dog import DoG


class SiftDBoW2Matcher(DBoW2MatcherBase):
    feature_extractor = "sift"
    matcher_norm = cv.NORM_L2

    def __init__(self, nfeatures: int = 1000, descriptor: str = "rootsift", k: int = 9, L: int = 3, vocabulary_path: Path | None = None):
        """Initialize the SIFT DBoW2 matcher."""
        """nfeatures: total number of SIFT features to extract per image"""
        """descriptor: type of SIFT descriptor to use"""
        """k: number of branches at each node in the DBoW2 vocabulary tree"""
        """L: depth of the DBoW2 vocabulary tree"""
        """vocabulary_path: pre-trained vocabulary path, if None, create a new one using reference images"""
        self.vocabulary_path = Path(vocabulary_path) if vocabulary_path is not None else None
        self.reference_images = []
        self.reference_places = []
        
        # init sift feature extractor
        self.dog = DoG({"max_keypoints": nfeatures, "descriptor": descriptor})
        self.dog.eval()

        # setup DBoW2 database
        self.db = dbow2_cpp.SiftDatabase(k=k, L=L)
        self.is_built = False


    def extract_features_descriptors(self, image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        img = cv.imread(image.as_posix(), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")

        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor[None, None]  # (1, 1, H, W)

        with torch.no_grad():
            output = self.dog({"image": img_tensor})

        kp_xy = output["keypoints"][0].numpy()            # (N, 2)
        scales = output["scales"][0].numpy()              # (N,)
        oris = output["oris"][0].numpy()                  # (N,) radians
        descriptors = output["descriptors"][0].T.numpy()  # (N, 128)
        keypoints = [
            cv.KeyPoint(float(x), float(y), float(s) * 12.0, float(np.degrees(a)))
            for (x, y), s, a in zip(kp_xy, scales, oris)
        ]

        if len(descriptors) == 0:
            return keypoints, None
        return keypoints, descriptors.astype(np.float32)
