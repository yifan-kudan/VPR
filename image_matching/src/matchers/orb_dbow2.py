from pathlib import Path

import numpy as np
import cv2 as cv

from .dbow2_base import DBoW2MatcherBase
from bindings import dbow2_cpp

class ORBDBoW2Matcher(DBoW2MatcherBase):
    feature_extractor = "orb"
    matcher_norm = cv.NORM_HAMMING

    def __init__(self, nfeatures: int = 1000, grid_size: tuple[int, int] = (4, 4), k: int = 9, L: int = 3, debug: bool = False, vocabulary_path: Path | None = None):
        """Initialize the ORB DBoW2 matcher."""
        """nfeatures: total number of ORB features to extract per image (divided into 4x4 grid)"""
        """grid_size: split the image into n x n grid"""
        """k: number of branches at each node in the DBoW2 vocabulary tree"""
        """L: depth of the DBoW2 vocabulary tree"""
        """vocabulary_path: pre-trained vocabulary path, if None, create a new one using reference images"""
        self.nfeatures = nfeatures
        self.grid_size = grid_size
        self.vocabulary_path = Path(vocabulary_path) if vocabulary_path is not None else None

        self.orb = cv.ORB_create(nfeatures=nfeatures // (grid_size[0] * grid_size[1]))
        self.reference_images = []
        self.reference_places = []

        self.db = dbow2_cpp.OrbDatabase(k=k, L=L)
        self.is_built = False


    def extract_features_descriptors(self, image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        img = cv.imread(image.as_posix(), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")

        kps, dess = self.get_tiled_keypoints(img, grid_size=self.grid_size, total_features=self.nfeatures)
        
        return kps, dess
    

    # directly extract keypoints usually concentrated in a small region
    # split the image into tiles, extract keypoints and descriptors for each tile, and combine them together
    def get_tiled_keypoints(self, image: np.ndarray, grid_size: tuple[int, int]=(4, 4), total_features: int=1000) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        """Extract keypoints from the image and return them in a tiled format."""
        h, w = image.shape[:2]
        tile_h, tile_w = h // grid_size[0], w // grid_size[1]

        all_keypoints = []
        all_descriptors = []

        # define the boundaries of the grid tiles and extract keypoints and descriptors for each tile
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y1, y2 = i * tile_h, (i + 1) * tile_h
                x1, x2 = j * tile_w, (j + 1) * tile_w

                # split the image
                tile = image[y1:y2, x1:x2]

                # assign the keypoints for each tile of the image
                sub_kp, sub_des = self.orb.detectAndCompute(tile, None)
                for kp in sub_kp:
                    kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1) # pt coordinates of the keypoint [x,y]
                    all_keypoints.append(kp)
                
                if sub_des is not None:
                    all_descriptors.append(sub_des)
        
        if not all_descriptors:
            return [], None

        return all_keypoints, np.vstack(all_descriptors)
