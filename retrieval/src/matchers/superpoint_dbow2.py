from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from .dbow2_base import DBoW2MatcherBase
from bindings import dbow2_cpp

# import superpoint extractor from hloc
from hloc.extractors.superpoint import SuperPoint


class SuperPointDBoW2Matcher(DBoW2MatcherBase):
    def __init__(self, nfeatures=1000, k=9, L=3, vocabulary_path: Path | None = None,
                 keypoint_threshold: float = 0.005, nms_radius: int = 4, resize_max: int = 1024):
        """Initialize the SuperPoint DBoW2 matcher."""
        """nfeatures: total number of SuperPoint features to extract per image"""
        """k: number of branches at each node in the DBoW2 vocabulary tree"""
        """L: depth of the DBoW2 vocabulary tree"""
        """vocabulary_path: pre-trained vocabulary path, if None, create a new one using reference images"""
        """keypoint_threshold: SuperPoint keypoint confidence threshold"""
        """nms_radius: Non-Maximum Suppression radius"""
        """resize_max: maximum size for the longest side of the image (to avoid GPU OOM)"""
        self.vocabulary_path = Path(vocabulary_path) if vocabulary_path is not None else None
        self.resize_max = resize_max
        self.reference_images = []
        self.reference_places = []

        # init superpoint feature extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SuperPoint running on: {self.device}")
        self.superpoint = SuperPoint({
            "max_keypoints": nfeatures,
            "keypoint_threshold": keypoint_threshold,
            "nms_radius": nms_radius,
        }).to(self.device)
        self.superpoint.eval()

        # setup DBoW2 database (256-d float descriptors)
        self.db = dbow2_cpp.SuperpointDatabase(k=k, L=L)
        self.is_built = False

    def extract_features_descriptors(self, image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        img = cv.imread(image.as_posix(), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")

        # downscale so the longest side is at most resize_max (avoid GPU OOM on large images)
        h, w = img.shape[:2]
        scale = self.resize_max / max(h, w) if max(h, w) > self.resize_max else 1.0
        if scale < 1.0:
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img_small = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        else:
            img_small = img

        img_tensor = torch.from_numpy(img_small).float() / 255.0
        img_tensor = img_tensor[None, None].to(self.device)  # (1, 1, H, W)

        with torch.no_grad():
            output = self.superpoint({"image": img_tensor})

        kp_xy = output["keypoints"][0].cpu().numpy()            # (N, 2)
        scores = output["scores"][0].cpu().numpy()              # (N,)
        descriptors = output["descriptors"][0].T.cpu().numpy()  # (N, 256)
        # rescale keypoint coords back to original image space for visualization
        if scale < 1.0:
            kp_xy = kp_xy / scale
        keypoints = [
            cv.KeyPoint(float(x), float(y), 8.0, -1.0, float(s))
            for (x, y), s in zip(kp_xy, scores)
        ]

        if len(descriptors) == 0:
            return keypoints, None
        return keypoints, descriptors.astype(np.float32)