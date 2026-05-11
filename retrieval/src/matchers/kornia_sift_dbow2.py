from pathlib import Path
import numpy as np
import cv2 as cv
import torch
import kornia.feature as KF
from .dbow2_base import DBoW2MatcherBase
from bindings import dbow2_cpp


class KorniaSiftDBoW2Matcher(DBoW2MatcherBase):
    feature_extractor = "sift"

    def __init__(self, nfeatures: int = 1000, rootsift: bool = True, k: int = 9, L: int = 3, resize_max: int = 1024, vocabulary_path: Path | None = None):
        self.vocabulary_path = Path(vocabulary_path) if vocabulary_path is not None else None
        self.reference_images = []
        self.reference_places = []
        self.resize_max = resize_max

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sift = KF.SIFTFeature(num_features=nfeatures, rootsift=rootsift, device=self.device).eval()

        self.db = dbow2_cpp.SiftDatabase(k=k, L=L)
        self.is_built = False


    def extract_features_descriptors(self, image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        img = cv.imread(image.as_posix(), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")

        h, w = img.shape
        scale = min(1.0, self.resize_max / max(h, w))
        if scale < 1.0:
            img = cv.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv.INTER_AREA)

        img_tensor = torch.from_numpy(img).float().div_(255.0)[None, None].to(self.device)

        with torch.no_grad():
            lafs, _, descriptors = self.sift(img_tensor)

        kp_xy = KF.get_laf_center(lafs)[0].cpu().numpy()                  # (N, 2) in resized coords
        scales = KF.get_laf_scale(lafs)[0].squeeze(-1).squeeze(-1).cpu().numpy()  # (N,)
        oris = KF.get_laf_orientation(lafs)[0].squeeze(-1).cpu().numpy()    # (N,) degrees
        descriptors = descriptors[0].cpu().numpy().astype(np.float32)       # (N, 128)

        inv_scale = 1.0 / scale
        keypoints = [
            cv.KeyPoint(float(x) * inv_scale, float(y) * inv_scale, float(s) * 2.0 * inv_scale, float(a))
            for (x, y), s, a in zip(kp_xy, scales, oris)
        ]

        if len(descriptors) == 0:
            return keypoints, None
        return keypoints, descriptors
