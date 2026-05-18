from pathlib import Path

import numpy as np
import cv2 as cv
import torch

from .dbow2_base import DBoW2MatcherBase
from .verification import RankedMatch, VerifiedMatch, SuperGlueVerifier, best_verified_match
from bindings import dbow2_cpp

# import superpoint extractor from hloc
from hloc.extractors.superpoint import SuperPoint


class SuperPointDBoW2Matcher(DBoW2MatcherBase):
    feature_extractor = "superpoint"

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
        self.nfeatures = nfeatures
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
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
        self.superglue_verifier: SuperGlueVerifier | None = None
        self.superpoint_feature_cache: dict[Path, dict[str, torch.Tensor]] = {}

    def prepare_matching(self) -> None:
        if self.superglue_verifier is None:
            self.superglue_verifier = SuperGlueVerifier(
                nfeatures=self.nfeatures,
                keypoint_threshold=self.keypoint_threshold,
                nms_radius=self.nms_radius,
                resize_max=self.resize_max,
                load_superpoint=False,
            )

    def match(self, query_image: Path, ranked_matches: list[RankedMatch]) -> VerifiedMatch | None:
        if not ranked_matches:
            return None
        self.prepare_matching()

        self._ensure_superglue_features(query_image)
        verified_matches = []
        for candidate in ranked_matches:
            reference_id, _, _ = candidate
            if reference_id >= len(self.reference_images):
                continue
            self._ensure_superglue_features(self.reference_images[reference_id])
            verified_matches.append(
                self.superglue_verifier.verify(
                    query_image=query_image,
                    candidate=candidate,
                    reference_image=self.reference_images[reference_id],
                )
            )
        return best_verified_match(verified_matches)

    def _ensure_superglue_features(self, image: Path) -> None:
        image = Path(image)
        if self.superglue_verifier is None or image in self.superglue_verifier.feature_cache:
            return
        features = self.superpoint_feature_cache.get(image)
        if features is None:
            self.extract_features_descriptors(image)
            features = self.superpoint_feature_cache.get(image)
        if features is None:
            raise RuntimeError(f"Failed to cache SuperPoint features for image: {image}")
        self.superglue_verifier.add_features(image, features)

    def extract_features_descriptors(self, image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        image = Path(image)
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

        keypoint_tensor = output["keypoints"][0].detach()
        score_tensor = output["scores"][0].detach()
        descriptor_tensor = output["descriptors"][0].detach()
        kp_xy = keypoint_tensor.cpu().numpy()            # (N, 2)
        scores = score_tensor.cpu().numpy()              # (N,)
        descriptors = descriptor_tensor.T.cpu().numpy()  # (N, 256)
        superglue_keypoints = keypoint_tensor
        # rescale keypoint coords back to original image space for visualization
        if scale < 1.0:
            kp_xy = kp_xy / scale
            keypoint_tensor = keypoint_tensor / scale
        keypoints = [
            cv.KeyPoint(float(x), float(y), 8.0, -1.0, float(s))
            for (x, y), s in zip(kp_xy, scores)
        ]
        self.superpoint_feature_cache[image] = {
            "image": img_tensor,
            "keypoints": superglue_keypoints,
            "scores": score_tensor,
            "descriptors": descriptor_tensor,
        }

        if len(descriptors) == 0:
            return keypoints, None
        return keypoints, descriptors.astype(np.float32)
