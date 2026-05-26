from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import torch

from hloc.extractors.superpoint import SuperPoint
from hloc.matchers.superglue import SuperGlue


RankedMatch = tuple[int, Any, float]


@dataclass(frozen=True)
class VerifiedMatch:
    reference_id: int
    place: Any
    reference_image: Path
    retrieval_score: float
    num_raw_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    verification_score: float = 0.0

# use RANSAC to verify the features
def _ransac_inliers(
    query_keypoints: list[cv.KeyPoint], 
    reference_keypoints: list[cv.KeyPoint], 
    matches: list[cv.DMatch], reprojection_threshold: float) -> int:
    if len(matches) < 4:
        return 0

    query_points = np.float32([query_keypoints[m.queryIdx].pt for m in matches])
    reference_points = np.float32([reference_keypoints[m.trainIdx].pt for m in matches])
    _, mask = cv.findHomography(
        query_points,
        reference_points,
        cv.RANSAC,
        reprojection_threshold,
    )
    if mask is None:
        return 0
    return int(mask.sum())


def geometric_verified_match(
    query_features: tuple[list[cv.KeyPoint], np.ndarray | None],
    reference_features: tuple[list[cv.KeyPoint], np.ndarray | None],
    candidate: RankedMatch,
    reference_image: Path,
    norm_type: int,
    ratio_threshold: float = 0.8,
    reprojection_threshold: float = 5.0,
) -> VerifiedMatch:
    query_keypoints, query_descriptors = query_features
    reference_keypoints, reference_descriptors = reference_features

    # use KNN to match the feature points
    matcher = cv.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)

    # call RANSAC verification to score the matching
    num_inliers = _ransac_inliers(
        query_keypoints,
        reference_keypoints,
        good_matches,
        reprojection_threshold,
    )
    num_raw_matches = len(good_matches)
    inlier_ratio = num_inliers / num_raw_matches if num_raw_matches else 0.0

    # record the matching score of each top K predicted place, and return
    reference_id, place, retrieval_score = candidate
    return VerifiedMatch(
        reference_id=reference_id,
        place=place,
        reference_image=reference_image,
        retrieval_score=retrieval_score,
        num_raw_matches=num_raw_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        verification_score=float(num_inliers),
    )


class SuperGlueVerifier:
    def __init__(
        self,
        nfeatures: int = 1024,
        keypoint_threshold: float = 0.005,
        nms_radius: int = 4,
        resize_max: int = 1024,
        weights: str = "outdoor",
        sinkhorn_iterations: int = 50,
        match_threshold: float = 0.2,
        ransac_reprojection_threshold: float = 5.0,
        load_superpoint: bool = True,
    ) -> None:
        self.resize_max = resize_max
        self.ransac_reprojection_threshold = ransac_reprojection_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.superpoint = None

        if load_superpoint:
            self.superpoint = SuperPoint(
                {
                    "max_keypoints": nfeatures,
                    "keypoint_threshold": keypoint_threshold,
                    "nms_radius": nms_radius,
                }
            ).to(self.device)
            self.superpoint.eval()

        self.superglue = SuperGlue(
            {
                "weights": weights,
                "sinkhorn_iterations": sinkhorn_iterations,
                "match_threshold": match_threshold,
            }
        ).to(self.device)
        self.superglue.eval()
        self.feature_cache: dict[Path, dict[str, torch.Tensor]] = {}

    def add_features(self, image: Path, features: dict[str, torch.Tensor]) -> None:
        self.feature_cache[Path(image)] = {
            "image": features["image"].detach().to(self.device),
            "keypoints": features["keypoints"].detach().to(self.device),
            "scores": features["scores"].detach().to(self.device),
            "descriptors": features["descriptors"].detach().to(self.device),
        }

    # extract features using superpoint
    def extract(self, image: Path) -> dict[str, torch.Tensor]:
        image = Path(image)
        cached = self.feature_cache.get(image)

        if cached is not None:
            return cached
        if self.superpoint is None:
            raise RuntimeError(f"No cached SuperPoint features available for image: {image}")

        img = cv.imread(image.as_posix(), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")

        h, w = img.shape[:2]
        scale = self.resize_max / max(h, w) if max(h, w) > self.resize_max else 1.0
        if scale < 1.0:
            resized_size = (int(round(w * scale)), int(round(h * scale)))
            img = cv.resize(img, resized_size, interpolation=cv.INTER_AREA)

        tensor = torch.from_numpy(img).float().div_(255.0)[None, None].to(self.device)
        with torch.no_grad():
            pred = self.superpoint({"image": tensor})

        features = {
            "image": tensor,
            "keypoints": pred["keypoints"][0].detach(),
            "scores": pred["scores"][0].detach(),
            "descriptors": pred["descriptors"][0].detach(),
        }
        self.feature_cache[image] = features
        return features

    # use superglue to match the features, and use RANSAC to verify the matching
    def verify(self, query_image: Path, candidate: RankedMatch, reference_image: Path) -> VerifiedMatch:
        query_features = self.extract(query_image)
        reference_features = self.extract(reference_image)

        data = {
            "image0": query_features["image"],
            "keypoints0": query_features["keypoints"][None].to(self.device),
            "scores0": query_features["scores"][None].to(self.device),
            "descriptors0": query_features["descriptors"][None].to(self.device),
            "image1": reference_features["image"],
            "keypoints1": reference_features["keypoints"][None].to(self.device),
            "scores1": reference_features["scores"][None].to(self.device),
            "descriptors1": reference_features["descriptors"][None].to(self.device),
        }

        with torch.no_grad():
            pred = self.superglue(data)

        matches0 = pred["matches0"][0].detach().cpu().numpy()
        valid = matches0 > -1
        num_raw_matches = int(valid.sum())

        num_inliers = 0
        if num_raw_matches >= 4:
            query_points = query_features["keypoints"].detach().cpu().numpy()[valid]
            reference_points = reference_features["keypoints"].detach().cpu().numpy()[matches0[valid]]
            _, mask = cv.findHomography(
                query_points.astype(np.float32),
                reference_points.astype(np.float32),
                cv.RANSAC,
                self.ransac_reprojection_threshold,
            )
            if mask is not None:
                num_inliers = int(mask.sum())

        inlier_ratio = num_inliers / num_raw_matches if num_raw_matches else 0.0
        verification_score = float(num_inliers)
        reference_id, place, retrieval_score = candidate

        return VerifiedMatch(
            reference_id=reference_id,
            place=place,
            reference_image=reference_image,
            retrieval_score=retrieval_score,
            num_raw_matches=num_raw_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            verification_score=verification_score,
        )


def best_verified_match(matches: list[VerifiedMatch]) -> VerifiedMatch | None:
    if not matches:
        return None
    return max(
        matches,
        key=lambda match: (
            match.verification_score,
            match.num_inliers,
            match.num_raw_matches,
            match.retrieval_score,
        ),
    )
