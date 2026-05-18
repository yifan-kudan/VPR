from pathlib import Path

import cv2 as cv
import numpy as np

from .matcher import LocalFeatureMatcher
from .verification import RankedMatch, VerifiedMatch, best_verified_match, geometric_verified_match

class DBoW2MatcherBase(LocalFeatureMatcher):
    feature_extractor: str = None
    matcher_norm: int = cv.NORM_L2

    def __init__(self, db, vocabulary_path: Path | None = None):
        self.db = db
        self.vocabulary_path = vocabulary_path
        self.reference_images: list[Path] = []
        self.reference_places: list[int] = []
        self.reference_features: list[tuple[list[cv.KeyPoint], np.ndarray | None]] = []
        self.is_built = False
        self._last_query_features: tuple[Path, tuple[list[cv.KeyPoint], np.ndarray | None]] | None = None

    # save the query image features temporarily for both retrieval and matching
    def _get_query_features(self, query_image: Path) -> tuple[list[cv.KeyPoint], np.ndarray | None]:
        # query() and match() are called back-to-back on the same image (see evaluation loop),
        # so a one-slot cache avoids extracting features twice per query.
        if self._last_query_features is not None and self._last_query_features[0] == query_image:
            return self._last_query_features[1]
        features = self.extract_features_descriptors(query_image)
        self._last_query_features = (query_image, features)
        return features

    def query(self, query_image: Path, top_k: int = 5) -> list[RankedMatch]:
        if not self.is_built:
            raise RuntimeError("Reference database has not been built. Call set_reference_database()")

        _, descriptors = self._get_query_features(query_image)

        if descriptors is None or descriptors.shape[0] == 0:
            raise ValueError(f"No {self.feature_extractor} descriptors found in query image: {query_image}")

        results = self.db.query(descriptors, top_k = top_k)
        return [(rid, self.reference_places[rid], score) for rid, score in results]

    def match(self, query_image: Path, ranked_matches: list[RankedMatch]) -> VerifiedMatch | None:
        """Verify retrieved candidates and return the best geometric match."""
        if not ranked_matches:
            return None

        query_features = self._get_query_features(query_image)
        verified_matches = []
        for candidate in ranked_matches:
            reference_id, _, _ = candidate

            verified_matches.append(
                geometric_verified_match(
                    query_features=query_features,
                    reference_features=self.reference_features[reference_id],
                    candidate=candidate,
                    reference_image=self.reference_images[reference_id],
                    norm_type=self.matcher_norm,
                )
            )
        return best_verified_match(verified_matches)

    def set_reference_database(self, reference_images, reference_places) -> None:
        descriptors_list, valid_images, valid_places, valid_features = [], [], [], []
        
        for img, place in zip(reference_images, reference_places):
            keypoints, descriptors = self.extract_features_descriptors(img)

            if descriptors is None or descriptors.shape[0] ==0:
                print(f"Skipping reference image with no {self.feature_extractor} descriptors: {img}")
                continue
    
            descriptors_list.append(descriptors)
            valid_images.append(img)
            valid_places.append(place)
            valid_features.append((keypoints, descriptors))
        
        if not descriptors_list:
            raise ValueError(f"No reference images produced {self.feature_extractor} descriptors.")
        
        if self.vocabulary_path is not None:
            print(f"Loading pre-trained vocabulary from {self.vocabulary_path}")
            self.db.load_vocabulary(str(self.vocabulary_path))
        else:
            print("Creating new vocabulary from reference images.")
            self.db.create_vocabulary(descriptors_list)
        
        for descriptors in descriptors_list:
            self.db.add(descriptors)
        
        self.reference_images = valid_images
        self.reference_places = valid_places
        self.reference_features = valid_features
        self.is_built = True
