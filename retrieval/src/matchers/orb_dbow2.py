from __future__ import annotations

from pathlib import Path

import cv2 as cv

from matcher import ImageMatcher


class ORBMatcher(ImageMatcher):
    def __init__(self, n_features: int = 500) -> None:
        self.orb = cv.ORB_create(nfeatures=n_features)
        self.reference_features = []

    def extract(self, image_path: str | Path):
        img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        keypoints = self.orb.detect(img, None)
        keypoints, descriptors = self.orb.compute(img, keypoints)

        return img, keypoints, descriptors

    def visualize(self, image_path: str | Path) -> None:
        img, keypoints, descriptors = self.extract(image_path)

        img_with_keypoints = cv.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=0,
        )

        print(f"Image: {image_path}")
        print(f"Keypoints: {len(keypoints)}")

        if descriptors is None:
            print("Descriptors: None")
        else:
            print(f"Descriptors shape: {descriptors.shape}")

        cv.imshow("ORB keypoints", img_with_keypoints)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def build(self, reference_images: list[Path], reference_places: list[int]) -> None:
        self.reference_features = []

        for image_path, place in zip(reference_images, reference_places):
            img, keypoints, descriptors = self.extract(image_path)
            self.reference_features.append(
                {
                    "image": image_path,
                    "place": place,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                }
            )

    def match(self, query_image: Path) -> int:
        raise NotImplementedError("DBoW matching will be added later.")
