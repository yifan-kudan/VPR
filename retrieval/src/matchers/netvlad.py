from pathlib import Path
import numpy as np
import cv2 as cv
import torch

from .matcher import ImageMatcher
from hloc.extractors.netvlad import NetVLAD


class NetVLADMatcher(ImageMatcher):
    default_config = {
        "resize_max": 1024,
        "resize_force": False,
    }

    def __init__(self, config: dict | None = None):
        """Initialize the NetVLAD matcher."""
        """config: optional dict. 'resize_max' caps the longest image side (GPU OOM guard);"""
        """'resize_force' resizes even when the image is already smaller;"""
        """remaining keys are forwarded to hloc's NetVLAD constructor."""
        self.config = {**self.default_config, **(config or {})}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NetVLAD running on: {self.device}")

        self.netvlad = NetVLAD(self.config).to(self.device)
        self.netvlad.eval()

        self.reference_images: list[Path] = []
        self.reference_places: list[int] = []
        self.reference_descriptors: np.ndarray | None = None  # (R, D), L2-normalized
        self.is_built = False

    def extract_features_descriptors(self, image: Path) -> tuple[list, np.ndarray | None]:
        img = cv.imread(str(image), cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Resize the image to adapt it the for NetVLAD
        h, w = img.shape[:2]
        resize_max = self.config["resize_max"]
        if resize_max and (self.config["resize_force"] or max(h, w) > resize_max):
            scale = resize_max / max(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

        # (H, W, 3) -> (1, 3, H, W) in [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.netvlad({"image": img_tensor})
        descriptor = output["global_descriptor"][0].cpu().numpy().astype(np.float32)  # (D,)

        return [], descriptor

    def set_reference_database(self, reference_images: list[Path], reference_places: list[int]) -> None:
        """Extract a global descriptor for each reference image and stack into a matrix."""
        descriptors = []
        valid_images = []
        valid_places = []

        for img, place in zip(reference_images, reference_places):
            _, desc = self.extract_features_descriptors(img)
            if desc is None or desc.size == 0:
                print(f"Skipping reference image with no NetVLAD descriptor: {img}")
                continue
            descriptors.append(desc)
            valid_images.append(img)
            valid_places.append(place)

        if not descriptors:
            raise ValueError("No reference images produced NetVLAD descriptors.")

        self.reference_descriptors = np.stack(descriptors, axis=0)  # (R, D)
        self.reference_images = valid_images
        self.reference_places = valid_places
        self.is_built = True

    def query(self, query_image: Path, top_k: int = 5) -> list[tuple[int, int, float]]:
        """Return ranked NetVLAD matches for a query image."""
        if not self.is_built:
            raise RuntimeError("Reference database has not been built. Call set_reference_database() first.")

        _, descriptor = self.extract_features_descriptors(query_image)
        if descriptor is None or descriptor.size == 0:
            raise ValueError(f"No NetVLAD descriptor found in query image: {query_image}")

        # calculate cosine similarity.
        scores = self.reference_descriptors @ descriptor 
        # TODO: calculation will be extremely memory intensive if reference database is large.
        # Consider using a more scalable approximate nearest neighbor search method if needed.

        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]

        return [(int(i), self.reference_places[int(i)], float(scores[int(i)])) for i in idx]

    def match(self, query_image: Path, potential_places: list[int] | None = None) -> int:
        """Return the predicted place for a query image."""
        # TODO: exact matching process
        pass
