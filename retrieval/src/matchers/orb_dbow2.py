from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from .matcher import ImageMatcher
from data_loader import load_retrieval_dataset

class ORBMatcher(ImageMatcher):
    def __init__(self, nfeatures=1000, grid_size=(4, 4)):
        self.nfeatures = nfeatures
        self.grid_size = grid_size
        self.orb = cv.ORB_create(nfeatures=nfeatures // (grid_size[0] * grid_size[1]))
        self.reference_places = {}

    def extract_features_descriptors(self, image: Path) -> tuple[np.ndarray, np.ndarray]:
        img = cv.imread(image.as_posix(), cv.IMREAD_GRAYSCALE)

        kp, des = self.get_tiled_keypoints(img, grid_size=self.grid_size, total_features=self.nfeatures)

        # indicate the keypoints on the image for debugging
        img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        plt.imshow(img2), plt.show()
        
        return kp, des
    
    def match(self, query_image: Path) -> int:
        """Return the predicted place for a query image."""
        return 0  # Placeholder implementation
    
    def set_reference_database(self, reference_images: list[Path], reference_places: list[int]) -> None:
        """Extract/index features for the reference images."""

        for img, place in zip(reference_images, reference_places):
            self.reference_places[place] = (img, self.extract_features_descriptors(img))

    # directly extract keypoints is not good enough
    # split the image into tiles, extract keypoints and descriptors for each tile, and combine them together
    def get_tiled_keypoints(self, image: np.ndarray, grid_size: tuple[int, int]=(4, 4), total_features: int=1000) -> tuple[list, list]:
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
                tile = image[y1:y2, x1:x2]

                kp, des = self.orb.detectAndCompute(tile, None)

                for k in kp:
                    k.pt = (k.pt[0] + x1, k.pt[1] + y1)
                    all_keypoints.append(k)
                
                if des is not None:
                    all_descriptors.append(des)
        
        if not all_descriptors:
            return [], None

        return all_keypoints, all_descriptors

def main() -> None:
    # setup the dataset root and load the dataset
    project_root = Path(__file__).resolve().parents[4]
    csv_path = project_root / "VPR" / "retrieval" / "data" / "images" / "converted_jpeg" / "labels_refined.csv"
    dataset = load_retrieval_dataset(csv_path, project_root=project_root)
    
    matcher = ORBMatcher()

    reference_image = dataset.references[10].image
    print(f"Reference image: {reference_image}")
    reference_places = [dataset.references[10].place]
    matcher.set_reference_database([reference_image], reference_places)

if __name__ == "__main__":
    main()  