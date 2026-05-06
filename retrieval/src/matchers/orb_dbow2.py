from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from .matcher import ImageMatcher
# from data_loader import load_retrieval_dataset
from bindings import dbow2_cpp

class ORBDBoW2Matcher(ImageMatcher):
    def __init__(self, nfeatures: int = 1000, grid_size: tuple[int, int] = (4, 4), k: int = 9, L: int = 3, debug: bool = False, vocabulary_path: Path | None = None):
        """Initialize the ORB DBoW2 matcher."""
        """nfeatures: total number of ORB features to extract per image (divided into 4x4 grid)"""
        """grid_size: split the image into n x n grid"""
        """k: number of branches at each node in the DBoW2 vocabulary tree"""
        """L: depth of the DBoW2 vocabulary tree"""
        """debug: only for debugging, show keypoints of a image"""
        """vocabulary_path: pre-trained vocabulary path, if None, create a new one using reference images"""
        self.nfeatures = nfeatures
        self.grid_size = grid_size
        self.debug = debug
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

        # print the extracted keypoints in the image
        # if self.debug:
        #     img2 = cv.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
        #     plt.imshow(img2)
        #     plt.show()
        
        return kps, dess
    
    def match(self, query_image: Path) -> int:
        """Return the predicted place for a query image."""
        if not self.is_built:
            raise RuntimeError("Reference database has not been built. Call set_reference_database() first.")

        _, descriptors = self.extract_features_descriptors(query_image)

        if descriptors is None or descriptors.shape[0] == 0:
            raise ValueError(f"No ORB descriptors found in query image: {query_image}")

        results = self.db.query(descriptors, top_k=1)

        if not results:
            raise RuntimeError(f"DBoW2 returned no matches for query image: {query_image}")

        best_reference_id, _score = results[0]
        return self.reference_places[best_reference_id]

    def query(self, query_image: Path, top_k: int = 5) -> list[tuple[int, int, float]]:
        """Return ranked DBoW2 matches for a query image."""
        if not self.is_built:
            raise RuntimeError("Reference database has not been built. Call set_reference_database() first.")

        _, descriptors = self.extract_features_descriptors(query_image)

        if descriptors is None or descriptors.shape[0] == 0:
            raise ValueError(f"No ORB descriptors found in query image: {query_image}")

        results = self.db.query(descriptors, top_k=top_k)
        return [
            (
                reference_id,
                self.reference_places[reference_id],
                score,
            )
            for reference_id, score in results
        ]
    
    def set_reference_database(self, reference_images: list[Path], reference_places: list[int]) -> None:
        """Extract/index features for the reference images."""
        descriptors_list = []
        valid_images = []
        valid_places = []

        for img, place in zip(reference_images, reference_places):
            _, descriptors = self.extract_features_descriptors(img)

            if descriptors is None or descriptors.shape[0] == 0:
                print(f"Skipping reference image with no ORB descriptors: {img}")
                continue

            descriptors_list.append(descriptors)
            valid_images.append(img)
            valid_places.append(place)

        if not descriptors_list:
            raise ValueError("No reference images produced ORB descriptors.")

        # check if use the pre-trained vocabulary
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
        self.is_built = True

    # directly extract keypoints is not good enough
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
                kps, dess = self.orb.detectAndCompute(tile, None)
                for k in kps:
                    k.pt = (k.pt[0] + x1, k.pt[1] + y1)
                    all_keypoints.append(k)
                
                if dess is not None:
                    all_descriptors.append(dess)
        
        if not all_descriptors:
            return [], None

        return all_keypoints, np.vstack(all_descriptors)

# def main() -> None:
#     # setup the dataset root and load the dataset
#     project_root = Path(__file__).resolve().parents[4]
#     csv_path = project_root / "VPR" / "retrieval" / "data" / "images" / "converted_jpeg" / "labels_refined.csv"
#     dataset = load_retrieval_dataset(csv_path, project_root=project_root)
    
#     matcher = ORBDBoW2Matcher()

#     reference_image = dataset.references[10].image

#     # i = 1
#     # for reference in dataset.references:
#     #     print(f"Processing reference image: {reference.image}, {i}")
#     #     i += 1

#     print(f"Reference image: {reference_image}")
#     reference_places = [dataset.references[10].place]
#     matcher.set_reference_database([reference_image], reference_places)

# if __name__ == "__main__":
#     main()  
