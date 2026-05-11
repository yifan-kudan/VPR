from pathlib import Path
from .matcher import LocalFeatureMatcher

class DBoW2MatcherBase(LocalFeatureMatcher):
    feature_extractor = str

    def __init__(self, db, vocabulary_path: Path | None = None):
        self.db = db
        self.vocabulary_path = vocabulary_path
        self.reference_images: list[Path] = []
        self.reference_places: list[int] = []
        self.is_built = False

    def query(self, query_image: Path, top_k: int = 5) -> list[tuple[int, int, float]]:
        if not self.is_built:
            raise RuntimeError("Reference database has not been built. Call set_reference_database()")
        
        _, descriptors = self.extract_features_descriptors(query_image)

        if descriptors is None or descriptors.shape[0] == 0:
            raise ValueError(f"No {self.feature_extractor} descriptors found in query image: {query_image}")

        results = self.db.query(descriptors, top_k = top_k)
        return [(rid, self.reference_places[rid], score) for rid, score in results]
    
    def match(self, query_image: Path, potential_places: list[int]) -> int:
        """Return the predicted place for a query image."""
        # TODO: implement matching method considering retrieval results
        pass

    def set_reference_database(self, reference_images, reference_places) -> None:
        descriptors_list, valid_images, valid_places = [], [], []
        
        for img, place in zip(reference_images, reference_places):
            _, descriptors = self.extract_features_descriptors(img)

            if descriptors is None or descriptors.shape[0] ==0:
                print(f"Skipping reference image with no {self.feature_extractor} descriptors: {img}")
                continue
    
            descriptors_list.append(descriptors)
            valid_images.append(img)
            valid_places.append(place)
        
        if not descriptors_list:
            raise ValueError(f"No reference images produced {self.feature_name} descriptors.")
        
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