from matchers.orb_dbow2 import ORBDBoW2Matcher
from retrieval.src.data_process.preprocess import preprocess_images
from evaluation import evaluate
from pathlib import Path
def main() -> None:
    # Example database and query images (replace with actual paths)
    database_images = [Path("database/image1.jpg"), Path("database/image2.jpg")]
    database_labels = ["label1", "label2"]
    query_images = [Path("query/image1.jpg"), Path("query/image2.jpg")]
    query_labels = ["label1", "label2"]

    # Preprocess images (if needed)
    preprocess_images(database_images + query_images)

    # Initialize and build the matcher
    matcher = ORBDBoW2Matcher(database_images, database_labels)
    matcher.build()

    # Evaluate the matcher
    accuracy = evaluate(matcher, query_images, query_labels)
    print(f"Accuracy: {accuracy:.2f}")