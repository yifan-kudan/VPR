import argparse
from pathlib import Path
from data_loader import load_retrieval_dataset
from evaluation import evaluate_dataset, print_false_matches, save_false_match_images, save_confusion_matrix, save_match_details
from matchers.orb_dbow2 import ORBDBoW2Matcher
from matchers.sift_dbow2 import SiftDBoW2Matcher
from matchers.kornia_sift_dbow2 import KorniaSiftDBoW2Matcher
from matchers.superpoint_dbow2 import SuperPointDBoW2Matcher
from matchers.netvlad import NetVLADMatcher

# main.py lives at <repo>/retrieval/src/main.py
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a VPR retrieval evaluation.")
    parser.add_argument(
        "--algorithm",
        choices=["ORB", "SIFT", "KORNIA_SIFT", "SUPERPOINT", "NetVLAD"],
        default="NetVLAD",
        help="Matching algorithm to use.",
    )
    parser.add_argument(
        "--vocabulary-path",
        type=Path,
        default=None,
        help="Path to a pre-trained vocabulary. If None, a new vocabulary is built. Ignored for NetVLAD.",
    )
    parser.add_argument(
        "--n-references",
        type=int,
        default=1,
        help="Number of reference images per place.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top matches to retrieve.",
    )
    parser.add_argument(
        "--netvlad-resize-max",
        type=int,
        default=1024,
        help="Max image dimension when resizing for NetVLAD.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root used to resolve dataset and output paths. Defaults to the repo containing this script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    algorithm_name = args.algorithm
    vocabulary_path = args.vocabulary_path
    is_pretrained_vocabulary = vocabulary_path is not None
    n_references = args.n_references
    top_k = args.top_k
    project_root = args.project_root

    netvlad_config = {
        "resize_max": args.netvlad_resize_max,
    }

    csv_path = project_root / "retrieval/data/images/converted_jpeg/labels_refined.csv"
    suffix = f"{algorithm_name}_{'' if not is_pretrained_vocabulary else 'pretrained'}_ref_{n_references}_top_{top_k}"
    false_match_image_path = f"retrieval/results/false_matches_{suffix}"
    match_details_path = f"retrieval/results/match_details_{suffix}.csv"
    confusion_matrix_path = f"retrieval/results/confusion_matrix_{suffix}.png"

    dataset = load_retrieval_dataset(csv_path, project_root=project_root, n_references=n_references)
    image_records = {r.image: r for r in dataset.queries}

    if algorithm_name == "ORB":
        matcher = ORBDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "SIFT":
        matcher = SiftDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "KORNIA_SIFT":
        matcher = KorniaSiftDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "SUPERPOINT":
        matcher = SuperPointDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "NetVLAD":
        matcher = NetVLADMatcher(config=netvlad_config)

    evaluation = evaluate_dataset(matcher, dataset, top_k=top_k)

    print(f"Accuracy: {evaluation.accuracy:.4f}")
    print_false_matches(evaluation)

    saved_paths = save_false_match_images(
        evaluation,
        matcher,
        project_root / false_match_image_path,
    )

    print(f"Saved {len(saved_paths)} false match images")

    details_path = save_match_details(
        evaluation,
        project_root / match_details_path,
        image_records=image_records,
    )
    print(f"Saved match details to {details_path}")

    cm_path = save_confusion_matrix(
        evaluation,
        project_root / confusion_matrix_path,
    )
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    main()