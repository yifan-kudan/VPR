from pathlib import Path
from data_loader import load_retrieval_dataset
from evaluation import evaluate_setup, print_false_matches, save_false_match_images, save_confusion_matrix, save_match_details
from matchers.orb_dbow2 import ORBDBoW2Matcher
from matchers.sift_dbow2 import SiftDBoW2Matcher
from matchers.superpoint_dbow2 import SuperPointDBoW2Matcher
from matchers.netvlad import NetVLADMatcher

def main() -> None:
    # matching algorithm: "ORB", "SIFT", "SUPERPOINT", or "NetVLAD"
    algorithm_name = "NetVLAD"

    netvlad_config = {
        "resize_max": 1024,
    }

    # setup the pre-trained vocabulary path
    # if None, a new vocabulary will be created
    # vocabulary_path = project_root / "retrieval/data/ORBvoc.txt"
    vocabulary_path = None
    is_pretrained_vocabulary = vocabulary_path is not None

    n_references = 1 # number of reference images per place
    top_k = 1 # number of top matches

    project_root = Path("/home/kudan/prj/VPR")

    csv_path = project_root / "retrieval/data/images/converted_jpeg/labels_refined.csv"
    false_match_image_dir = f"retrieval/results/false_matches_{algorithm_name}_{'' if not is_pretrained_vocabulary else 'pretrained'}_ref_{n_references}_top_{top_k}"
    match_details_dir = f"retrieval/results/match_details_{algorithm_name}_{'' if not is_pretrained_vocabulary else 'pretrained'}_ref_{n_references}_top_{top_k}.csv"
    confusion_matrix_dir = f"retrieval/results/confusion_matrix_{algorithm_name}_{'' if not is_pretrained_vocabulary else 'pretrained'}_ref_{n_references}_top_{top_k}.png"

    dataset = load_retrieval_dataset(csv_path, project_root=project_root, n_references=n_references)
    image_records = {r.image: r for r in dataset.queries}

    # initialize the matcher based on chosen algorithm
    if algorithm_name == "ORB":
        matcher = ORBDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "SIFT":
        matcher = SiftDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "SUPERPOINT":
        matcher = SuperPointDBoW2Matcher(vocabulary_path=vocabulary_path)
    elif algorithm_name == "NetVLAD":
        matcher = NetVLADMatcher(config=netvlad_config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name!r}. Choose 'ORB', 'SIFT', 'SUPERPOINT', or 'NetVLAD'.")

    evaluation = evaluate_setup(
        matcher,
        csv_path,
        project_root=project_root,
        top_k=top_k,
        n_references=n_references
    )

    print(f"Accuracy: {evaluation.accuracy:.4f}")
    print_false_matches(evaluation)

    saved_paths = save_false_match_images(
        evaluation,
        matcher,
        project_root / false_match_image_dir,
    )

    print(f"Saved {len(saved_paths)} false match images")

    details_path = save_match_details(
        evaluation,
        project_root / match_details_dir,
        image_records=image_records,
    )
    print(f"Saved match details to {details_path}")

    cm_path = save_confusion_matrix(
        evaluation,
        project_root / confusion_matrix_dir,
    )
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    main()