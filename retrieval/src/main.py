from pathlib import Path
from evaluation import evaluate_csv, print_false_matches, save_false_match_images
from matchers.orb_dbow2 import ORBDBoW2Matcher

def main():
    project_root = Path("/home/kudan/prj/VPR")
    csv_path = project_root / "retrieval/data/images/converted_jpeg/labels_refined.csv"

    matcher = ORBDBoW2Matcher()

    evaluation = evaluate_csv(
        matcher,
        csv_path,
        project_root=project_root,
        top_k=5,
        show_progress=True,
    )

    print(f"Accuracy: {evaluation.accuracy:.4f}")
    print_false_matches(evaluation)

    saved_paths = save_false_match_images(
        evaluation,
        matcher,
        project_root / "retrieval/results/false_matches",
    )

    print(f"Saved {len(saved_paths)} false match images")

if __name__ == "__main__":
    main()