from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from data_loader import RetrievalDataset, ImageRecord, load_retrieval_dataset

from tqdm import tqdm


RankedMatch = tuple[int, Any, float]


@dataclass(frozen=True)
class MatchResult:
    query_image: Path
    true_place: Any
    predicted_place: Any
    matched_reference_image: Path | None
    correct: bool
    ranked_matches: list[RankedMatch]


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    results: list[MatchResult]


def query_matcher(matcher, query_image: Path, top_k: int) -> tuple[Any, list[RankedMatch]]:
    if hasattr(matcher, "query"):
        ranked_matches = matcher.query(query_image, top_k=top_k)
        predicted_place = ranked_matches[0][1] if ranked_matches else None
        return predicted_place, ranked_matches

    return matcher.match(query_image), []


def matched_reference_image(matcher, ranked_matches: list[RankedMatch]) -> Path | None:
    if not ranked_matches or not hasattr(matcher, "reference_images"):
        return None

    reference_id = ranked_matches[0][0]
    return matcher.reference_images[reference_id]


def accuracy_calculation(results: list[MatchResult]) -> float:
    accuracy = sum(result.correct for result in results) / len(results) if results else 0.0
    return accuracy


def evaluate(
    matcher,
    query_images: list[Path],
    query_places: list[Any],
    top_k: int = 5,
) -> EvaluationResult:
    """Evaluate query images and record per-query retrieval results."""
    results = []
    query_pairs = list(zip(query_images, query_places))

    query_pairs = tqdm(query_pairs, desc="Matching queries", unit="image")

    for query_image, true_place in query_pairs:
        predicted_place, ranked_matches = query_matcher(matcher, query_image, top_k)

        results.append(
            MatchResult(
                query_image=query_image,
                true_place=true_place,
                predicted_place=predicted_place,
                matched_reference_image=matched_reference_image(matcher, ranked_matches),
                correct=predicted_place == true_place,
                ranked_matches=ranked_matches,
            )
        )

    accuracy = accuracy_calculation(results)
    return EvaluationResult(accuracy=accuracy, results=results)


def evaluate_dataset(
    matcher,
    dataset: RetrievalDataset,
    top_k: int = 5,
) -> EvaluationResult:
    """Build the matcher from dataset references and evaluate dataset queries."""
    matcher.set_reference_database(dataset.reference_images, dataset.reference_places)
    return evaluate(
        matcher,
        dataset.query_images,
        dataset.query_places,
        top_k=top_k
    )


def evaluate_csv(
    matcher,
    csv_path: str | Path,
    project_root: str | Path | None = None,
    top_k: int = 5,
    n_references: int = 1,
    validate_files: bool = True,
) -> EvaluationResult:
    """Load a retrieval dataset CSV, build references, and evaluate queries."""
    dataset = load_retrieval_dataset(
        csv_path,
        project_root=project_root,
        validate_files=validate_files,
        n_references=n_references,
    )
    return evaluate_dataset(matcher, dataset, top_k=top_k)


def print_false_matches(evaluation: EvaluationResult) -> None:
    """Print query/reference image pairs for incorrect retrievals."""
    for result in evaluation.results:
        if result.correct:
            continue

        print(
            "False match: "
            f"query={result.query_image} "
            f"matched_reference={result.matched_reference_image} "
            f"true_place={result.true_place} "
            f"predicted_place={result.predicted_place}"
        )


def image_with_keypoints(matcher, image_path: Path):
    if cv is None:
        raise ImportError("OpenCV is required to save false-match images. Install cv2 first.")

    image = cv.imread(image_path.as_posix(), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if not hasattr(matcher, "extract_features_descriptors"):
        return image

    keypoints, _descriptors = matcher.extract_features_descriptors(image_path)
    return cv.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def add_label(image, label: str):
    output = image.copy()
    cv.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), thickness=-1)
    cv.putText(
        output,
        label,
        (10, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        thickness=2,
        lineType=cv.LINE_AA,
    )
    return output


def resize_to_height(image, height: int):
    scale = height / image.shape[0]
    width = max(1, int(image.shape[1] * scale))
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def save_match_details(
    evaluation: EvaluationResult,
    output_path: str | Path,
    image_records: dict[Path, "ImageRecord"] | None = None,
) -> Path:
    """Save per-query match details to a CSV for offline analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for result in evaluation.results:
        top_score = result.ranked_matches[0][2] if result.ranked_matches else None
        row: dict[str, Any] = {
            "query_image": result.query_image,
            "true_place": result.true_place,
            "predicted_place": result.predicted_place,
            "correct": result.correct,
            "top_score": top_score,
            "matched_reference_image": result.matched_reference_image,
        }
        if image_records is not None:
            rec = image_records.get(result.query_image)
            if rec is not None:
                row["direction"] = rec.direction
                row["light"] = rec.light
                row["weather"] = rec.weather
                row["indoor"] = rec.indoor
                row["construction"] = rec.construction
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def save_confusion_matrix(
    evaluation: EvaluationResult,
    output_path: str | Path,
) -> Path:
    """Plot and save a confusion matrix of true vs predicted places."""
    if plt is None:
        raise ImportError("matplotlib is required to save the confusion matrix.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    true_places = sorted(set(r.true_place for r in evaluation.results))
    pred_places = sorted(set(r.predicted_place for r in evaluation.results if r.predicted_place is not None))
    all_places = sorted(set(true_places) | set(pred_places))
    place_to_idx = {p: i for i, p in enumerate(all_places)}

    n = len(all_places)
    matrix = np.zeros((n, n), dtype=int)
    for result in evaluation.results:
        if result.predicted_place is None:
            continue
        true_idx = place_to_idx[result.true_place]
        pred_idx = place_to_idx.get(result.predicted_place)
        if pred_idx is not None:
            matrix[true_idx, pred_idx] += 1

    fig_size = max(8, n // 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_font = max(4, min(8, 120 // n))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_places, rotation=90, fontsize=tick_font)
    ax.set_yticklabels(all_places, fontsize=tick_font)
    ax.set_xlabel("Predicted place")
    ax.set_ylabel("True place")
    ax.set_title(f"Confusion Matrix  —  Accuracy {evaluation.accuracy:.4f}")

    if n <= 40:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=tick_font, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_false_match_images(
    evaluation: EvaluationResult,
    matcher,
    output_dir: str | Path,
) -> list[Path]:
    """Save each false query/reference pair as one side-by-side JPG with ORB keypoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    false_results = [result for result in evaluation.results if not result.correct]

    iterator = false_results
    if tqdm is not None:
        iterator = tqdm(false_results, desc="Saving false matches", unit="pair")

    for index, result in enumerate(iterator):
        if result.matched_reference_image is None:
            continue

        query_image = image_with_keypoints(matcher, result.query_image)
        reference_image = image_with_keypoints(matcher, result.matched_reference_image)

        height = min(query_image.shape[0], reference_image.shape[0], 900)
        query_image = resize_to_height(query_image, height)
        reference_image = resize_to_height(reference_image, height)

        query_image = add_label(query_image, f"Query true={result.true_place}")
        reference_image = add_label(reference_image, f"Matched predicted={result.predicted_place}")

        pair_image = cv.hconcat([query_image, reference_image])
        output_path = output_dir / (
            f"false_match_{index:04d}_true_{result.true_place}_pred_{result.predicted_place}.jpg"
        )

        cv.imwrite(output_path.as_posix(), pair_image)
        saved_paths.append(output_path)

    return saved_paths
