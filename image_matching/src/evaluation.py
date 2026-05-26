from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from data_loader import RetrievalDataset, ImageRecord
from matchers.matcher import GlobalDescriptorMatcher, LocalFeatureMatcher
from matchers.verification import VerifiedMatch

from tqdm import tqdm


RankedMatch = tuple[int, Any, float]


@dataclass(frozen=True)
class MatchResult:
    query_image: Path
    true_place: Any
    retrieval_top1_place: Any
    retrieval_top1_score: float | None
    retrieval_hit_at_k: bool
    matched_place: Any
    matched_reference_image: Path | None
    matching_correct: bool
    verified_match: VerifiedMatch | None
    ranked_matches: list[RankedMatch]


@dataclass(frozen=True)
class EvaluationResult:
    retrieval_top1_accuracy: float
    retrieval_recall_at_k: float
    matching_accuracy: float
    matching_success_rate: float
    mean_raw_matches: float
    mean_inliers: float
    mean_inlier_ratio: float
    results: list[MatchResult]


def query_matcher(matcher, query_image: Path, top_k: int) -> list[RankedMatch]:

    ranked_matches = matcher.query(query_image, top_k=top_k)
    return ranked_matches


def retrieval_top1_place(ranked_matches: list[RankedMatch]) -> Any:
    return ranked_matches[0][1] if ranked_matches else None


def retrieval_top1_score(ranked_matches: list[RankedMatch]) -> float | None:
    return ranked_matches[0][2] if ranked_matches else None


def retrieval_hit_at_k(ranked_matches: list[RankedMatch], true_place: Any) -> bool:
    return any(place == true_place for _, place, _ in ranked_matches)


def matched_reference_image(verified_match: VerifiedMatch | None) -> Path | None:
    if verified_match is None:
        return None
    return verified_match.reference_image


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluation_metrics(results: list[MatchResult]) -> dict[str, float]:
    if not results:
        return {
            "retrieval_top1_accuracy": 0.0,
            "retrieval_recall_at_k": 0.0,
            "matching_accuracy": 0.0,
            "matching_success_rate": 0.0,
            "mean_raw_matches": 0.0,
            "mean_inliers": 0.0,
            "mean_inlier_ratio": 0.0,
        }

    verified_matches = [result.verified_match for result in results if result.verified_match is not None]
    return {
        "retrieval_top1_accuracy": sum(result.retrieval_top1_place == result.true_place for result in results) / len(results),
        "retrieval_recall_at_k": sum(result.retrieval_hit_at_k for result in results) / len(results),
        "matching_accuracy": sum(result.matching_correct for result in results) / len(results),
        "matching_success_rate": sum(match.num_raw_matches > 0 for match in verified_matches) / len(results),
        "mean_raw_matches": safe_mean([float(match.num_raw_matches) for match in verified_matches]),
        "mean_inliers": safe_mean([float(match.num_inliers) for match in verified_matches]),
        "mean_inlier_ratio": safe_mean([float(match.inlier_ratio) for match in verified_matches]),
    }


def evaluate_dataset(
    matcher,
    dataset: RetrievalDataset,
    top_k: int = 5,
) -> EvaluationResult:
    """Build references from the dataset and evaluate its queries."""
    matcher.set_reference_database(dataset.reference_images, dataset.reference_places)
    
    # prepare verifier first to avoid progress bar out of alignment
    if hasattr(matcher, "prepare_matching"):
        matcher.prepare_matching()

    results = []
    query_pairs = tqdm(
        list(zip(dataset.query_images, dataset.query_places)),
        desc="Matching queries",
        unit="image",
    )
    
    for query_image, true_place in query_pairs:
        ranked_matches = query_matcher(matcher, query_image, top_k)
        verified_match = matcher.match(query_image, ranked_matches)
        matched_place = verified_match.place if verified_match is not None else None
        results.append(
            MatchResult(
                query_image=query_image,
                true_place=true_place,
                retrieval_top1_place=retrieval_top1_place(ranked_matches),
                retrieval_top1_score=retrieval_top1_score(ranked_matches),
                retrieval_hit_at_k=retrieval_hit_at_k(ranked_matches, true_place),
                matched_place=matched_place,
                matched_reference_image=matched_reference_image(verified_match),
                matching_correct=matched_place == true_place,
                verified_match=verified_match,
                ranked_matches=ranked_matches,
            )
        )

    metrics = evaluation_metrics(results)
    return EvaluationResult(
        retrieval_top1_accuracy=metrics["retrieval_top1_accuracy"],
        retrieval_recall_at_k=metrics["retrieval_recall_at_k"],
        matching_accuracy=metrics["matching_accuracy"],
        matching_success_rate=metrics["matching_success_rate"],
        mean_raw_matches=metrics["mean_raw_matches"],
        mean_inliers=metrics["mean_inliers"],
        mean_inlier_ratio=metrics["mean_inlier_ratio"],
        results=results,
    )


def print_false_matches(evaluation: EvaluationResult) -> None:
    """Print query/reference image pairs for incorrect retrievals."""
    for result in evaluation.results:
        if result.matching_correct:
            continue

        print(
            "False match: "
            f"query={result.query_image} "
            f"matched_reference={result.matched_reference_image} "
            f"true_place={result.true_place} "
            f"matched_place={result.matched_place} "
            f"retrieval_top1_place={result.retrieval_top1_place}"
        )


def image_with_keypoints(matcher, image_path: Path):
    if cv is None:
        raise ImportError("OpenCV is required to save false-match images. Install cv2 first.")

    image = cv.imread(image_path.as_posix(), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if not hasattr(matcher, "extract_features_descriptors"):
        return image

    keypoints, _ = matcher.extract_features_descriptors(image_path)
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
        verified_match = result.verified_match
        row: dict[str, Any] = {
            "query_image": result.query_image,
            "true_place": result.true_place,
            "retrieval_top1_place": result.retrieval_top1_place,
            "retrieval_top1_score": result.retrieval_top1_score,
            "retrieval_hit_at_k": result.retrieval_hit_at_k,
            "matched_place": result.matched_place,
            "matching_correct": result.matching_correct,
            "matched_reference_image": result.matched_reference_image,
            "verification_score": verified_match.verification_score if verified_match is not None else None,
            "num_raw_matches": verified_match.num_raw_matches if verified_match is not None else None,
            "num_inliers": verified_match.num_inliers if verified_match is not None else None,
            "inlier_ratio": verified_match.inlier_ratio if verified_match is not None else None,
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
    pred_places = sorted(set(r.matched_place for r in evaluation.results if r.matched_place is not None))
    all_places = sorted(set(true_places) | set(pred_places))
    place_to_idx = {p: i for i, p in enumerate(all_places)}

    n = len(all_places)
    matrix = np.zeros((n, n), dtype=int)
    for result in evaluation.results:
        if result.matched_place is None:
            continue
        true_idx = place_to_idx[result.true_place]
        pred_idx = place_to_idx.get(result.matched_place)
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
    ax.set_title(f"Confusion Matrix  —  Matching Accuracy {evaluation.matching_accuracy:.4f}")

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
    matcher: LocalFeatureMatcher | GlobalDescriptorMatcher,
    output_dir: str | Path,
) -> list[Path]:
    """Save each false query/reference pair as one side-by-side JPG with keypoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    false_results = [result for result in evaluation.results if not result.matching_correct]

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
        reference_image = add_label(reference_image, f"Matched predicted={result.matched_place}")

        pair_image = cv.hconcat([query_image, reference_image])
        output_path = output_dir / (
            f"false_match_{index:04d}_true_{result.true_place}_pred_{result.matched_place}.jpg"
        )

        cv.imwrite(output_path.as_posix(), pair_image)
        saved_paths.append(output_path)

    return saved_paths
