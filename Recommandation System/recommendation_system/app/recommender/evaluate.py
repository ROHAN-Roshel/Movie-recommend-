



"""
app/recommender/evaluate.py

Evaluation module for the Movie Recommendation System.

Metrics:
1. Average recommendation similarity
2. Catalog coverage
3. Intra-list diversity
4. Recommendation correctness

Results are saved to artifacts/evaluation_report.json.
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from app.config import (
    ARTIFACTS_DIR,
    TOP_N_RECOMMENDATIONS,
    create_directories,
)

from app.recommender.features import (
    load_cleaned_dataset,
)

from app.recommender.predict import (
    MovieRecommender,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. LOAD EVALUATION DATA
# --------------------------------------------------

def load_evaluation_data():
    """
    Load the cleaned dataset and trained model.

    Verify that movie IDs match the saved
    feature matrix.
    """

    df = load_cleaned_dataset()

    recommender = MovieRecommender()
    recommender.load_model()

    dataset_ids = df["id"].astype(int).tolist()

    if dataset_ids != recommender.movie_ids:
        raise ValueError(
            "Dataset and model movie IDs do not "
            "match. Run train.py again."
        )

    return df, recommender


# --------------------------------------------------
# 2. SELECT EVALUATION MOVIES
# --------------------------------------------------

def select_evaluation_movies(
    recommender,
    sample_size=100,
    random_state=42,
):
    """
    Select a reproducible sample of movie IDs.

    For small datasets, evaluate all movies.
    """

    movie_ids = recommender.movie_ids

    if sample_size < 1:
        raise ValueError(
            "sample_size must be at least 1."
        )

    if len(movie_ids) < 2:
        raise ValueError(
            "At least two movies are required."
        )

    if sample_size >= len(movie_ids):
        return list(movie_ids)

    rng = np.random.default_rng(random_state)

    selected = rng.choice(
        movie_ids,
        size=sample_size,
        replace=False,
    )

    return selected.tolist()


# --------------------------------------------------
# 3. CHECK RECOMMENDATION CORRECTNESS
# --------------------------------------------------

def validate_recommendations(
    selected_movie_id,
    recommendations,
    valid_movie_ids,
    expected_count,
):
    """
    Validate one recommendation list.

    Returns:
        True if all checks pass.
    """

    recommended_ids = [
        item["movie_id"]
        for item in recommendations
    ]

    scores = [
        item["similarity_score"]
        for item in recommendations
    ]

    # Check the number of recommendations.
    if len(recommendations) != expected_count:
        return False

    # The selected movie must not appear.
    if selected_movie_id in recommended_ids:
        return False

    # Recommended movies must be unique.
    if len(recommended_ids) != len(
        set(recommended_ids)
    ):
        return False

    # Every movie ID must exist.
    if not set(recommended_ids).issubset(
        valid_movie_ids
    ):
        return False

    # Scores must be finite and between 0 and 1.
    if not all(
        np.isfinite(score) and 0 <= score <= 1
        for score in scores
    ):
        return False

    # Scores must be in descending order.
    if any(
        scores[i] < scores[i + 1]
        for i in range(len(scores) - 1)
    ):
        return False

    return True


# --------------------------------------------------
# 4. CALCULATE INTRA-LIST DIVERSITY
# --------------------------------------------------

def calculate_diversity(
    recommender,
    recommended_ids,
):
    """
    Measure differences among recommended movies.

    Diversity = 1 - mean pairwise cosine similarity

    Higher values indicate that recommended
    movies are less textually similar to
    each other.

    Requires at least two recommendations.
    """

    if len(recommended_ids) < 2:
        return None

    indices = [
        recommender.movie_id_to_index[movie_id]
        for movie_id in recommended_ids
    ]

    vectors = recommender.feature_matrix[indices]

    similarity_matrix = cosine_similarity(
        vectors
    )

    # Select only unique pairs.
    upper_indices = np.triu_indices(
        len(indices),
        k=1,
    )

    pairwise_scores = similarity_matrix[
        upper_indices
    ]

    diversity = 1.0 - float(
        np.mean(pairwise_scores)
    )

    return float(np.clip(diversity, 0, 1))


# --------------------------------------------------
# 5. EVALUATE THE RECOMMENDER
# --------------------------------------------------

def evaluate_model(
    recommender,
    sample_size=100,
    top_k=TOP_N_RECOMMENDATIONS,
):
    """
    Evaluate the recommendation engine.

    Returns:
        Dictionary of evaluation metrics.
    """

    if not 1 <= top_k <= len(
        recommender.movie_ids
    ) - 1:
        raise ValueError(
            "top_k must be between 1 and "
            "the catalog size minus one."
        )

    evaluation_ids = select_evaluation_movies(
        recommender,
        sample_size=sample_size,
    )

    valid_ids = set(recommender.movie_ids)

    all_recommended_ids = set()

    similarity_values = []
    diversity_values = []

    valid_lists = 0

    for movie_id in evaluation_ids:

        recommendations = recommender.recommend(
            movie_id=movie_id,
            top_n=top_k,
        )

        # Check correctness.
        is_valid = validate_recommendations(
            selected_movie_id=movie_id,
            recommendations=recommendations,
            valid_movie_ids=valid_ids,
            expected_count=top_k,
        )

        if not is_valid:
            raise ValueError(
                "Invalid recommendations for "
                f"movie ID {movie_id}."
            )

        valid_lists += 1

        # Collect similarity scores.
        similarity_values.extend(
            item["similarity_score"]
            for item in recommendations
        )

        # Collect recommended movie IDs.
        recommended_ids = [
            item["movie_id"]
            for item in recommendations
        ]

        all_recommended_ids.update(
            recommended_ids
        )

        # Calculate diversity for this list.
        diversity = calculate_diversity(
            recommender,
            recommended_ids,
        )

        if diversity is not None:
            diversity_values.append(diversity)

    # Average similarity across all results.
    average_similarity = float(
        np.mean(similarity_values)
    )

    # Fraction of catalog appearing in
    # recommendations for sampled movies.
    catalog_coverage = (
        len(all_recommended_ids)
        / len(valid_ids)
    )

    # Average diversity across recommendation lists.
    average_diversity = (
        float(np.mean(diversity_values))
        if diversity_values
        else None
    )

    correctness_rate = (
        valid_lists / len(evaluation_ids)
    )

    report = {
        "evaluated_movies": len(evaluation_ids),
        "catalog_size": len(valid_ids),
        "top_k": top_k,
        "average_similarity": round(
            average_similarity, 4
        ),
        "catalog_coverage": round(
            catalog_coverage, 4
        ),
        "average_diversity": (
            round(average_diversity, 4)
            if average_diversity is not None
            else None
        ),
        "correctness_rate": round(
            correctness_rate, 4
        ),
        "unique_recommended_movies": len(
            all_recommended_ids
        ),
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    return report


# --------------------------------------------------
# 6. SAVE EVALUATION REPORT
# --------------------------------------------------

def save_report(report):
    """
    Save evaluation results as a JSON file.
    """

    create_directories()

    report_path = (
        ARTIFACTS_DIR / "evaluation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Evaluation report saved to %s",
        report_path,
    )

    return report_path


# --------------------------------------------------
# 7. MAIN ENTRY POINT
# --------------------------------------------------

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    df, recommender = load_evaluation_data()

    logger.info(
        "Loaded %s movies for evaluation.",
        len(df),
    )

    # Use all movies for a small dataset.
    sample_size = min(100, len(df))

    # Do not request more recommendations
    # than the available catalog allows.
    top_k = min(
        TOP_N_RECOMMENDATIONS,
        len(df) - 1,
    )

    if top_k < 1:
        raise ValueError(
            "At least two movies are required."
        )

    report = evaluate_model(
        recommender,
        sample_size=sample_size,
        top_k=top_k,
    )

    save_report(report)

    print("\nRECOMMENDATION EVALUATION")
    print("-" * 45)

    for metric, value in report.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()