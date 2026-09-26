



"""
app/recommender/train.py

Training pipeline for the Movie Recommendation System.

Responsibilities:
1. Load the cleaned movie dataset.
2. Extract TF-IDF features.
3. Validate the feature matrix.
4. Preserve movie IDs in feature-matrix row order.
5. Save trained model artifacts.
6. Display training information.
"""

import logging
from datetime import datetime, timezone

import joblib

from app.config import (
    ARTIFACTS_DIR,
    VECTORIZER_PATH,
    FEATURE_MATRIX_PATH,
    MOVIE_IDS_PATH,
    create_directories,
)

from app.recommender.features import (
    load_cleaned_dataset,
    extract_features,
)


# --------------------------------------------------
# 1. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. VALIDATE TRAINING OUTPUT
# --------------------------------------------------

def validate_training_output(
    df,
    vectorizer,
    feature_matrix,
):
    """
    Validate the fitted vectorizer and
    generated feature matrix.
    """

    if feature_matrix.shape[0] != len(df):
        raise ValueError(
            "Feature matrix row count does not "
            "match the number of movies."
        )

    if feature_matrix.shape[1] == 0:
        raise ValueError(
            "No TF-IDF features were generated."
        )

    if df["id"].isna().any():
        raise ValueError(
            "Movie IDs contain missing values."
        )

    if df["id"].duplicated().any():
        raise ValueError(
            "Duplicate movie IDs detected."
        )

    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError(
            "TF-IDF vectorizer is not fitted."
        )

    logger.info(
        "Training output validation passed."
    )


# --------------------------------------------------
# 3. SAVE MODEL ARTIFACTS
# --------------------------------------------------

def save_artifacts(
    vectorizer,
    feature_matrix,
    movie_ids,
):
    """
    Save the trained vectorizer, feature
    matrix and movie IDs using joblib.
    """

    create_directories()

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    joblib.dump(
        feature_matrix,
        FEATURE_MATRIX_PATH,
    )

    joblib.dump(
        movie_ids,
        MOVIE_IDS_PATH,
    )

    logger.info(
        "Vectorizer saved: %s",
        VECTORIZER_PATH,
    )

    logger.info(
        "Feature matrix saved: %s",
        FEATURE_MATRIX_PATH,
    )

    logger.info(
        "Movie IDs saved: %s",
        MOVIE_IDS_PATH,
    )


# --------------------------------------------------
# 4. TRAIN RECOMMENDATION MODEL
# --------------------------------------------------

def train_model():
    """
    Run the complete training pipeline.

    Returns:
        Dictionary containing training metadata.
    """

    logger.info(
        "Starting recommendation model training..."
    )

    start_time = datetime.now(timezone.utc)

    # Step 1: Load cleaned movie data
    df = load_cleaned_dataset()

    logger.info(
        "Training dataset contains %s movies.",
        len(df),
    )

    # Step 2: Extract TF-IDF features
    vectorizer, feature_matrix = (
        extract_features(df)
    )

    # Step 3: Validate training results
    validate_training_output(
        df,
        vectorizer,
        feature_matrix,
    )

    # Step 4: Preserve the exact row order
    # of the feature matrix.
    movie_ids = df["id"].astype(int).tolist()

    # Step 5: Save model artifacts
    save_artifacts(
        vectorizer,
        feature_matrix,
        movie_ids,
    )

    # Step 6: Calculate training duration
    end_time = datetime.now(timezone.utc)

    duration = (
        end_time - start_time
    ).total_seconds()

    metadata = {
        "number_of_movies": len(movie_ids),
        "number_of_features": feature_matrix.shape[1],
        "training_duration_seconds": round(
            duration, 2
        ),
        "trained_at": end_time.isoformat(),
    }

    logger.info(
        "Training completed successfully."
    )

    logger.info(
        "Movies: %s",
        metadata["number_of_movies"],
    )

    logger.info(
        "Features: %s",
        metadata["number_of_features"],
    )

    logger.info(
        "Duration: %.2f seconds",
        metadata["training_duration_seconds"],
    )

    return metadata


# --------------------------------------------------
# 5. MAIN ENTRY POINT
# --------------------------------------------------

def main():
    """
    Execute the training pipeline.
    """

    metadata = train_model()

    print("\nTraining Summary")
    print("-" * 40)

    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()