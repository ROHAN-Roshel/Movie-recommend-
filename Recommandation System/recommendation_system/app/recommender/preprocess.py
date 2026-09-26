



"""
app/recommender/preprocess.py

Data preprocessing pipeline for the
Movie Recommendation System.

Responsibilities:
1. Load the raw movie dataset.
2. Validate required columns.
3. Remove duplicate and invalid records.
4. Handle missing values.
5. Normalize movie genres.
6. Clean movie descriptions.
7. Save the processed dataset.
"""

import ast
import logging
import re

import pandas as pd

from app.config import (
    RAW_MOVIES_PATH,
    CLEANED_MOVIES_PATH,
    create_directories,
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
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset():
    """
    Load the raw movie dataset.

    Returns:
        pandas.DataFrame
    """

    if not RAW_MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_MOVIES_PATH}"
        )

    logger.info("Loading movie dataset...")

    df = pd.read_csv(RAW_MOVIES_PATH)

    logger.info(
        "Loaded %s movie records",
        len(df),
    )

    return df


# --------------------------------------------------
# 3. VALIDATE DATASET
# --------------------------------------------------

def validate_dataset(df):
    """
    Check whether required columns exist.

    Accepts 'id' or TMDB's 'movie_id'.
    """

    if "movie_id" in df.columns and "id" not in df.columns:
        df = df.rename(
            columns={"movie_id": "id"}
        )

    required_columns = {
        "id",
        "title",
        "overview",
        "genres",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


# --------------------------------------------------
# 4. NORMALIZE GENRES
# --------------------------------------------------

def normalize_genres(value):
    """
    Convert genres to comma-separated text.

    Supports:
        Action,Adventure

    And TMDB-style data:
        [
            {"id": 28, "name": "Action"},
            {"id": 12, "name": "Adventure"}
        ]
    """

    if pd.isna(value):
        return ""

    if not isinstance(value, str):
        return ""

    value = value.strip()

    if not value:
        return ""

    # TMDB genres are stored as JSON-like strings.
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                names = [
                    item["name"].strip()
                    for item in parsed
                    if isinstance(item, dict)
                    and isinstance(
                        item.get("name"), str
                    )
                ]

                return ",".join(names)

        except (ValueError, SyntaxError):
            logger.warning(
                "Could not parse genres: %s",
                value[:80],
            )

            return ""

    # Already comma-separated
    genres = [
        genre.strip()
        for genre in value.split(",")
        if genre.strip()
    ]

    return ",".join(genres)


# --------------------------------------------------
# 5. CLEAN TEXT
# --------------------------------------------------

def clean_text(value):
    """
    Normalize whitespace in movie descriptions.

    Preserve punctuation and capitalization
    for display purposes.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    # Replace repeated whitespace with one space
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# --------------------------------------------------
# 6. PREPROCESS DATASET
# --------------------------------------------------

def preprocess_dataset(df):
    """
    Execute the complete cleaning pipeline.
    """

    logger.info("Starting preprocessing...")

    df = validate_dataset(df)

    # Keep only the columns needed by our system
    df = df[
        ["id", "title", "overview", "genres"]
    ].copy()

    # Convert IDs to numeric values
    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    # Remove records without valid IDs
    df = df.dropna(subset=["id"])

    # Keep only integer IDs
    df = df[
        df["id"] % 1 == 0
    ].copy()

    df["id"] = df["id"].astype("int64")

    # Remove duplicate movie IDs
    df = df.drop_duplicates(
        subset=["id"],
        keep="first",
    )

    # Clean movie titles
    df["title"] = df["title"].apply(
        clean_text
    )

    # Remove records without titles
    df = df[df["title"] != ""]

    # Clean movie descriptions
    df["overview"] = df["overview"].apply(
        clean_text
    )

    # Normalize movie genres
    df["genres"] = df["genres"].apply(
        normalize_genres
    )

    # Keep movies with a description or genres
    df = df[
        (df["overview"] != "")
        | (df["genres"] != "")
    ].copy()

    # Create a combined text column for ML
    df["combined_features"] = (
        df["genres"].str.replace(
            ",",
            " ",
            regex=False,
        )
        + " "
        + df["overview"]
    ).str.strip()

    # Reset row indexes
    df = df.reset_index(drop=True)

    logger.info(
        "Preprocessing complete: %s movies",
        len(df),
    )

    return df


# --------------------------------------------------
# 7. SAVE PROCESSED DATASET
# --------------------------------------------------

def save_dataset(df):
    """
    Save the cleaned dataset as a CSV file.
    """

    create_directories()

    df.to_csv(
        CLEANED_MOVIES_PATH,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Cleaned dataset saved to %s",
        CLEANED_MOVIES_PATH,
    )


# --------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------

def main():
    """
    Run the complete preprocessing pipeline.
    """

    df = load_dataset()

    cleaned_df = preprocess_dataset(df)

    if cleaned_df.empty:
        raise ValueError(
            "No valid movies remain after preprocessing."
        )

    save_dataset(cleaned_df)

    logger.info(
        "Successfully processed %s movies",
        len(cleaned_df),
    )


if __name__ == "__main__":
    main()