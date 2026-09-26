
"""
app/config.py

Central configuration for the Recommendation System.

This module defines:
1. Project directory paths
2. Dataset locations
3. Database configuration
4. Machine learning artifact paths
5. Recommendation settings
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# 1. PROJECT ROOT DIRECTORY
# --------------------------------------------------

# Absolute path to recommendation_system/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the project root
load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# 2. DATASET PATHS
# --------------------------------------------------

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_MOVIES_PATH = RAW_DATA_DIR / "movies.csv"

CLEANED_MOVIES_PATH = (
    PROCESSED_DATA_DIR / "cleaned_movies.csv"
)


# --------------------------------------------------
# 3. DATABASE CONFIGURATION
# --------------------------------------------------

# SQLite database file
DATABASE_PATH = BASE_DIR / "recommendation.db"

# Can be overridden using the DATABASE_URL
# environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


# --------------------------------------------------
# 4. MACHINE LEARNING ARTIFACT PATHS
# --------------------------------------------------

ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Trained TF-IDF vectorizer
VECTORIZER_PATH = (
    ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
)

# TF-IDF feature matrix
FEATURE_MATRIX_PATH = (
    ARTIFACTS_DIR / "feature_matrix.joblib"
)

# Movie IDs corresponding to feature matrix rows
MOVIE_IDS_PATH = (
    ARTIFACTS_DIR / "movie_ids.joblib"
)


# --------------------------------------------------
# 5. RECOMMENDATION SETTINGS
# --------------------------------------------------

# Number of movies to recommend by default
TOP_N_RECOMMENDATIONS = 5

# Maximum number of recommendations allowed
MAX_RECOMMENDATIONS = 20


# --------------------------------------------------
# 6. APPLICATION SETTINGS
# --------------------------------------------------

APP_NAME = "Movie Recommendation System"

APP_VERSION = "1.0.0"

DEBUG = os.getenv(
    "DEBUG", "false"
).lower() == "true"


# --------------------------------------------------
# 7. CREATE REQUIRED DIRECTORIES
# --------------------------------------------------

def create_directories():
    """
    Create required project directories
    if they do not already exist.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )