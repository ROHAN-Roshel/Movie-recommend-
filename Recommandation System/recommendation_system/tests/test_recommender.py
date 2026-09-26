



"""
tests/test_recommender.py

Automated tests for the content-based
movie recommendation engine.

Run:
    python -m pytest tests/test_recommender.py -v
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from app.config import (
    VECTORIZER_PATH,
    FEATURE_MATRIX_PATH,
    MOVIE_IDS_PATH,
    MAX_RECOMMENDATIONS,
)

from app.recommender.features import (
    extract_features,
    transform_text,
)

from app.recommender.predict import (
    MovieRecommender,
)

import app.recommender.predict as predict_module


# --------------------------------------------------
# 1. SYNTHETIC TEST DATA
# --------------------------------------------------

@pytest.fixture
def sample_movies():
    """
    A small, predictable dataset.

    These IDs are independent of the
    real TMDB dataset.
    """

    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "title": [
                "Space Journey",
                "Mars Mission",
                "Galaxy Adventure",
                "Romantic Story",
                "Haunted House",
            ],
            "combined_features": [
                (
                    "science fiction space "
                    "astronaut galaxy exploration"
                ),
                (
                    "science fiction astronaut "
                    "mars space exploration"
                ),
                (
                    "science fiction galaxy "
                    "space adventure"
                ),
                (
                    "romance drama love "
                    "relationship"
                ),
                (
                    "horror mystery haunted "
                    "ghost house"
                ),
            ],
        }
    )


# --------------------------------------------------
# 2. TRAINED TEST RECOMMENDER
# --------------------------------------------------

@pytest.fixture
def recommender(
    sample_movies,
    tmp_path,
    monkeypatch,
):
    """
    Train a small TF-IDF model and save
    artifacts in pytest's temporary folder.

    The real project artifacts are not
    overwritten or modified.
    """

    vectorizer, feature_matrix = (
        extract_features(sample_movies)
    )

    movie_ids = (
        sample_movies["id"]
        .astype(int)
        .tolist()
    )

    vectorizer_path = (
        tmp_path / "tfidf_vectorizer.joblib"
    )

    matrix_path = (
        tmp_path / "feature_matrix.joblib"
    )

    ids_path = (
        tmp_path / "movie_ids.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    joblib.dump(
        feature_matrix,
        matrix_path,
    )

    joblib.dump(
        movie_ids,
        ids_path,
    )

    # Redirect predict.py to temporary
    # test artifacts.
    monkeypatch.setattr(
        predict_module,
        "VECTORIZER_PATH",
        vectorizer_path,
    )

    monkeypatch.setattr(
        predict_module,
        "FEATURE_MATRIX_PATH",
        matrix_path,
    )

    monkeypatch.setattr(
        predict_module,
        "MOVIE_IDS_PATH",
        ids_path,
    )

    model = MovieRecommender()

    model.load_model()

    return model


# --------------------------------------------------
# 3. FEATURE EXTRACTION TESTS
# --------------------------------------------------

def test_feature_matrix_shape(
    sample_movies,
):
    """
    Every movie should produce
    exactly one feature-matrix row.
    """

    vectorizer, matrix = extract_features(
        sample_movies
    )

    assert matrix.shape[0] == len(
        sample_movies
    )

    assert matrix.shape[1] > 0

    assert len(
        vectorizer.vocabulary_
    ) > 0


def test_transform_text(
    sample_movies,
):
    """
    New text should be transformed into
    the same feature space as the movies.
    """

    vectorizer, matrix = extract_features(
        sample_movies
    )

    new_vector = transform_text(
        vectorizer,
        "astronaut exploring space",
    )

    assert new_vector.shape == (
        1,
        matrix.shape[1],
    )

    assert new_vector.nnz > 0


def test_empty_text_rejected(
    sample_movies,
):
    """
    Empty text should not be accepted
    by the feature transformer.
    """

    vectorizer, _ = extract_features(
        sample_movies
    )

    with pytest.raises(ValueError):
        transform_text(
            vectorizer,
            "   ",
        )


# --------------------------------------------------
# 4. MODEL LOADING TESTS
# --------------------------------------------------

def test_model_loads(
    recommender,
):
    """
    Verify that all artifacts
    load successfully.
    """

    assert recommender.is_loaded is True

    assert recommender.vectorizer is not None

    assert recommender.feature_matrix is not None

    assert len(recommender.movie_ids) == 5


def test_movie_id_mapping(
    recommender,
):
    """
    Verify that each movie ID maps
    to its correct matrix row.
    """

    assert (
        recommender.get_movie_index(101)
        == 0
    )

    assert (
        recommender.get_movie_index(103)
        == 2
    )

    assert (
        recommender.get_movie_index(105)
        == 4
    )


def test_missing_artifact(
    tmp_path,
    monkeypatch,
):
    """
    Loading must fail if a required
    artifact does not exist.
    """

    monkeypatch.setattr(
        predict_module,
        "VECTORIZER_PATH",
        tmp_path / "missing_vectorizer.joblib",
    )

    model = MovieRecommender()

    with pytest.raises(
        FileNotFoundError
    ):
        model.load_model()


# --------------------------------------------------
# 5. SIMILARITY TESTS
# --------------------------------------------------

def test_similarity_scores(
    recommender,
):
    """
    Cosine similarity scores should
    fall between 0 and 1.
    """

    scores = (
        recommender.calculate_similarity(
            movie_id=101
        )
    )

    assert len(scores) == 5

    assert np.all(scores >= -1e-10)

    assert np.all(scores <= 1.0 + 1e-10)


def test_self_similarity(
    recommender,
):
    """
    A nonempty TF-IDF movie vector
    should have similarity 1 with itself.
    """

    scores = (
        recommender.calculate_similarity(
            movie_id=101
        )
    )

    movie_index = (
        recommender.get_movie_index(101)
    )

    assert scores[movie_index] == (
        pytest.approx(1.0)
    )


# --------------------------------------------------
# 6. RECOMMENDATION TESTS
# --------------------------------------------------

def test_recommendation_count(
    recommender,
):
    """
    Return the requested number
    when enough candidates exist.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=3,
    )

    assert len(results) == 3


def test_selected_movie_excluded(
    recommender,
):
    """
    The selected movie must never
    appear in its own recommendations.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=4,
    )

    recommended_ids = [
        item["movie_id"]
        for item in results
    ]

    assert 101 not in recommended_ids


def test_no_duplicate_recommendations(
    recommender,
):
    """
    Every recommended movie
    should appear only once.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=4,
    )

    recommended_ids = [
        item["movie_id"]
        for item in results
    ]

    assert len(recommended_ids) == len(
        set(recommended_ids)
    )


def test_recommendations_sorted(
    recommender,
):
    """
    Recommendations should be ordered
    by descending similarity.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=4,
    )

    scores = [
        item["similarity_score"]
        for item in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_recommendation_structure(
    recommender,
):
    """
    Each recommendation must contain
    a movie ID and similarity score.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=3,
    )

    for item in results:
        assert set(item) == {
            "movie_id",
            "similarity_score",
        }

        assert isinstance(
            item["movie_id"],
            int,
        )

        assert isinstance(
            item["similarity_score"],
            float,
        )

        assert (
            0.0
            <= item["similarity_score"]
            <= 1.0
        )


def test_space_movies_are_similar(
    recommender,
):
    """
    The two strongest recommendations
    for Space Journey should be
    other space-related movies.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=2,
    )

    recommended_ids = {
        item["movie_id"]
        for item in results
    }

    assert recommended_ids == {
        102,
        103,
    }


# --------------------------------------------------
# 7. INVALID INPUT TESTS
# --------------------------------------------------

def test_unknown_movie_id(
    recommender,
):
    """
    Unknown movie IDs should
    raise ValueError.
    """

    with pytest.raises(
        ValueError,
        match="not found",
    ):
        recommender.recommend(
            movie_id=999999,
            top_n=3,
        )


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
        MAX_RECOMMENDATIONS + 1,
    ],
)
def test_invalid_top_n(
    recommender,
    top_n,
):
    """
    Reject recommendation counts
    outside the configured range.
    """

    with pytest.raises(ValueError):
        recommender.recommend(
            movie_id=101,
            top_n=top_n,
        )


def test_small_catalog(
    recommender,
):
    """
    When fewer movies are available
    than requested, return all valid
    candidates without the source movie.
    """

    results = recommender.recommend(
        movie_id=101,
        top_n=MAX_RECOMMENDATIONS,
    )

    assert len(results) == 4


# --------------------------------------------------
# 8. OPTIONAL REAL TMDB MODEL TEST
# --------------------------------------------------

def test_real_tmdb_model():
    """
    Test the project's actual trained artifacts.

    Skip this test if artifacts are unavailable.
    """

    artifact_paths = [
        VECTORIZER_PATH,
        FEATURE_MATRIX_PATH,
        MOVIE_IDS_PATH,
    ]

    if not all(
        path.exists()
        for path in artifact_paths
    ):
        pytest.skip(
            "TMDB model artifacts not found."
        )

    model = MovieRecommender()

    model.load_model()

    assert model.is_loaded

    assert len(model.movie_ids) > 1

    # Use an actual ID from the trained
    # dataset instead of assuming one.
    movie_id = model.movie_ids[0]

    results = model.recommend(
        movie_id=movie_id,
        top_n=5,
    )

    assert len(results) == min(
        5,
        len(model.movie_ids) - 1,
    )

    assert all(
        item["movie_id"] != movie_id
        for item in results
    )

    assert all(
        0.0 <= item["similarity_score"] <= 1.0
        for item in results
    )