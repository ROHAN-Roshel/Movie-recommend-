



"""
app/services/recommendation_service.py

Service layer for generating movie recommendations.

Responsibilities:
1. Validate recommendation requests.
2. Verify that the selected movie exists.
3. Generate content-based recommendations.
4. Retrieve recommended movie details.
5. Combine movie details with similarity scores.
6. Preserve recommendation ranking.

The service connects MovieRecommender
with MovieService.
"""

import logging

from sqlalchemy.orm import Session

from app.config import (
    TOP_N_RECOMMENDATIONS,
    MAX_RECOMMENDATIONS,
)

from app.recommender.predict import (
    MovieRecommender,
)

from app.services.movie_service import (
    MovieService,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. RECOMMENDATION SERVICE
# --------------------------------------------------

class RecommendationService:
    """
    Coordinates the recommendation engine
    and movie database operations.
    """

    def __init__(
        self,
        db: Session,
        recommender: MovieRecommender | None = None,
    ):
        self.movie_service = MovieService(db)

        # Allow an existing recommender instance
        # to be reused across multiple requests.
        self.recommender = (
            recommender
            if recommender is not None
            else MovieRecommender()
        )

    # --------------------------------------------------
    # 2. VALIDATE REQUEST
    # --------------------------------------------------

    def validate_request(
        self,
        movie_id: int,
        top_n: int,
    ):
        """
        Validate the requested movie ID
        and number of recommendations.
        """

        if (
            not isinstance(movie_id, int)
            or isinstance(movie_id, bool)
            or movie_id < 1
        ):
            raise ValueError(
                "movie_id must be a positive integer."
            )

        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or not 1 <= top_n <= MAX_RECOMMENDATIONS
        ):
            raise ValueError(
                "top_n must be an integer between "
                f"1 and {MAX_RECOMMENDATIONS}."
            )

    # --------------------------------------------------
    # 3. GET RECOMMENDATIONS
    # --------------------------------------------------

    def get_recommendations(
        self,
        movie_id: int,
        top_n: int = TOP_N_RECOMMENDATIONS,
    ):
        """
        Generate recommendations for a movie.

        Returns:
            Dictionary containing the selected
            movie and recommended movie details.
        """

        self.validate_request(
            movie_id=movie_id,
            top_n=top_n,
        )

        # Step 1: Verify the selected movie
        # exists in the database.
        selected_movie = (
            self.movie_service.get_movie(movie_id)
        )

        if selected_movie is None:
            raise ValueError(
                f"Movie ID {movie_id} "
                "does not exist in the database."
            )

        # Step 2: Load the recommendation model.
        if not self.recommender.is_loaded:
            self.recommender.load_model()

        # Step 3: Verify the selected movie
        # exists in the trained model.
        if (
            movie_id
            not in self.recommender.movie_id_to_index
        ):
            raise ValueError(
                f"Movie ID {movie_id} "
                "is not available in the "
                "trained recommendation model."
            )

        # Step 4: Generate ranked predictions.
        predictions = self.recommender.recommend(
            movie_id=movie_id,
            top_n=top_n,
        )

        # Step 5: Extract recommended movie IDs.
        recommended_ids = [
            item["movie_id"]
            for item in predictions
        ]

        # Step 6: Retrieve movie details
        # in the same ranking order.
        movies = (
            self.movie_service.get_movies_by_ids(
                recommended_ids
            )
        )

        movie_map = {
            movie.id: movie
            for movie in movies
        }

        # Step 7: Combine predictions
        # with movie details.
        recommendations = []

        for prediction in predictions:
            recommended_movie_id = (
                prediction["movie_id"]
            )

            movie = movie_map.get(
                recommended_movie_id
            )

            if movie is None:
                logger.warning(
                    "Recommended movie ID %s "
                    "is missing from the database.",
                    recommended_movie_id,
                )
                continue

            recommendations.append({
                "movie_id": movie.id,
                "title": movie.title,
                "overview": movie.overview,
                "genres": movie.genres,
                "similarity_score": (
                    prediction["similarity_score"]
                ),
            })

        # Step 8: Build the final response.
        result = {
            "selected_movie": {
                "movie_id": selected_movie.id,
                "title": selected_movie.title,
                "overview": selected_movie.overview,
                "genres": selected_movie.genres,
            },
            "recommendations": recommendations,
            "total_recommendations": len(
                recommendations
            ),
        }

        logger.info(
            "Generated %s recommendations "
            "for movie ID %s.",
            len(recommendations),
            movie_id,
        )

        return result


# --------------------------------------------------
# 4. MAIN FUNCTION FOR MANUAL TESTING
# --------------------------------------------------

def main():
    """
    Test the recommendation service.
    """

    from app.db.database import SessionLocal

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    db = SessionLocal()

    try:
        service = RecommendationService(db)

        result = service.get_recommendations(
            movie_id=19995,
            top_n=5,
        )

        print("\nSelected movie:")
        print(
            result["selected_movie"]["title"]
        )

        print("\nRecommended movies:")
        print("-" * 50)

        for rank, movie in enumerate(
            result["recommendations"],
            start=1,
        ):
            print(
                f"{rank}. {movie['title']} "
                f"(ID: {movie['movie_id']}) "
                f"- Similarity: "
                f"{movie['similarity_score']:.4f}"
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()