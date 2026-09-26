



"""
app/services/movie_service.py

Service layer for movie-related operations.

Responsibilities:
1. Retrieve movie details by ID.
2. Retrieve movies by title.
3. List movies with pagination.
4. Search movies by title.
5. Retrieve multiple movies by ID.
6. Import cleaned movies into the database.

The service uses MovieRepository instead of
executing database queries directly.
"""

import logging

import pandas as pd

from sqlalchemy.orm import Session

from app.config import (
    CLEANED_MOVIES_PATH,
    MAX_RECOMMENDATIONS,
)

from app.db.repository import MovieRepository


logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. MOVIE SERVICE
# --------------------------------------------------

class MovieService:
    """
    Business logic for movie operations.
    """

    def __init__(self, db: Session):
        self.repository = MovieRepository(db)

    # --------------------------------------------------
    # 2. GET MOVIE BY ID
    # --------------------------------------------------

    def get_movie(self, movie_id: int):
        """
        Retrieve one movie by its ID.

        Returns:
            Movie object or None.
        """

        if movie_id < 1:
            raise ValueError(
                "Movie ID must be positive."
            )

        return self.repository.get_by_id(
            movie_id
        )

    # --------------------------------------------------
    # 3. GET MOVIES BY TITLE
    # --------------------------------------------------

    def get_movies_by_title(self, title: str):
        """
        Retrieve movies with an exact title match.

        The repository performs a
        case-insensitive comparison.

        Returns:
            List of Movie objects.
        """

        if not isinstance(title, str):
            raise TypeError(
                "Movie title must be a string."
            )

        title = title.strip()

        if not title:
            raise ValueError(
                "Movie title cannot be empty."
            )

        return self.repository.get_by_title(
            title
        )

    # --------------------------------------------------
    # 4. LIST MOVIES
    # --------------------------------------------------

    def list_movies(
        self,
        skip: int = 0,
        limit: int = 20,
    ):
        """
        Retrieve a paginated list of movies.
        """

        if skip < 0:
            raise ValueError(
                "skip cannot be negative."
            )

        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        return self.repository.get_all(
            skip=skip,
            limit=limit,
        )

    # --------------------------------------------------
    # 5. SEARCH MOVIES
    # --------------------------------------------------

    def search_movies(
        self,
        keyword: str,
        limit: int = 20,
    ):
        """
        Search movies by title.
        """

        if not isinstance(keyword, str):
            raise TypeError(
                "Search keyword must be a string."
            )

        keyword = keyword.strip()

        if not keyword:
            return []

        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        return self.repository.search(
            keyword=keyword,
            limit=limit,
        )

    # --------------------------------------------------
    # 6. GET MULTIPLE MOVIES BY ID
    # --------------------------------------------------

    def get_movies_by_ids(
        self,
        movie_ids: list[int],
    ):
        """
        Retrieve multiple movies.

        Preserves the input order of movie IDs.

        Useful when the recommendation engine
        returns ranked movie IDs.
        """

        if not movie_ids:
            return []

        if any(
            movie_id < 1
            for movie_id in movie_ids
        ):
            raise ValueError(
                "All movie IDs must be positive."
            )

        return self.repository.get_by_ids(
            movie_ids
        )

    # --------------------------------------------------
    # 7. COUNT MOVIES
    # --------------------------------------------------

    def count_movies(self):
        """
        Return the number of movies
        currently stored in SQLite.
        """

        return self.repository.count()

    # --------------------------------------------------
    # 8. IMPORT CLEANED DATASET
    # --------------------------------------------------

    def import_movies_from_csv(
        self,
        csv_path=CLEANED_MOVIES_PATH,
    ):
        """
        Import cleaned movies into SQLite.

        Existing movie IDs are skipped to
        avoid duplicate primary keys.

        Returns:
            Dictionary containing import statistics.
        """

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        df = pd.read_csv(
            csv_path,
            keep_default_na=False,
        )

        required_columns = {
            "id",
            "title",
            "overview",
            "genres",
        }

        missing = required_columns - set(
            df.columns
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing)}"
            )

        if df["id"].duplicated().any():
            raise ValueError(
                "Duplicate movie IDs in CSV."
            )

        imported = 0
        skipped = 0

        for row in df.itertuples(
            index=False
        ):
            movie_id = int(row.id)

            existing_movie = (
                self.repository.get_by_id(
                    movie_id
                )
            )

            if existing_movie is not None:
                skipped += 1
                continue

            self.repository.create(
                movie_id=movie_id,
                title=str(row.title),
                overview=str(row.overview),
                genres=str(row.genres),
            )

            imported += 1

        logger.info(
            "Imported %s movies; skipped %s.",
            imported,
            skipped,
        )

        return {
            "imported": imported,
            "skipped": skipped,
            "total_rows": len(df),
        }