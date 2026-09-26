



"""
app/db/repository.py

Database access layer for the Movie
Recommendation System.

All database queries are centralized here.

Repositories:
    MovieRepository
    InteractionRepository
"""

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.movie import Movie
from app.models.interaction import Interaction


# --------------------------------------------------
# 1. MOVIE REPOSITORY
# --------------------------------------------------

class MovieRepository:
    """
    Handles database operations related to movies.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, movie_id: int):
        """Retrieve a movie using its ID."""

        return self.db.get(Movie, movie_id)

    def get_by_title(self, title: str):
        """Find movies with an exact title."""

        stmt = select(Movie).where(
            func.lower(Movie.title) == title.lower()
        )

        return self.db.scalars(stmt).all()

    def get_all(self, skip: int = 0, limit: int = 20):
        """Retrieve a paginated list of movies."""

        if skip < 0 or limit < 1:
            raise ValueError(
                "skip must be >= 0 and limit must be >= 1"
            )

        stmt = (
            select(Movie)
            .order_by(Movie.id)
            .offset(skip)
            .limit(limit)
        )

        return self.db.scalars(stmt).all()

    def search(self, keyword: str, limit: int = 20):
        """Search for movies by title."""

        if not keyword.strip():
            return []

        if limit < 1:
            raise ValueError("limit must be >= 1")

        stmt = (
            select(Movie)
            .where(Movie.title.ilike(f"%{keyword}%"))
            .order_by(Movie.title)
            .limit(limit)
        )

        return self.db.scalars(stmt).all()

    def get_by_ids(self, movie_ids: list[int]):
        """
        Retrieve multiple movies.

        Preserve the order of the supplied IDs.
        """

        if not movie_ids:
            return []

        stmt = select(Movie).where(
            Movie.id.in_(movie_ids)
        )

        movies = self.db.scalars(stmt).all()

        movie_map = {
            movie.id: movie
            for movie in movies
        }

        return [
            movie_map[movie_id]
            for movie_id in movie_ids
            if movie_id in movie_map
        ]

    def create(
        self,
        movie_id: int,
        title: str,
        overview: str | None = None,
        genres: str | None = None,
    ):
        """Insert a new movie."""

        movie = Movie(
            id=movie_id,
            title=title,
            overview=overview,
            genres=genres,
        )

        self.db.add(movie)
        self.db.commit()
        self.db.refresh(movie)

        return movie

    def update(
        self,
        movie_id: int,
        **changes,
    ):
        """Update allowed movie fields."""

        movie = self.get_by_id(movie_id)

        if movie is None:
            return None

        allowed_fields = {
            "title",
            "overview",
            "genres",
        }

        unknown_fields = set(changes) - allowed_fields

        if unknown_fields:
            raise ValueError(
                f"Invalid fields: {unknown_fields}"
            )

        for field, value in changes.items():
            setattr(movie, field, value)

        self.db.commit()
        self.db.refresh(movie)

        return movie

    def delete(self, movie_id: int):
        """Delete a movie without interaction records."""

        movie = self.get_by_id(movie_id)

        if movie is None:
            return False

        interaction_exists = self.db.scalar(
            select(Interaction.id)
            .where(Interaction.movie_id == movie_id)
            .limit(1)
        )

        if interaction_exists is not None:
            raise ValueError(
                "Cannot delete a movie with interactions"
            )

        self.db.delete(movie)
        self.db.commit()

        return True

    def count(self):
        """Return the total number of movies."""

        stmt = select(func.count(Movie.id))

        return self.db.scalar(stmt)


# --------------------------------------------------
# 2. INTERACTION REPOSITORY
# --------------------------------------------------

class InteractionRepository:
    """
    Handles user interactions with movies.
    """

    VALID_TYPES = {
        "view",
        "like",
        "dislike",
        "rating",
    }

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        user_id: int,
        movie_id: int,
        interaction_type: str,
        rating: float | None = None,
    ):
        """Record a user interaction."""

        if interaction_type not in self.VALID_TYPES:
            raise ValueError(
                "Invalid interaction type"
            )

        if interaction_type == "rating":
            if (
                rating is None
                or not 1 <= rating <= 5
            ):
                raise ValueError(
                    "Rating must be between 1 and 5"
                )
        elif rating is not None:
            raise ValueError(
                "Only rating interactions can "
                "have a rating value"
            )

        movie = self.db.get(Movie, movie_id)

        if movie is None:
            raise ValueError(
                f"Movie {movie_id} does not exist"
            )

        interaction = Interaction(
            user_id=user_id,
            movie_id=movie_id,
            interaction_type=interaction_type,
            rating=rating,
        )

        self.db.add(interaction)
        self.db.commit()
        self.db.refresh(interaction)

        return interaction

    def get_by_id(self, interaction_id: int):
        """Retrieve one interaction."""

        return self.db.get(
            Interaction,
            interaction_id,
        )

    def get_by_user(
        self,
        user_id: int,
        limit: int = 100,
    ):
        """Retrieve a user's recent interactions."""

        if limit < 1:
            raise ValueError("limit must be >= 1")

        stmt = (
            select(Interaction)
            .where(Interaction.user_id == user_id)
            .order_by(
                Interaction.created_at.desc(),
                Interaction.id.desc(),
            )
            .limit(limit)
        )

        return self.db.scalars(stmt).all()

    def get_by_movie(
        self,
        movie_id: int,
        limit: int = 100,
    ):
        """Retrieve interactions for a movie."""

        if limit < 1:
            raise ValueError("limit must be >= 1")

        stmt = (
            select(Interaction)
            .where(Interaction.movie_id == movie_id)
            .order_by(
                Interaction.created_at.desc(),
                Interaction.id.desc(),
            )
            .limit(limit)
        )

        return self.db.scalars(stmt).all()

    def get_user_ratings(self, user_id: int):
        """Retrieve all rating events for a user."""

        stmt = (
            select(Interaction)
            .where(
                Interaction.user_id == user_id,
                Interaction.interaction_type == "rating",
            )
            .order_by(Interaction.created_at.desc())
        )

        return self.db.scalars(stmt).all()

    def delete(self, interaction_id: int):
        """Delete an interaction."""

        interaction = self.get_by_id(
            interaction_id
        )

        if interaction is None:
            return False

        self.db.delete(interaction)
        self.db.commit()

        return True