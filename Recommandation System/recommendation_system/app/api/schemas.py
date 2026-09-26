



"""
app/api/schemas.py

Pydantic schemas for the Movie Recommendation System.

Responsibilities:
1. Validate movie-related requests.
2. Define movie response structures.
3. Validate recommendation requests.
4. Define recommendation responses.
5. Validate user feedback.
6. Define feedback responses.

Compatible with Pydantic v2.
"""

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.config import (
    TOP_N_RECOMMENDATIONS,
    MAX_RECOMMENDATIONS,
)


# --------------------------------------------------
# 1. COMMON BASE SCHEMA
# --------------------------------------------------

class APIModel(BaseModel):
    """
    Base class for API schemas.

    Forbid unexpected fields to catch
    accidental request mistakes.
    """

    model_config = ConfigDict(
        extra="forbid",
    )


# --------------------------------------------------
# 2. MOVIE SCHEMAS
# --------------------------------------------------

class MovieResponse(APIModel):
    """
    Response containing one movie's details.

    Compatible with the SQLAlchemy Movie model.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )

    id: int = Field(
        ...,
        gt=0,
        description="Unique TMDB movie ID",
    )

    title: str = Field(
        ...,
        min_length=1,
        description="Movie title",
    )

    overview: str | None = Field(
        default=None,
        description="Movie description",
    )

    genres: str | None = Field(
        default=None,
        description="Comma-separated movie genres",
    )


class MovieListResponse(APIModel):
    """
    Paginated list of movies.
    """

    movies: list[MovieResponse]

    total: int = Field(
        ...,
        ge=0,
    )

    skip: int = Field(
        default=0,
        ge=0,
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )


# --------------------------------------------------
# 3. RECOMMENDATION REQUEST
# --------------------------------------------------

class RecommendationRequest(APIModel):
    """
    Request for content-based recommendations.
    """

    movie_id: int = Field(
        ...,
        gt=0,
        description="ID of the selected movie",
        examples=[19995],
    )

    top_n: int = Field(
        default=TOP_N_RECOMMENDATIONS,
        ge=1,
        le=MAX_RECOMMENDATIONS,
        description="Number of recommendations",
    )


# --------------------------------------------------
# 4. RECOMMENDATION RESPONSES
# --------------------------------------------------

class SelectedMovieResponse(APIModel):
    """
    Details of the movie selected by the user.
    """

    movie_id: int = Field(
        ...,
        gt=0,
    )

    title: str

    overview: str | None = None

    genres: str | None = None


class RecommendedMovieResponse(
    SelectedMovieResponse
):
    """
    Recommended movie with similarity score.
    """

    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score",
    )


class RecommendationResponse(APIModel):
    """
    Complete recommendation response.

    Matches the dictionary returned by
    RecommendationService.get_recommendations().
    """

    selected_movie: SelectedMovieResponse

    recommendations: list[
        RecommendedMovieResponse
    ]

    total_recommendations: int = Field(
        ...,
        ge=0,
    )


# --------------------------------------------------
# 5. FEEDBACK REQUEST
# --------------------------------------------------

class FeedbackRequest(APIModel):
    """
    Request for recording a user interaction.

    Supported interaction types:
        view
        like
        dislike
        rating

    A rating is required only when
    interaction_type is "rating".
    """

    user_id: int = Field(
        ...,
        gt=0,
        description="Application user ID",
    )

    movie_id: int = Field(
        ...,
        gt=0,
        description="Movie receiving feedback",
    )

    interaction_type: Literal[
        "view",
        "like",
        "dislike",
        "rating",
    ]

    rating: float | None = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Rating from 1 to 5",
    )

    @model_validator(mode="after")
    def validate_rating(self):
        """
        Ensure rating values match
        the interaction type.
        """

        if self.interaction_type == "rating":
            if self.rating is None:
                raise ValueError(
                    "rating is required when "
                    "interaction_type is 'rating'."
                )

        elif self.rating is not None:
            raise ValueError(
                "rating must be omitted unless "
                "interaction_type is 'rating'."
            )

        return self


# --------------------------------------------------
# 6. FEEDBACK RESPONSE
# --------------------------------------------------

class FeedbackResponse(APIModel):
    """
    Response after saving user feedback.

    Compatible with the SQLAlchemy
    Interaction model.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )

    id: int = Field(
        ...,
        gt=0,
    )

    user_id: int = Field(
        ...,
        gt=0,
    )

    movie_id: int = Field(
        ...,
        gt=0,
    )

    interaction_type: Literal[
        "view",
        "like",
        "dislike",
        "rating",
    ]

    rating: float | None = None

    created_at: datetime


# --------------------------------------------------
# 7. COMMON ERROR RESPONSE
# --------------------------------------------------

class ErrorResponse(APIModel):
    """
    Standard structure for API errors.
    """

    detail: str