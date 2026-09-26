




"""
app/api/routes/feedback.py

FastAPI endpoints for user feedback.

Endpoints:
    POST   /feedback
    GET    /feedback/users/{user_id}
    GET    /feedback/movies/{movie_id}
    DELETE /feedback/{interaction_id}

Supported interactions:
    view
    like
    dislike
    rating
"""

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Response,
    status,
)

from sqlalchemy.orm import Session

from app.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
)

from app.db.database import get_db

from app.db.repository import (
    InteractionRepository,
    MovieRepository,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. CREATE ROUTER
# --------------------------------------------------

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)


# --------------------------------------------------
# 2. DATABASE DEPENDENCY
# --------------------------------------------------

def get_interaction_repository(
    db: Session = Depends(get_db),
) -> InteractionRepository:
    """
    Create an interaction repository
    using the current database session.
    """

    return InteractionRepository(db)


InteractionRepositoryDep = Annotated[
    InteractionRepository,
    Depends(get_interaction_repository),
]


def get_movie_repository(
    db: Session = Depends(get_db),
) -> MovieRepository:
    """
    Create a movie repository to verify
    that movie IDs exist.
    """

    return MovieRepository(db)


MovieRepositoryDep = Annotated[
    MovieRepository,
    Depends(get_movie_repository),
]


# --------------------------------------------------
# 3. CREATE USER FEEDBACK
# --------------------------------------------------

@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record user feedback",
    description=(
        "Record a movie view, like, "
        "dislike or rating."
    ),
)
def create_feedback(
    request: FeedbackRequest,
    interaction_repository: InteractionRepositoryDep,
    movie_repository: MovieRepositoryDep,
):
    """
    Save a new interaction.

    The Pydantic schema validates the
    interaction type and rating value.
    """

    # Verify that the movie exists.
    movie = movie_repository.get_by_id(
        request.movie_id
    )

    if movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Movie with ID {request.movie_id} "
                "was not found."
            ),
        )

    try:
        interaction = (
            interaction_repository.create(
                user_id=request.user_id,
                movie_id=request.movie_id,
                interaction_type=(
                    request.interaction_type
                ),
                rating=request.rating,
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(exc),
        ) from exc

    logger.info(
        "Recorded %s interaction: "
        "user=%s, movie=%s",
        request.interaction_type,
        request.user_id,
        request.movie_id,
    )

    return interaction


# --------------------------------------------------
# 4. GET USER FEEDBACK HISTORY
# --------------------------------------------------

@router.get(
    "/users/{user_id}",
    response_model=list[FeedbackResponse],
    summary="Get user feedback history",
)
def get_user_feedback(
    interaction_repository: InteractionRepositoryDep,
    user_id: Annotated[
        int,
        Path(
            gt=0,
            description="Application user ID",
        ),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description=(
                "Maximum interactions to return"
            ),
        ),
    ] = 20,
):
    """
    Retrieve the user's recent interactions.

    Results are ordered from newest
    to oldest by the repository.
    """

    return interaction_repository.get_by_user(
        user_id=user_id,
        limit=limit,
    )


# --------------------------------------------------
# 5. GET FEEDBACK FOR A MOVIE
# --------------------------------------------------

@router.get(
    "/movies/{movie_id}",
    response_model=list[FeedbackResponse],
    summary="Get feedback for a movie",
)
def get_movie_feedback(
    interaction_repository: InteractionRepositoryDep,
    movie_repository: MovieRepositoryDep,
    movie_id: Annotated[
        int,
        Path(
            gt=0,
            description="TMDB movie ID",
        ),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description=(
                "Maximum interactions to return"
            ),
        ),
    ] = 20,
):
    """
    Retrieve recent feedback for a movie.
    """

    movie = movie_repository.get_by_id(
        movie_id
    )

    if movie is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                f"Movie with ID {movie_id} "
                "was not found."
            ),
        )

    return interaction_repository.get_by_movie(
        movie_id=movie_id,
        limit=limit,
    )


# --------------------------------------------------
# 6. DELETE FEEDBACK
# --------------------------------------------------

@router.delete(
    "/{interaction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete feedback",
)
def delete_feedback(
    interaction_repository: InteractionRepositoryDep,
    interaction_id: Annotated[
        int,
        Path(
            gt=0,
            description="Interaction ID",
        ),
    ],
):
    """
    Delete an interaction by its ID.
    """

    deleted = interaction_repository.delete(
        interaction_id
    )

    if not deleted:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                f"Interaction with ID "
                f"{interaction_id} was not found."
            ),
        )

    logger.info(
        "Deleted interaction ID %s",
        interaction_id,
    )

    return Response(
        status_code=(
            status.HTTP_204_NO_CONTENT
        )
    )