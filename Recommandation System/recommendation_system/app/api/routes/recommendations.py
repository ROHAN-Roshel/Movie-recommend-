




"""
app/api/routes/recommendations.py

FastAPI endpoints for movie recommendations.

Endpoints:
    POST /recommendations
    GET  /recommendations/{movie_id}

Responsibilities:
1. Validate recommendation requests.
2. Load and reuse the trained model.
3. Call RecommendationService.
4. Return movie details and similarity scores.
5. Convert service errors into HTTP responses.
"""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)

from sqlalchemy.orm import Session

from app.config import (
    TOP_N_RECOMMENDATIONS,
    MAX_RECOMMENDATIONS,
)

from app.db.database import get_db

from app.api.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    ErrorResponse,
)

from app.recommender.predict import (
    MovieRecommender,
)

from app.services.recommendation_service import (
    RecommendationService,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. CREATE ROUTER
# --------------------------------------------------

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


# --------------------------------------------------
# 2. LOAD AND CACHE THE RECOMMENDER
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_recommender():
    """
    Load the trained model once per process.

    Subsequent requests reuse the same
    MovieRecommender instance.
    """

    recommender = MovieRecommender()

    recommender.load_model()

    return recommender


# --------------------------------------------------
# 3. RECOMMENDATION SERVICE DEPENDENCY
# --------------------------------------------------

def get_recommendation_service(
    db: Session = Depends(get_db),
):
    """
    Create a service using the current
    database session and cached model.
    """

    try:
        recommender = get_recommender()

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:
        logger.error(
            "Recommendation model unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Recommendation model is unavailable. "
                "Check the trained artifacts."
            ),
        ) from exc

    return RecommendationService(
        db=db,
        recommender=recommender,
    )


RecommendationServiceDependency = Annotated[
    RecommendationService,
    Depends(get_recommendation_service),
]


# --------------------------------------------------
# 4. HANDLE SERVICE ERRORS
# --------------------------------------------------

def generate_recommendations(
    service: RecommendationService,
    movie_id: int,
    top_n: int,
):
    """
    Generate recommendations and translate
    service errors into HTTP responses.
    """

    try:
        return service.get_recommendations(
            movie_id=movie_id,
            top_n=top_n,
        )

    except ValueError as exc:
        message = str(exc)

        logger.warning(
            "Recommendation request failed: %s",
            message,
        )

        if (
            "does not exist in the database"
            in message
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_404_NOT_FOUND
                ),
                detail=message,
            ) from exc

        if (
            "not available in the "
            "trained recommendation model"
            in message
        ):
            raise HTTPException(
                status_code=(
                    status.HTTP_409_CONFLICT
                ),
                detail=(
                    "This movie exists in the "
                    "database but is missing from "
                    "the trained model. "
                    "Retrain the model."
                ),
            ) from exc

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=message,
        ) from exc


# --------------------------------------------------
# 5. POST RECOMMENDATIONS
# --------------------------------------------------

@router.post(
    "",
    response_model=RecommendationResponse,
    summary="Generate movie recommendations",
    description=(
        "Generate content-based movie "
        "recommendations using a TMDB movie ID."
    ),
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Movie not found",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "Database and model are out of sync"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Recommendation model unavailable"
            ),
        },
    },
)
def recommend_movies(
    request: RecommendationRequest,
    service: RecommendationServiceDependency,
):
    """
    Generate recommendations from a JSON body.
    """

    return generate_recommendations(
        service=service,
        movie_id=request.movie_id,
        top_n=request.top_n,
    )


# --------------------------------------------------
# 6. GET RECOMMENDATIONS BY MOVIE ID
# --------------------------------------------------

@router.get(
    "/{movie_id}",
    response_model=RecommendationResponse,
    summary="Get recommendations by movie ID",
    description=(
        "Generate recommendations using "
        "a movie ID in the URL."
    ),
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Movie not found",
        },
        409: {
            "model": ErrorResponse,
            "description": (
                "Database and model are out of sync"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Recommendation model unavailable"
            ),
        },
    },
)
def get_movie_recommendations(
    service: RecommendationServiceDependency,
    movie_id: Annotated[
        int,
        Path(
            gt=0,
            description="TMDB movie ID",
            examples=[19995],
        ),
    ],
    top_n: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_RECOMMENDATIONS,
            description=(
                "Number of recommendations"
            ),
        ),
    ] = TOP_N_RECOMMENDATIONS,
):
    """
    Generate recommendations using
    URL and query parameters.
    """

    return generate_recommendations(
        service=service,
        movie_id=movie_id,
        top_n=top_n,
    )
