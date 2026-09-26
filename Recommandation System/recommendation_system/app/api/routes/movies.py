




"""
app/api/routes/movies.py

FastAPI endpoints for movie-related operations.

Endpoints:
    GET /movies
    GET /movies/search
    GET /movies/count
    GET /movies/{movie_id}

Responsibilities:
1. Retrieve movies with pagination.
2. Search movies by title.
3. Retrieve movie details by ID.
4. Return the total movie count.
5. Validate HTTP request parameters.

Business logic is handled by MovieService.
"""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)

from pydantic import BaseModel, Field

from sqlalchemy.orm import Session

from app.db.database import get_db

from app.services.movie_service import (
    MovieService,
)

from app.api.schemas import (
    MovieResponse,
    MovieListResponse,
)


# --------------------------------------------------
# 1. CREATE ROUTER
# --------------------------------------------------

router = APIRouter(
    prefix="/movies",
    tags=["Movies"],
)


# --------------------------------------------------
# 2. DATABASE DEPENDENCY
# --------------------------------------------------

def get_movie_service(
    db: Session = Depends(get_db),
):
    """
    Create a MovieService for each request.

    The database session is automatically
    closed by get_db().
    """

    return MovieService(db)


MovieServiceDependency = Annotated[
    MovieService,
    Depends(get_movie_service),
]


# --------------------------------------------------
# 3. MOVIE COUNT RESPONSE
# --------------------------------------------------

class MovieCountResponse(BaseModel):
    """
    Response containing the total
    number of movies in SQLite.
    """

    total_movies: int = Field(
        ...,
        ge=0,
    )


# --------------------------------------------------
# 4. LIST MOVIES
# --------------------------------------------------

@router.get(
    "",
    response_model=MovieListResponse,
    summary="List movies",
    description=(
        "Retrieve movies from the TMDB catalog "
        "using pagination."
    ),
)
def list_movies(
    service: MovieServiceDependency,
    skip: Annotated[
        int,
        Query(
            ge=0,
            description="Number of movies to skip",
        ),
    ] = 0,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Maximum movies to return",
        ),
    ] = 20,
):
    """
    Return a paginated list of movies.
    """

    movies = service.list_movies(
        skip=skip,
        limit=limit,
    )

    total = service.count_movies()

    return MovieListResponse(
        movies=movies,
        total=total,
        skip=skip,
        limit=limit,
    )


# --------------------------------------------------
# 5. SEARCH MOVIES
# --------------------------------------------------

@router.get(
    "/search",
    response_model=list[MovieResponse],
    summary="Search movies",
    description=(
        "Search movies using a partial "
        "case-insensitive title match."
    ),
)
def search_movies(
    service: MovieServiceDependency,
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=100,
            description="Movie title search keyword",
        ),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Maximum search results",
        ),
    ] = 20,
):
    """
    Search for movies by title.
    """

    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search keyword cannot be empty.",
        )

    return service.search_movies(
        keyword=q,
        limit=limit,
    )


# --------------------------------------------------
# 6. COUNT MOVIES
# --------------------------------------------------

@router.get(
    "/count",
    response_model=MovieCountResponse,
    summary="Count movies",
    description=(
        "Return the number of movies "
        "currently stored in SQLite."
    ),
)
def count_movies(
    service: MovieServiceDependency,
):
    """
    Return the total number of movies.
    """

    return MovieCountResponse(
        total_movies=service.count_movies()
    )


# --------------------------------------------------
# 7. GET MOVIE BY ID
# --------------------------------------------------

@router.get(
    "/{movie_id}",
    response_model=MovieResponse,
    summary="Get movie details",
    description=(
        "Retrieve movie information using "
        "its TMDB movie ID."
    ),
    responses={
        404: {
            "description": "Movie not found",
        },
    },
)
def get_movie(
    service: MovieServiceDependency,
    movie_id: Annotated[
        int,
        Path(
            gt=0,
            description="TMDB movie ID",
            examples=[19995],
        ),
    ],
):
    """
    Retrieve one movie by its ID.

    Return HTTP 404 if the movie
    does not exist.
    """

    movie = service.get_movie(
        movie_id=movie_id
    )

    if movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Movie with ID {movie_id} "
                "was not found."
            ),
        )

    return movie