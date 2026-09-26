
"""
app/main.py

Main entry point for the Movie Recommendation System.

Responsibilities:
1. Create the FastAPI application.
2. Initialize database tables.
3. Load the trained recommendation model.
4. Register all API routers.
5. Configure CORS.
6. Provide health and root endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEBUG,
)

from app.db.database import init_db

# Import ORM models before calling init_db()
# so SQLAlchemy registers both tables.
from app.models.movie import Movie
from app.models.interaction import Interaction

from app.api.routes.movies import (
    router as movies_router,
)

from app.api.routes.recommendations import (
    router as recommendations_router,
    get_recommender,
)

from app.api.routes.feedback import (
    router as feedback_router,
)


# --------------------------------------------------
# 1. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format=(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Execute startup and shutdown operations.
    """

    logger.info(
        "Starting %s version %s",
        APP_NAME,
        APP_VERSION,
    )

    # Initialize database tables.
    init_db()

    logger.info(
        "Database initialized successfully."
    )

    # Load the trained recommendation model.
    try:
        recommender = get_recommender()

        logger.info(
            "Recommendation model loaded. "
            "Movies available: %s",
            len(recommender.movie_ids),
        )

        app.state.model_ready = True

    except (FileNotFoundError, ValueError):
        logger.exception(
            "Could not load the recommendation model."
        )

        # Movie browsing and feedback can still work
        # even if model artifacts are unavailable.
        app.state.model_ready = False

    yield

    logger.info(
        "Shutting down %s",
        APP_NAME,
    )


# --------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Content-based Movie Recommendation "
        "System using TMDB movie data, "
        "TF-IDF and cosine similarity."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------
# 4. CONFIGURE CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# 5. REGISTER ROUTERS
# --------------------------------------------------

app.include_router(
    movies_router,
)

app.include_router(
    recommendations_router,
)

app.include_router(
    feedback_router,
)


# --------------------------------------------------
# 6. ROOT ENDPOINT
# --------------------------------------------------

@app.get(
    "/",
    tags=["System"],
    summary="API information",
)
def root():
    """
    Return basic application information.
    """

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "documentation": "/docs",
    }


# --------------------------------------------------
# 7. HEALTH CHECK
# --------------------------------------------------

@app.get(
    "/health",
    tags=["System"],
    summary="Check application health",
)
def health_check():
    """
    Report whether the recommendation
    model is available.
    """

    from app.api.routes.recommendations import (
        get_recommender,
    )

    return {
        "status": "healthy",
        "model_loaded": (
            get_recommender.cache_info().currsize > 0
        ),
    }


# --------------------------------------------------
# 8. RUN APPLICATION DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=DEBUG,
    )