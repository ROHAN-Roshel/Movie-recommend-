from app.db.database import (
    SessionLocal,
    init_db,
)

from app.models.movie import Movie
from app.models.interaction import Interaction

from app.db.repository import (
    MovieRepository,
    InteractionRepository,
)

# Create tables
init_db()

# Open a database session
db = SessionLocal()

try:
    movies = MovieRepository(db)
    interactions = InteractionRepository(db)

    # Insert a sample movie if it doesn't exist
    movie = movies.get_by_id(1)

    if movie is None:
        movie = movies.create(
            movie_id=1,
            title="Interstellar",
            overview="Explorers travel through a wormhole.",
            genres="Science Fiction,Adventure",
        )

    # Retrieve the movie
    print(movies.get_by_id(1))

    # Record a user rating
    rating = interactions.create(
        user_id=1,
        movie_id=1,
        interaction_type="rating",
        rating=4.5,
    )

    print("Rating saved:", rating.id)

    # Retrieve the user's ratings
    print(interactions.get_user_ratings(1))

finally:
    db.close()