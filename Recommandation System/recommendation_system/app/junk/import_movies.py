
from app.db.database import (
    SessionLocal,
    init_db,
)

# Import models before creating tables.
from app.models.movie import Movie
from app.models.interaction import Interaction

from app.services.movie_service import (
    MovieService,
)


def main():
    init_db()

    db = SessionLocal()

    try:
        service = MovieService(db)

        result = service.import_movies_from_csv()

        print("Import result:", result)

    finally:
        db.close()


if __name__ == "__main__":
    main()