


from app.db.database import SessionLocal

from app.services.movie_service import (
    MovieService,
)


def main():
    db = SessionLocal()

    try:
        service = MovieService(db)

        # Retrieve one movie.
        movie = service.get_movie(1)

        if movie:
            print("Movie:", movie.title)
            print("Overview:", movie.overview)
            print("Genres:", movie.genres)

        # Search for movies.
        results = service.search_movies("star")

        print("\nSearch results:")

        for movie in results:
            print(movie.id, movie.title)

        # Retrieve multiple movies.
        movies = service.get_movies_by_ids(
            [2, 3, 4]
        )

        print("\nMovies by ID:")

        for movie in movies:
            print(movie.id, movie.title)

        # Count movies.
        print(
            "\nTotal movies:",
            service.count_movies(),
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()