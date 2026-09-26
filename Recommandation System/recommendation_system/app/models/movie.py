


"""
app/models/movie.py

SQLAlchemy ORM model for storing movie information.

Each Movie object represents one row in
the movies table.

Fields:
    id       : Unique movie identifier
    title    : Movie title
    overview : Movie description
    genres   : Comma-separated movie genres
"""

from sqlalchemy import Column, Integer, String, Text

from app.db.database import Base


class Movie(Base):
    """
    Represents a movie stored in the database.
    """

    __tablename__ = "movies"

    # Unique movie ID
    # We use the ID from our movie dataset.
    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=False,
    )

    # Movie title
    title = Column(
        String(255),
        nullable=False,
        index=True,
    )

    # Movie description
    overview = Column(
        Text,
        nullable=True,
    )

    # Comma-separated genres
    genres = Column(
        Text,
        nullable=True,
    )

    def __repr__(self):
        return (
            f"Movie(id={self.id}, "
            f"title={self.title!r})"
        )