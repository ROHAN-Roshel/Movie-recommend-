



"""
app/models/interaction.py

SQLAlchemy model for recording user interactions
with movies.

Supported interactions:
    - view
    - like
    - dislike
    - rating
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    ForeignKey,
    CheckConstraint,
    Index,
)

from sqlalchemy.orm import relationship

from app.db.database import Base


class Interaction(Base):
    """
    Represents an interaction between a user
    and a movie.
    """

    __tablename__ = "interactions"

    # Unique interaction identifier
    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True,
    )

    # User identifier
    user_id = Column(
        Integer,
        nullable=False,
        index=True,
    )

    # Movie identifier
    movie_id = Column(
        Integer,
        ForeignKey("movies.id"),
        nullable=False,
        index=True,
    )

    # Type of interaction
    interaction_type = Column(
        String(20),
        nullable=False,
    )

    # Optional rating from 1 to 5
    rating = Column(
        Float,
        nullable=True,
    )

    # Time of interaction (UTC)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship with Movie model
    movie = relationship("Movie")

    # Database constraints and indexes
    __table_args__ = (
        CheckConstraint(
            "interaction_type IN "
            "('view', 'like', 'dislike', 'rating')",
            name="valid_interaction_type",
        ),
        CheckConstraint(
            "rating IS NULL OR "
            "(rating >= 1 AND rating <= 5)",
            name="valid_rating_range",
        ),
        CheckConstraint(
            "(interaction_type = 'rating' "
            "AND rating IS NOT NULL) OR "
            "(interaction_type != 'rating' "
            "AND rating IS NULL)",
            name="rating_matches_type",
        ),
        Index(
            "ix_interactions_user_movie",
            "user_id",
            "movie_id",
        ),
    )

    def __repr__(self):
        return (
            f"Interaction("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"movie_id={self.movie_id}, "
            f"type={self.interaction_type!r}"
            f")"
        )