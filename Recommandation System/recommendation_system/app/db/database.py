


"""
app/db/database.py

Central database configuration for the
Movie Recommendation System.

Responsibilities:
1. Create the SQLAlchemy database engine.
2. Configure the database session factory.
3. Provide a Base class for ORM models.
4. Manage database sessions.
5. Initialize database tables.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    Session,
)

from app.config import DATABASE_URL


# --------------------------------------------------
# 1. DATABASE ENGINE
# --------------------------------------------------

# SQLite requires this setting when the same
# connection is used across different threads.
connect_args = {}

if DATABASE_URL.startswith("sqlite"):
    connect_args = {
        "check_same_thread": False
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)


# --------------------------------------------------
# 2. SESSION FACTORY
# --------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# --------------------------------------------------
# 3. DECLARATIVE BASE
# --------------------------------------------------

# Every database model will inherit from Base.
Base = declarative_base()


# --------------------------------------------------
# 4. DATABASE SESSION DEPENDENCY
# --------------------------------------------------

def get_db():
    """
    Provide a database session.

    Intended for FastAPI dependency injection.

    The session is closed automatically
    when the request finishes.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# --------------------------------------------------
# 5. DATABASE INITIALIZATION
# --------------------------------------------------

def init_db():
    """
    Create all database tables registered
    with SQLAlchemy's Base.

    Import the ORM models before calling
    this function.
    """

    Base.metadata.create_all(bind=engine)