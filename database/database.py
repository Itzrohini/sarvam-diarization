# database/database.py
# ============================================================
# WHY THIS FILE?
# This creates the database connection and session.
# A "session" is like a conversation with the database —
# you open it, do your work, then close it.
# ============================================================

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base
from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# create_engine creates the connection pool to PostgreSQL
# pool_pre_ping=True means it tests the connection before using it
# — prevents "connection lost" errors after idle time
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=settings.debug,  # echo=True prints all SQL queries (useful for debugging)
)

# SessionLocal is a factory that creates new database sessions
# autocommit=False means changes aren't saved until you call commit()
# autoflush=False means changes aren't sent to DB until you commit
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def create_tables() -> None:
    """
    Create all database tables if they don't exist.
    Call this once at startup.
    
    SQLAlchemy reads all models that inherit from Base
    and creates their tables automatically.
    """
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


def get_db() -> Session:
    """
    FastAPI dependency that provides a database session.
    
    WHY yield INSTEAD OF return?
    yield pauses the function after giving the session.
    FastAPI runs the endpoint, then comes back here to run
    the finally block — which closes the session cleanly.
    This guarantees the session is ALWAYS closed even if
    the endpoint crashes halfway through.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()