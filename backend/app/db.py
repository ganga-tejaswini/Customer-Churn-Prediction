"""
db.py
-----
Async SQLAlchemy engine/session pointed at Neon Postgres.
Every router that touches the DB depends on get_db() below.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Creates tables if they don't exist. Alembic migrations are the
    real source of truth for schema changes — this is a convenience
    for first-time local setup only."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)