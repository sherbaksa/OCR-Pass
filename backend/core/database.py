from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from backend.core.settings import settings

# Создаем Base для моделей
Base = declarative_base()

# Преобразуем URL для async драйвера
# postgresql:// -> postgresql+asyncpg://
DATABASE_URL = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

# Создаем async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=settings.debug,  # Логирование SQL запросов в debug режиме
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Создаем фабрику сессий
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncSession:
    """
    Dependency для получения сессии базы данных
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Инициализация базы данных (создание таблиц)
    Используется только для разработки, в продакшене - миграции
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
