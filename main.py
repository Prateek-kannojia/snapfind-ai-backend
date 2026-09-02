from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router as photo_router
from api.schemas import HealthResponse
from core.settings import settings
from db.database import Base, engine, ensure_pgvector_extension, ensure_runtime_schema
from db import orm_models as db_models  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_pgvector_extension()  # must run before create_all — see db/database.py
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Upload event photos and a selfie to create a matching job.",
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
def home() -> HealthResponse:
    return HealthResponse(message="Event photo finder API is working")


app.include_router(photo_router, tags=["jobs"])
