from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router as photo_router
from api.schemas import HealthResponse
from core.errors import AppError
from core.settings import settings
from db.database import Base, engine, ensure_pgvector_extension, ensure_runtime_schema
from db import orm_models as db_models  # noqa: F401
from services.storage_service import ensure_bucket


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_pgvector_extension()  # must run before create_all — see db/database.py
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()
    ensure_bucket()  # same idea, for object storage — see services/storage_service.py
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


# Single place that turns any AppError into an HTTP response. Every route
# just calls service functions directly and lets exceptions propagate — no
# per-route try/except, no per-error-type HTTPException construction. Add a
# new AppError subclass anywhere in services/ and it's handled automatically
# with zero changes here.
@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc), "error_code": exc.error_code},
    )


app.include_router(photo_router, tags=["jobs"])
