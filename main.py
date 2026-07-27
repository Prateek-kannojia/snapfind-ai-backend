from fastapi import FastAPI

from api.routes import router as photo_router
from api.schemas import HealthResponse
from db.database import Base, engine, ensure_runtime_schema
from db import orm_models as db_models  # noqa: F401

app = FastAPI(
    title="Event Photo Finder API",
    description="Upload event photos and a selfie to create a matching job.",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema()


@app.get("/", response_model=HealthResponse)
def home() -> HealthResponse:
    return HealthResponse(message="Event photo finder API is working")


app.include_router(photo_router, tags=["jobs"])
