from __future__ import annotations

from redis import Redis
from rq import Queue

from core.settings import settings


def _get_queue() -> Queue:
    redis_conn = Redis.from_url(settings.redis_url)
    return Queue(settings.rq_queue_name, connection=redis_conn)


def enqueue_face_matching_job(job_id: str, threshold: float) -> str:
    queue = _get_queue()
    rq_job = queue.enqueue(
        "services.job_service.run_job_processing",
        job_id,
        threshold,
        job_timeout=settings.rq_job_timeout_seconds,
        result_ttl=3600,
        failure_ttl=86400,
    )
    return rq_job.id
