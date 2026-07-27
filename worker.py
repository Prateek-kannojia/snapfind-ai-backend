"""RQ worker entry point. Claims queued jobs from Redis and runs face matching."""
from __future__ import annotations

from redis import Redis
from rq import Queue, SimpleWorker, Worker

from core.settings import settings


def main() -> None:
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue(settings.rq_queue_name, connection=redis_conn)
    worker_class = SimpleWorker if settings.rq_worker_class == "simple" else Worker
    worker = worker_class([queue], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
