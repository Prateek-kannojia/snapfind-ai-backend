"""RQ worker entry point. Claims queued jobs from Redis and runs face matching.

Runs as a single worker by default (`python worker.py`), matching the
original simple setup. Set WORKER_COUNT to run more than one worker process
against the same queue for horizontal scaling — see "Why multiple worker
processes?" in README.md. This is one file instead of two (there used to be
a separate run_workers.py launcher) because a launcher script is really just
this same script re-invoking itself; folding it in avoids the "why are there
two worker files" question entirely.

    python worker.py                       # 1 worker (default, unchanged)
    WORKER_COUNT=4 python worker.py        # 4 worker processes
"""
from __future__ import annotations

import os
import subprocess
import sys

from redis import Redis
from rq import Queue, SimpleWorker, Worker

from core.settings import settings


def _run_single_worker() -> None:
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue(settings.rq_queue_name, connection=redis_conn)
    worker_class = SimpleWorker if settings.rq_worker_class == "simple" else Worker
    worker = worker_class([queue], connection=redis_conn)
    worker.work()


def _run_multiple_workers(count: int) -> None:
    """Launches `count` copies of this same script as separate OS processes.

    Each process runs one worker against the same Redis queue. RQ's atomic
    dequeue guarantees two workers can never claim the same job twice, so
    this is safe — it is horizontal scaling, not a change to how any single
    job is processed.
    """
    print(f"Starting {count} worker process(es) against queue '{settings.rq_queue_name}'...")
    child_env = os.environ.copy()
    child_env["WORKER_COUNT"] = "1"  # each child runs as a plain single worker

    processes = [subprocess.Popen([sys.executable, __file__], env=child_env) for _ in range(count)]
    for i, proc in enumerate(processes, start=1):
        print(f"  worker {i}/{count} started (pid {proc.pid})")

    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nStopping all workers...")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            proc.wait()


def main() -> None:
    worker_count = int(os.getenv("WORKER_COUNT", "1"))
    if worker_count <= 1:
        _run_single_worker()
    else:
        _run_multiple_workers(worker_count)


if __name__ == "__main__":
    main()
