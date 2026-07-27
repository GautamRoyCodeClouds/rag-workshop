"""Thread-safe background-job progress and SSE subscriptions."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    job_id: str
    queue: asyncio.Queue[dict | None] = field(default_factory=asyncio.Queue)
    status: str = "running"
    events: list[dict] = field(default_factory=list)
    error: str = ""


def sse_format(event: dict) -> str:
    """Render one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


class JobRegistry:
    """Track jobs, their polling history, and exact-once SSE subscriptions.

    Pipeline functions run in worker threads.  Queue operations therefore always
    hop back to the job's owning event loop; neither asyncio.Queue nor a list
    append is used as a cross-thread synchronisation primitive.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop | None] = {}
        self._subscribers: dict[str, list[asyncio.Queue[dict | None]]] = {}
        self._lock = threading.RLock()
        self._default_loop = loop
        self._owner_thread = threading.get_ident()

    def create(self) -> Job:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._default_loop
        job = Job(job_id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.job_id] = job
            self._loops[job.job_id] = loop
            self._subscribers[job.job_id] = []
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _enqueue(self, job: Job, item: dict | None, queues: list[asyncio.Queue]) -> None:
        """Schedule a queue write on the owning loop, never from a worker."""
        loop = self._loops[job.job_id]
        if loop is not None and not loop.is_closed():
            for queue in queues:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("A worker-thread job needs an owning event loop.")
        for queue in queues:
            queue.put_nowait(item)

    def _publish_locked(self, job: Job, event: dict) -> None:
        job.events.append(event)
        self._enqueue(job, event, [job.queue, *self._subscribers[job.job_id]])

    def publish(self, job: Job, event: dict) -> None:
        """Atomically retain and publish a progress event from any thread."""
        with self._lock:
            self._publish_locked(job, event)

    def finish(self, job: Job, status: str, error: str = "") -> None:
        """Mark a job terminal, publish its terminal event, and close streams."""
        with self._lock:
            job.status = status
            job.error = error
            event = {"type": status}
            if error:
                event["error"] = error
            self._publish_locked(job, event)
            self._enqueue(job, None, [job.queue, *self._subscribers[job.job_id]])

    def subscribe(self, job: Job) -> tuple[list[dict], asyncio.Queue[dict | None], bool]:
        """Atomically snapshot history and subscribe to future events.

        The returned replay and queue form a hand-off boundary: events before
        the lock are in replay; every later event goes only to this subscriber.
        This avoids duplicate late-SSE frames from the job's polling queue.
        """
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        with self._lock:
            history = list(job.events)
            terminal = job.status != "running"
            if not terminal:
                self._subscribers[job.job_id].append(queue)
        return history, queue, terminal

    def unsubscribe(self, job: Job, queue: asyncio.Queue[dict | None]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job.job_id, [])
            if queue in subscribers:
                subscribers.remove(queue)


registry = JobRegistry()
