"""In-process job state, resumable SSE events, and task lifecycle ownership."""

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
    session_id: str = ""
    generation: int = 0
    cursor: int = 0
    session: dict | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


def sse_format(event: dict) -> str:
    """Render an event as SSE, retaining the old no-cursor frame contract."""
    data = f"data: {json.dumps(event)}\n\n"
    return f"id: {event['id']}\n{data}" if "id" in event else data


class JobRegistry:
    """Own job history, exact-once subscribers, and in-process task handles."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop | None] = {}
        self._subscribers: dict[str, list[asyncio.Queue[dict | None]]] = {}
        self._generations: dict[str, int] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._lock = threading.RLock()
        self._default_loop = loop
        self._owner_thread = threading.get_ident()

    def create(self, session_id: str = "") -> Job:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._default_loop
        with self._lock:
            generation = self._generations.get(session_id, 0)
            job = Job(job_id=uuid.uuid4().hex, session_id=session_id, generation=generation)
            self._jobs[job.job_id] = job
            self._loops[job.job_id] = loop
            self._subscribers[job.job_id] = []
        return job

    def session_lock(self, session_id: str) -> asyncio.Lock:
        """The short commit lock shared by every mutation for one session."""
        with self._lock:
            return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def claim_session(self, session_id: str) -> int:
        """Invalidate older work and return the generation owned by this action."""
        async with self.session_lock(session_id):
            self.invalidate_session(session_id)
            with self._lock:
                return self._generations[session_id]

    def generation_is_current(self, session_id: str, generation: int) -> bool:
        with self._lock:
            return generation == self._generations.get(session_id, 0)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _enqueue(self, job: Job, item: dict | None, queues: list[asyncio.Queue]) -> None:
        loop = self._loops[job.job_id]
        if loop is not None and not loop.is_closed():
            for queue in queues:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            return
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("A worker-thread job needs an owning event loop.")
        for queue in queues:
            queue.put_nowait(item)

    def _publish_locked(self, job: Job, event: dict) -> dict:
        job.cursor += 1
        recorded = {**event, "id": job.cursor}
        job.events.append(recorded)
        self._enqueue(job, recorded, [job.queue, *self._subscribers[job.job_id]])
        return recorded

    def publish(self, job: Job, event: dict) -> None:
        """Append a cursor-bearing event and safely notify all live consumers."""
        with self._lock:
            # A cancelled asyncio task can leave its to_thread worker running.
            # Its late callback must not put progress after the terminal event.
            if job.status != "running":
                return
            self._publish_locked(job, event)

    def finish(self, job: Job, status: str, error: str = "", session: dict | None = None) -> None:
        """Atomically make a job terminal and pair status with its terminal event."""
        with self._lock:
            if job.status != "running":
                return
            job.status = status
            job.error = error
            job.session = dict(session) if session is not None else job.session
            event = {"type": status}
            if error:
                event["error"] = error
            self._publish_locked(job, event)
            self._enqueue(job, None, [job.queue, *self._subscribers[job.job_id]])

    def snapshot(self, job: Job, after: int = 0) -> dict:
        """Read status, error, cursor, and unseen events under one lock."""
        with self._lock:
            return {
                "job_id": job.job_id,
                "status": job.status,
                "error": job.error,
                "cursor": job.cursor,
                "events": [dict(event) for event in job.events if event["id"] > after],
                "session": dict(job.session) if job.session is not None else None,
            }

    def subscribe(self, job: Job, after: int = 0) -> tuple[list[dict], asyncio.Queue[dict | None], bool]:
        """Atomically replay only unseen history and subscribe for later events."""
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        with self._lock:
            history = [dict(event) for event in job.events if event["id"] > after]
            terminal = job.status != "running"
            if not terminal:
                self._subscribers[job.job_id].append(queue)
        return history, queue, terminal

    def unsubscribe(self, job: Job, queue: asyncio.Queue[dict | None]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job.job_id, [])
            if queue in subscribers:
                subscribers.remove(queue)

    def register_task(self, job: Job, task: asyncio.Task) -> None:
        """Retain a handle so invalidation and application shutdown can drain it."""
        with self._lock:
            job.task = task

        def consume(done: asyncio.Task) -> None:
            if not done.cancelled():
                done.exception()
            with self._lock:
                if job.task is done:
                    job.task = None

        task.add_done_callback(consume)

    def is_current(self, job: Job) -> bool:
        with self._lock:
            return job.status == "running" and job.generation == self._generations.get(job.session_id, 0)

    def invalidate_session(self, session_id: str) -> None:
        """Cancel every current session job; thread work later fails ownership checks."""
        with self._lock:
            self._generations[session_id] = self._generations.get(session_id, 0) + 1
            jobs = [job for job in self._jobs.values() if job.session_id == session_id and job.status == "running"]
            for job in jobs:
                self.finish(job, "cancelled", "Superseded by a newer session action.")
            tasks = [(job.task, self._loops[job.job_id]) for job in jobs if job.task is not None]
        for task, loop in tasks:
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

    async def shutdown(self) -> None:
        """Cancel and await active jobs during FastAPI lifespan shutdown."""
        with self._lock:
            session_ids = {job.session_id for job in self._jobs.values() if job.status == "running"}
        for session_id in session_ids:
            self.invalidate_session(session_id)
        with self._lock:
            tasks = [job.task for job in self._jobs.values() if job.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


registry = JobRegistry()
