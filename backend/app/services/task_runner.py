"""
Background Task Runner
======================

Simple in-memory task system for long-running operations (model fitting).
Tasks run in a thread via asyncio.to_thread() to avoid blocking the event loop.
"""

import asyncio
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """State of a background task."""
    id: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None


# In-memory task store (fine for single-user app)
_tasks: dict[str, TaskInfo] = {}


async def submit_task(fn, *args, **kwargs) -> str:
    """Submit a function to run in a background thread.

    Args:
        fn: Callable to execute (will run in a thread)
        *args, **kwargs: Arguments to pass to fn

    Returns:
        task_id: Unique identifier for polling status
    """
    task_id = str(uuid.uuid4())
    task = TaskInfo(id=task_id)
    _tasks[task_id] = task

    async def _run():
        task.status = TaskStatus.RUNNING
        try:
            result = await asyncio.to_thread(fn, *args, **kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            task.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task_id} failed: {e}")
        finally:
            task.completed_at = datetime.now()

    asyncio.create_task(_run())
    logger.info(f"Submitted task {task_id}")
    return task_id


def get_task(task_id: str) -> TaskInfo | None:
    """Get task status by ID."""
    return _tasks.get(task_id)


def cleanup_old_tasks(max_age_hours: int = 24) -> int:
    """Remove completed/failed tasks older than max_age_hours."""
    now = datetime.now()
    to_remove = []
    for tid, task in _tasks.items():
        if task.completed_at and (now - task.completed_at).total_seconds() > max_age_hours * 3600:
            to_remove.append(tid)
    for tid in to_remove:
        del _tasks[tid]
    return len(to_remove)
