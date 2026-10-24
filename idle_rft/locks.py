"""Atomic directory ownership for filesystems without advisory file locking."""
from contextlib import contextmanager
from pathlib import Path
import time


@contextmanager
def directory_lock(path, wait_seconds=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise BlockingIOError("Directory already owned: " + str(path))
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    try:
        yield
    finally:
        path.rmdir()
