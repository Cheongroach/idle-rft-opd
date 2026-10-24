"""Small durable-file utilities; one owner per GPU experiment directory."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager
from .locks import directory_lock


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path):
    return json.loads(Path(path).read_text())


class Window:
    """Monotonic finite window, including model loading and checkpoint time."""
    def __init__(self, seconds, started=None):
        if not 0 < seconds <= 43200:
            raise ValueError("Each allocated window must be in (0, 43200] seconds")
        self.started = time.monotonic() if started is None else started
        self.seconds = seconds
        self.stopped = False

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    @property
    def active(self):
        return not self.stopped and self.elapsed < self.seconds

    def stop(self, *_):
        self.stopped = True


@contextmanager
def gpu_session(root, stage, seconds, started=None):
    """Reserve budget before work. A hard crash charges its full reservation."""
    limits = {"preflight": 4, "generate": 20, "train": 16, "evaluate": 16, "retry": 4}
    if stage not in limits:
        raise ValueError("Unknown budget stage")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if os.environ.get("IDLE_RFT_PARENT_SESSION"):
        parent = json.loads(os.environ["IDLE_RFT_PARENT_SESSION"])
        ledger = read_json(root / "budget.json")
        if (parent["root"] != str(root.resolve()) or parent["stage"] != stage
                or parent["seconds"] != seconds or not any(
                    e["id"] == parent["id"] and e["status"] == "reserved"
                    for e in ledger["sessions"])):
            raise ValueError("Invalid parent budget lease")
        yield Window(max(1, seconds - 60), started=parent["started"])
        return
    with directory_lock(root / ".gpu-owner.directory"):
        ledger_path = root / "budget.json"
        ledger = read_json(ledger_path) if ledger_path.exists() else {"sessions": []}
        stage_used = sum(s["charged_seconds"] for s in ledger["sessions"] if s["stage"] == stage)
        used = sum(s["charged_seconds"] for s in ledger["sessions"])
        if stage_used + seconds > limits[stage] * 3600 or used + seconds > 60 * 3600:
            raise RuntimeError("Reserved cumulative GPU budget would exceed the cap")
        window = Window(seconds, started=started)
        entry = {"id": uuid.uuid4().hex, "stage": stage, "reserved_seconds": seconds,
                 "charged_seconds": seconds, "status": "reserved", "started_unix": time.time()}
        ledger["sessions"].append(entry)
        atomic_json(ledger_path, ledger)
        window.session_id = entry["id"]
        try:
            yield window
        finally:
            entry.update(charged_seconds=max(window.elapsed, 0), status="finished",
                         stopped=window.stopped)
            atomic_json(ledger_path, ledger)
