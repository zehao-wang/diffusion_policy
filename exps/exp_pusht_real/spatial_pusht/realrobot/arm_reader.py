"""Background poller for `RealRobotArmClient`.

Same pattern as minye's `robot/http_arm_reader.py`: a daemon thread calls
`client.read()` at a fixed rate and caches the latest `ArmReading`, so
the GUI / main loop can pull state without blocking on HTTP every tick.

While the upstream `pusht_service` arm is disconnected the read calls
raise; the reader stores the last error string and keeps trying. Once
the GUI fires `/arm/connect` through the coordinator the next poll
succeeds and readings start flowing again.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .arm_client import ArmReading, RealRobotArmClient


class ArmReader:
    def __init__(
        self,
        client: RealRobotArmClient,
        *,
        poll_hz: float = 30.0,
    ):
        self._client = client
        self._poll_dt = 1.0 / max(float(poll_hz), 1.0)

        self._lock = threading.Lock()
        self._reading: Optional[ArmReading] = None
        self._reading_ts: float = 0.0
        self._last_error: str = ""

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get_reading(self) -> tuple[Optional[ArmReading], float, str]:
        """Return (reading, wall-clock ts, last_error). reading is None
        until at least one successful poll has completed."""
        with self._lock:
            return self._reading, self._reading_ts, self._last_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                reading = self._client.read()
                with self._lock:
                    self._reading = reading
                    self._reading_ts = time.time()
                    self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            remaining = self._poll_dt - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)
