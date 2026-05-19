"""Shared-memory frame transport for external camera capture services."""

from __future__ import annotations

import json
import os
import socket
import struct
import time
from multiprocessing import shared_memory

import numpy as np

DEFAULT_POINTGREY_SOCKET_PATH = "/tmp/robodata_pointgrey.sock"
DEFAULT_POINTGREY_SHM_PREFIX = "robodata_pointgrey"

MAGIC = b"RDCAM01\x00"
VERSION = 1
STATUS_INIT = 0
STATUS_RUNNING = 1
STATUS_ERROR = 2
STATUS_STOPPED = 3
HEADER_STRUCT = struct.Struct("<8s6I2Q2IQ")
HEADER_READ_RETRIES = 8
HEADER_READ_RETRY_DELAY_S = 0.0005


def create_shared_memory(name: str, size: int) -> shared_memory.SharedMemory:
    """Create a fresh shared-memory block, replacing any stale segment."""
    try:
        existing = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        existing.close()
        try:
            existing.unlink()
        except FileNotFoundError:
            pass
    shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    _chmod_shared_memory(name, 0o666)
    return shm


def unlink_shared_memory(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    finally:
        shm.close()


def _unregister_shm_from_resource_tracker(name: str) -> None:
    """Tell Python's resource_tracker to stop tracking this shm name.

    Without this, every reader process that attaches will cause an
    `shm_unlink` on the segment when the reader exits, even though only the
    creator should ever unlink. See https://bugs.python.org/issue38119.
    """
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(f"/{name.lstrip('/')}", "shared_memory")
    except Exception:
        pass


def _chmod_shared_memory(name: str, mode: int) -> None:
    """Best-effort permission fixup for POSIX shared-memory files on Linux."""
    shm_basename = name.lstrip("/")
    candidate_paths = [
        f"/dev/shm/{shm_basename}",
        f"/private/var/run/shm/{shm_basename}",
    ]
    for path in candidate_paths:
        try:
            os.chmod(path, mode)
            return
        except FileNotFoundError:
            continue
        except PermissionError:
            continue
        except OSError:
            continue


def recv_json_message(conn: socket.socket) -> dict:
    data = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if not data:
        return {}
    line = bytes(data).split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def send_json_message(conn: socket.socket, payload: dict) -> None:
    conn.sendall(json.dumps(payload).encode("utf-8") + b"\n")


def send_json_request(socket_path: str, payload: dict, timeout_s: float = 1.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        send_json_message(sock, payload)
        return recv_json_message(sock)


class SharedFrameRingWriter:
    """Publishes the latest RGB frame into a small shared-memory ring buffer."""

    def __init__(
        self,
        *,
        shm_prefix: str,
        width: int,
        height: int,
        channels: int = 3,
        slot_count: int = 3,
    ):
        if slot_count < 2:
            raise ValueError("slot_count must be >= 2")

        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.slot_count = int(slot_count)
        self.frame_bytes = self.width * self.height * self.channels
        self.meta_shm_name = f"{shm_prefix}_meta"
        self.color_shm_name = f"{shm_prefix}_color"
        self._publish_seq = 0
        self._frame_id = 0
        self._latest_slot = 0
        self._status = STATUS_INIT
        self._timestamp_ns = 0

        self._meta_shm = create_shared_memory(self.meta_shm_name, HEADER_STRUCT.size)
        self._color_shm = create_shared_memory(
            self.color_shm_name,
            self.slot_count * self.frame_bytes,
        )
        self._write_header()

    def set_status(self, status: int, timestamp_ns: int | None = None) -> None:
        self._status = int(status)
        if timestamp_ns is not None:
            self._timestamp_ns = int(timestamp_ns)
        self._write_header()

    def publish(self, frame: np.ndarray, timestamp_ns: int) -> None:
        expected_shape = (self.height, self.width, self.channels)
        if frame.shape != expected_shape:
            raise ValueError(f"Expected frame shape {expected_shape}, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame.dtype}")

        next_slot = (self._latest_slot + 1) % self.slot_count
        self._publish_seq += 1
        self._write_header()

        frame_view = np.ndarray(
            expected_shape,
            dtype=np.uint8,
            buffer=self._color_shm.buf,
            offset=next_slot * self.frame_bytes,
        )
        frame_view[...] = frame

        self._latest_slot = next_slot
        self._frame_id += 1
        self._timestamp_ns = int(timestamp_ns)
        self._status = STATUS_RUNNING
        self._publish_seq += 1
        self._write_header()

    def descriptor(self) -> dict:
        return {
            "version": VERSION,
            "meta_shm_name": self.meta_shm_name,
            "color_shm_name": self.color_shm_name,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "slot_count": self.slot_count,
            "frame_bytes": self.frame_bytes,
        }

    def close(self) -> None:
        self.set_status(STATUS_STOPPED, timestamp_ns=time.time_ns())
        self._meta_shm.close()
        self._color_shm.close()
        try:
            self._meta_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            self._color_shm.unlink()
        except FileNotFoundError:
            pass

    def _write_header(self) -> None:
        HEADER_STRUCT.pack_into(
            self._meta_shm.buf,
            0,
            MAGIC,
            VERSION,
            self.slot_count,
            self.width,
            self.height,
            self.channels,
            self.frame_bytes,
            self._publish_seq,
            self._frame_id,
            self._latest_slot,
            self._status,
            self._timestamp_ns,
        )


class SharedFrameRingReader:
    """Reads the latest published RGB frame from shared memory."""

    def __init__(self, descriptor: dict):
        self.width = int(descriptor["width"])
        self.height = int(descriptor["height"])
        self.channels = int(descriptor["channels"])
        self.slot_count = int(descriptor["slot_count"])
        self.frame_bytes = int(descriptor["frame_bytes"])
        self.meta_shm_name = descriptor["meta_shm_name"]
        self.color_shm_name = descriptor["color_shm_name"]
        self._meta_shm = shared_memory.SharedMemory(name=self.meta_shm_name, create=False)
        self._color_shm = shared_memory.SharedMemory(name=self.color_shm_name, create=False)
        # Python 3.9 multiprocessing.shared_memory has a long-standing bug
        # (https://bugs.python.org/issue38119) where the resource_tracker in
        # *any* attacher will call shm_unlink on the segment when the attacher
        # exits, even though only the creator should. Unregister so a reader
        # crash doesn't poison the producer's shm name.
        _unregister_shm_from_resource_tracker(self._meta_shm.name)
        _unregister_shm_from_resource_tracker(self._color_shm.name)
        self._buffers = [
            np.zeros((self.height, self.width, self.channels), dtype=np.uint8),
            np.zeros((self.height, self.width, self.channels), dtype=np.uint8),
        ]
        self._last_frame = self._buffers[0]
        self._next_buffer_idx = 1
        self._last_frame_id = 0
        self._last_timestamp = 0.0
        self._expected_header_fields = {
            "version": VERSION,
            "slot_count": self.slot_count,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "frame_bytes": self.frame_bytes,
        }

    def read_latest(self, retries: int = 5) -> tuple[np.ndarray, float, int]:
        for _ in range(max(1, retries)):
            header1 = self._read_header()
            if header1["publish_seq"] % 2 == 1:
                time.sleep(0.0005)
                continue
            if header1["frame_id"] == 0:
                return self._last_frame.copy(), 0.0, 0

            slot = header1["latest_slot"]
            frame_view = np.ndarray(
                (self.height, self.width, self.channels),
                dtype=np.uint8,
                buffer=self._color_shm.buf,
                offset=slot * self.frame_bytes,
            )
            snapshot = self._buffers[self._next_buffer_idx]
            np.copyto(snapshot, frame_view)
            header2 = self._read_header()
            if header1 == header2 and header2["publish_seq"] % 2 == 0:
                self._last_frame = snapshot
                self._next_buffer_idx = 1 - self._next_buffer_idx
                self._last_frame_id = header2["frame_id"]
                self._last_timestamp = header2["timestamp_ns"] / 1_000_000_000.0
                return self._last_frame, self._last_timestamp, self._last_frame_id

        return self._last_frame, self._last_timestamp, self._last_frame_id

    def wait_for_frame(self, timeout_s: float = 5.0) -> tuple[np.ndarray, float, int]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frame, timestamp, frame_id = self.read_latest()
            if frame_id > 0:
                return frame, timestamp, frame_id
            time.sleep(0.01)
        return self.read_latest()

    def close(self) -> None:
        self._meta_shm.close()
        self._color_shm.close()

    def _read_header(self) -> dict:
        last_error = "invalid magic"
        for attempt in range(HEADER_READ_RETRIES):
            values = HEADER_STRUCT.unpack_from(self._meta_shm.buf, 0)
            if values[0] != MAGIC:
                last_error = f"invalid magic {values[0]!r}"
            else:
                header = {
                    "version": values[1],
                    "slot_count": values[2],
                    "width": values[3],
                    "height": values[4],
                    "channels": values[5],
                    "frame_bytes": values[6],
                    "publish_seq": values[7],
                    "frame_id": values[8],
                    "latest_slot": values[9],
                    "status": values[10],
                    "timestamp_ns": values[11],
                }
                mismatch = next(
                    (
                        f"{field}={header[field]} expected {expected}"
                        for field, expected in self._expected_header_fields.items()
                        if header[field] != expected
                    ),
                    None,
                )
                if mismatch is None:
                    return header
                last_error = mismatch

            if attempt + 1 < HEADER_READ_RETRIES:
                time.sleep(HEADER_READ_RETRY_DELAY_S)

        raise RuntimeError(
            "Invalid camera shared-memory header "
            f"for {self.meta_shm_name}: {last_error}"
        )
