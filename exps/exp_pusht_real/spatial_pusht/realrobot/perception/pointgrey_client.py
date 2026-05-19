"""Remote PointGrey camera client backed by an external capture service."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .camera_ipc import (
    DEFAULT_POINTGREY_SHM_PREFIX,
    DEFAULT_POINTGREY_SOCKET_PATH,
    SharedFrameRingReader,
    send_json_request,
)
from .pointgrey_calibration import (
    load_pointgrey_calibration,
    merge_pointgrey_camera_info,
)


class PointGreyCamera:
    """Reads the latest PointGrey RGB frame from a dedicated capture service."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        streams: str = "rgb",
        device_serial: str | None = None,
        *,
        socket_path: str = DEFAULT_POINTGREY_SOCKET_PATH,
        shm_prefix: str = DEFAULT_POINTGREY_SHM_PREFIX,
        service_python: str | None = None,
        service_script: str | None = None,
        calibration_path: str | None = None,
        startup_timeout: float = 10.0,
    ):
        if streams != "rgb":
            raise ValueError("PointGrey camera module currently supports RGB capture only")

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.streams = streams
        self.device_serial = device_serial
        self.socket_path = socket_path
        self.shm_prefix = shm_prefix
        self.service_python = service_python
        self.service_script = service_script or str(
            Path(__file__).resolve().parent.parent / "pointgrey_capture_service.py"
        )
        self.calibration_path = calibration_path
        self.startup_timeout = float(startup_timeout)

        self._reader = None
        self._service_proc = None
        self._owns_service = False
        self._color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_frame = np.zeros((height, width), dtype=np.uint16)
        self._timestamp = 0.0
        self._camera_info = None
        self._last_frame_id = 0

    @staticmethod
    def describe_service(socket_path: str = DEFAULT_POINTGREY_SOCKET_PATH) -> dict | None:
        try:
            response = send_json_request(socket_path, {"cmd": "describe"}, timeout_s=0.5)
        except Exception:
            return None
        if not response.get("ok"):
            return None
        return response

    @staticmethod
    def list_devices(socket_path: str = DEFAULT_POINTGREY_SOCKET_PATH) -> list[dict[str, str]]:
        response = PointGreyCamera.describe_service(socket_path)
        if response is None:
            return []
        camera_info = response.get("camera_info") or {}
        serial = response.get("serial") or camera_info.get("serial") or "service"
        model = camera_info.get("camera_model", "PointGrey Service")
        return [{"name": model, "serial": serial, "vendor": "FLIR"}]

    def start(self):
        """Attach to the remote capture service, optionally auto-starting it."""
        try:
            describe = self.describe_service(self.socket_path)
            if describe is None:
                if self.service_python is None:
                    raise RuntimeError(
                        "PointGrey capture service is not running. "
                        "Start pointgrey_capture_service.py in the PySpin environment "
                        "or pass --pointgrey-python so collect_viser.py can launch it."
                    )
                self._launch_service()
                describe = self._wait_for_service()
            else:
                self._owns_service = False

            transport = describe.get("transport")
            if not isinstance(transport, dict):
                raise RuntimeError("PointGrey service did not return a shared-memory transport descriptor")

            self._reader = SharedFrameRingReader(transport)
            color, timestamp, frame_id = self._reader.wait_for_frame(timeout_s=self.startup_timeout)
            if frame_id <= 0:
                raise RuntimeError("PointGrey capture service is up but no frame has been published yet")
            self._color_frame = color
            self.height, self.width = color.shape[:2]
            self._depth_frame = np.zeros((self.height, self.width), dtype=np.uint16)
            self._timestamp = timestamp
            self._last_frame_id = frame_id
            self._camera_info = self._build_camera_info(describe.get("camera_info"))
            print(
                f"[PointGreyCamera] Connected to service ({self.width}x{self.height} @ {self.fps}fps, "
                f"socket={self.socket_path})"
            )
        except Exception:
            self.stop()
            raise

    def stop(self):
        """Detach from the remote capture service and stop it if we launched it."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

        if self._service_proc is not None:
            try:
                send_json_request(self.socket_path, {"cmd": "shutdown"}, timeout_s=0.5)
            except Exception:
                pass
            try:
                self._service_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._service_proc.terminate()
                try:
                    self._service_proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._service_proc.kill()
            self._service_proc = None
            self._owns_service = False

        self._camera_info = None
        print("[PointGreyCamera] Stopped")

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self._reader is not None:
            color, timestamp, frame_id = self._reader.read_latest()
            if frame_id > 0 and frame_id != self._last_frame_id:
                self._color_frame = color
                if self._depth_frame.shape != color.shape[:2]:
                    self._depth_frame = np.zeros(color.shape[:2], dtype=np.uint16)
                self._timestamp = timestamp
                self._last_frame_id = frame_id
        return self._color_frame.copy(), self._depth_frame.copy(), self._timestamp

    def capture_sync(self, arm_state_provider) -> tuple[np.ndarray, np.ndarray, float, object]:
        """Grab the latest shared-memory frame and then sample arm state."""
        if self._reader is None:
            raise RuntimeError("PointGrey camera not started")

        color, timestamp, frame_id = self._reader.read_latest()
        if frame_id > 0:
            self._color_frame = color
            if self._depth_frame.shape != color.shape[:2]:
                self._depth_frame = np.zeros(color.shape[:2], dtype=np.uint16)
            self._timestamp = timestamp
            self._last_frame_id = frame_id
        arm_state = arm_state_provider()
        return self._color_frame, self._depth_frame, self._timestamp, arm_state

    def get_camera_info(self) -> dict | None:
        if self._camera_info is None:
            return None
        return {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in self._camera_info.items()
        }

    def _launch_service(self) -> None:
        command = [
            self.service_python,
            self.service_script,
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            str(self.fps),
            "--socket-path",
            self.socket_path,
            "--shm-prefix",
            self.shm_prefix,
        ]
        if self.device_serial:
            command.extend(["--serial", self.device_serial])
        self._service_proc = subprocess.Popen(command)
        self._owns_service = True

    def _wait_for_service(self) -> dict:
        deadline = time.time() + self.startup_timeout
        last_error = "service did not respond"
        while time.time() < deadline:
            if self._service_proc is not None and self._service_proc.poll() is not None:
                raise RuntimeError("PointGrey capture service exited before becoming ready")
            describe = self.describe_service(self.socket_path)
            if describe is not None:
                return describe
            time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for PointGrey capture service: {last_error}")

    def _build_camera_info(self, service_camera_info: dict | None) -> dict | None:
        calibration = load_pointgrey_calibration(self.calibration_path)
        if calibration is None:
            return service_camera_info
        try:
            merged = merge_pointgrey_camera_info(
                service_camera_info,
                calibration,
                calibration_path=self.calibration_path,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply PointGrey calibration file {self.calibration_path}: {exc}"
            ) from exc
        print(f"[PointGreyCamera] Applied calibration file: {self.calibration_path}")
        return merged
