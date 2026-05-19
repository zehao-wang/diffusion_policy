#!/usr/bin/env python3
"""Standalone PointGrey capture service for a dedicated PySpin environment."""
#sudo /home/minye/miniconda3/bin/conda run -n pcamera python ./pointgrey_capture_service.py 


from __future__ import annotations

import argparse
import gc
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .camera_ipc import (
    DEFAULT_POINTGREY_SHM_PREFIX,
    DEFAULT_POINTGREY_SOCKET_PATH,
    STATUS_ERROR,
    STATUS_RUNNING,
    SharedFrameRingWriter,
    recv_json_message,
    send_json_message,
)


class PointGreyCaptureDevice:
    """Owns the physical PointGrey / FLIR camera using PySpin."""

    def __init__(self, *, width: int, height: int, fps: int, device_serial: str | None):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device_serial = device_serial
        self._pyspin = None
        self._system = None
        self._cam_list = None
        self._cam = None
        self._camera_info = None
        self._image_processor = None
        self._frame_counter = 0

    def start(self) -> None:
        try:
            import PySpin
        except ImportError as exc:
            raise RuntimeError("PySpin is not installed in this environment") from exc

        self._pyspin = PySpin
        if hasattr(self._pyspin, "ImageProcessor"):
            try:
                self._image_processor = self._pyspin.ImageProcessor()
                if hasattr(self._image_processor, "SetColorProcessing"):
                    self._image_processor.SetColorProcessing(self._pyspin.HQ_LINEAR)
            except Exception:
                self._image_processor = None
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        if self._cam_list.GetSize() == 0:
            raise RuntimeError("No PointGrey / FLIR camera found")

        camera = None
        nodemap = None
        nodemap_tldevice = None
        stream_nodemap = None
        for idx in range(self._cam_list.GetSize()):
            candidate = self._cam_list.GetByIndex(idx)
            serial = self._read_tl_string(candidate, "DeviceSerialNumber")
            if self.device_serial is None or serial == self.device_serial:
                camera = candidate
                break
            del candidate

        if camera is None:
            raise RuntimeError(f"Requested PointGrey serial not found: {self.device_serial}")

        try:
            camera.Init()
            nodemap = camera.GetNodeMap()
            nodemap_tldevice = camera.GetTLDeviceNodeMap()
            stream_nodemap = camera.GetTLStreamNodeMap()

            self._set_enum_value(stream_nodemap, "StreamBufferHandlingMode", "NewestOnly")
            self._set_enum_value(stream_nodemap, "StreamBufferCountMode", "Manual")
            self._set_int_value(stream_nodemap, "StreamBufferCountManual", 1)
            self._set_enum_value(nodemap, "AcquisitionMode", "Continuous")
            self._set_int_value(nodemap, "Width", self.width)
            self._set_int_value(nodemap, "Height", self.height)
            self._set_float_value(nodemap, "AcquisitionFrameRate", float(self.fps))

            camera.BeginAcquisition()
            frame, timestamp_ns = self.capture_frame(camera)
            self.height, self.width = frame.shape[:2]
            self._camera_info = self._build_camera_info(nodemap, nodemap_tldevice)
            self._cam = camera
            print(
                f"[PointGreyService] Camera started ({self.width}x{self.height} @ {self.fps}fps, "
                f"serial={self._camera_info.get('serial', 'unknown')})"
            )
        except Exception:
            stream_nodemap = None
            nodemap_tldevice = None
            nodemap = None
            camera = None
            raise

    def stop(self) -> None:
        camera = self._cam
        self._cam = None
        if camera is not None:
            try:
                if camera.IsStreaming():
                    camera.EndAcquisition()
            except Exception:
                pass
            try:
                if camera.IsInitialized():
                    camera.DeInit()
            except Exception:
                pass
            try:
                del camera
            except Exception:
                pass
            gc.collect()
        if self._cam_list is not None:
            try:
                self._cam_list.Clear()
            except Exception:
                pass
            try:
                del self._cam_list
            except Exception:
                pass
            self._cam_list = None
            gc.collect()
        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception:
                pass
            self._system = None
        self._camera_info = None
        self._image_processor = None

    @property
    def camera_info(self) -> dict | None:
        return None if self._camera_info is None else dict(self._camera_info)

    def capture_frame(self, camera=None) -> tuple[np.ndarray, int]:
        if camera is None:
            camera = self._cam
        if camera is None:
            raise RuntimeError("PointGrey camera is not started")
        image_result = self._get_freshest_image(camera, timeout_ms=1000)
        try:
            if image_result.IsIncomplete():
                raise RuntimeError(
                    f"Incomplete PointGrey frame with status {image_result.GetImageStatus()}"
                )
            timestamp_ns = time.time_ns()
            frame = self._image_to_rgb_array(image_result)
            return frame, timestamp_ns
        finally:
            image_result.Release()

    def _get_freshest_image(self, camera, timeout_ms: int):
        """Block for one image, then drain any queued images and keep only the newest."""
        latest = camera.GetNextImage(timeout_ms)
        drained = 0
        while True:
            try:
                newer = camera.GetNextImage(0)
            except Exception:
                break
            try:
                latest.Release()
            except Exception:
                pass
            latest = newer
            drained += 1

        self._frame_counter += 1
        if drained > 0 and self._frame_counter % 30 == 1:
            print(f"[PointGreyService] Dropped {drained} stale buffered frame(s) before publish")
        return latest

    def _image_to_rgb_array(self, image_result) -> np.ndarray:
        if hasattr(image_result, "Convert"):
            rgb_image = image_result.Convert(self._pyspin.PixelFormat_RGB8, self._pyspin.HQ_LINEAR)
            try:
                return np.array(rgb_image.GetNDArray(), copy=True)
            finally:
                try:
                    del rgb_image
                except Exception:
                    pass

        if self._image_processor is not None:
            converted = self._image_processor.Convert(image_result, self._pyspin.PixelFormat_RGB8)
            try:
                return np.array(converted.GetNDArray(), copy=True)
            finally:
                try:
                    del converted
                except Exception:
                    pass

        frame = np.array(image_result.GetNDArray(), copy=True)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        return frame

    def _build_camera_info(self, nodemap, nodemap_tldevice) -> dict:
        width = self._read_int(nodemap, "Width", self.width)
        height = self._read_int(nodemap, "Height", self.height)
        fps = self._read_float(nodemap, "AcquisitionFrameRate", float(self.fps))
        model = self._read_tl_string_from_map(nodemap_tldevice, "DeviceModelName", "PointGrey")
        serial = self._read_tl_string_from_map(nodemap_tldevice, "DeviceSerialNumber", "")

        fx = self._read_float(nodemap, "ChunkScan3dFocalLength", 0.0)
        fy = fx
        cx = self._read_float(nodemap, "ChunkScan3dPrincipalPointU", width / 2.0)
        cy = self._read_float(nodemap, "ChunkScan3dPrincipalPointV", height / 2.0)

        info = {
            "backend": "pointgrey",
            "camera_model": model,
            "serial": serial,
            "stream": "color",
            "image_rectified": False,
            "resolution": {"width": int(width), "height": int(height)},
            "fps": float(fps),
            "color_space": "RGB",
            "distortion_model": "unknown",
            "distortion": {
                "k1": self._read_float(nodemap, "ChunkLensDistortionValue", 0.0),
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
            },
        }
        if fx > 0.0 and fy > 0.0:
            info["intrinsics"] = {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }
        return info

    @staticmethod
    def _read_tl_string(camera, node_name: str) -> str:
        return PointGreyCaptureDevice._read_tl_string_from_map(
            camera.GetTLDeviceNodeMap(),
            node_name,
            "",
        )

    @staticmethod
    def _read_tl_string_from_map(nodemap, node_name: str, default: str) -> str:
        try:
            import PySpin

            node = PySpin.CStringPtr(nodemap.GetNode(node_name))
            if PySpin.IsReadable(node):
                return node.GetValue()
        except Exception:
            pass
        return default

    def _set_enum_value(self, nodemap, node_name: str, entry_name: str) -> None:
        try:
            node = self._pyspin.CEnumerationPtr(nodemap.GetNode(node_name))
            if not self._pyspin.IsReadable(node) or not self._pyspin.IsWritable(node):
                return
            entry = node.GetEntryByName(entry_name)
            if not self._pyspin.IsReadable(entry):
                return
            node.SetIntValue(entry.GetValue())
        except Exception:
            pass

    def _set_int_value(self, nodemap, node_name: str, value: int) -> None:
        try:
            node = self._pyspin.CIntegerPtr(nodemap.GetNode(node_name))
            if not self._pyspin.IsReadable(node) or not self._pyspin.IsWritable(node):
                return
            target = max(int(node.GetMin()), min(int(value), int(node.GetMax())))
            node.SetValue(target)
        except Exception:
            pass

    def _set_float_value(self, nodemap, node_name: str, value: float) -> None:
        try:
            enable_node = self._pyspin.CBooleanPtr(nodemap.GetNode(f"{node_name}Enable"))
            if self._pyspin.IsWritable(enable_node):
                enable_node.SetValue(True)
        except Exception:
            pass
        try:
            node = self._pyspin.CFloatPtr(nodemap.GetNode(node_name))
            if not self._pyspin.IsReadable(node) or not self._pyspin.IsWritable(node):
                return
            target = max(float(node.GetMin()), min(float(value), float(node.GetMax())))
            node.SetValue(target)
        except Exception:
            pass

    def _read_int(self, nodemap, node_name: str, default: int) -> int:
        try:
            node = self._pyspin.CIntegerPtr(nodemap.GetNode(node_name))
            if self._pyspin.IsReadable(node):
                return int(node.GetValue())
        except Exception:
            pass
        return int(default)

    def _read_float(self, nodemap, node_name: str, default: float) -> float:
        try:
            node = self._pyspin.CFloatPtr(nodemap.GetNode(node_name))
            if self._pyspin.IsReadable(node):
                return float(node.GetValue())
        except Exception:
            pass
        try:
            node = self._pyspin.CIntegerPtr(nodemap.GetNode(node_name))
            if self._pyspin.IsReadable(node):
                return float(node.GetValue())
        except Exception:
            pass
        return float(default)


class PointGreyCaptureService:
    """Runs capture plus a tiny Unix-socket control plane."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        device_serial: str | None,
        socket_path: str,
        shm_prefix: str,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.device_serial = device_serial
        self.socket_path = socket_path
        self.shm_prefix = shm_prefix
        self._device = PointGreyCaptureDevice(
            width=self.width,
            height=self.height,
            fps=self.fps,
            device_serial=self.device_serial,
        )
        self._writer = None
        self._server_socket = None
        self._server_thread = None
        self._capture_thread = None
        self._running = threading.Event()
        self._error_message = ""
        self._state_lock = threading.Lock()

    def start(self) -> None:
        self._device.start()
        self.width = self._device.width
        self.height = self._device.height
        self._writer = SharedFrameRingWriter(
            shm_prefix=self.shm_prefix,
            width=self.width,
            height=self.height,
            channels=3,
            slot_count=3,
        )
        self._writer.set_status(STATUS_RUNNING, timestamp_ns=time.time_ns())
        self._running.set()
        self._start_server()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def serve_forever(self) -> None:
        try:
            while self._running.is_set():
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("[PointGreyService] Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running.clear()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._device.stop()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        print("[PointGreyService] Stopped")

    def _start_server(self) -> None:
        socket_path = Path(self.socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o666)
        server.listen(8)
        server.settimeout(0.5)
        self._server_socket = server
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()

    def _server_loop(self) -> None:
        while self._running.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    request = recv_json_message(conn)
                    response = self._handle_request(request)
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                send_json_message(conn, response)

    def _handle_request(self, request: dict) -> dict:
        command = request.get("cmd")
        if command == "ping":
            return {"ok": True, "service": "pointgrey_capture_service"}
        if command == "describe":
            return {
                "ok": True,
                "service": "pointgrey_capture_service",
                "socket_path": self.socket_path,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "serial": self.device_serial or (self._device.camera_info or {}).get("serial"),
                "camera_info": self._device.camera_info,
                "error": self._error_message,
                "transport": self._writer.descriptor() if self._writer is not None else None,
            }
        if command == "shutdown":
            self._running.clear()
            return {"ok": True}
        return {"ok": False, "error": f"Unknown command: {command}"}

    def _capture_loop(self) -> None:
        while self._running.is_set():
            try:
                frame, timestamp_ns = self._device.capture_frame()
                self._writer.publish(frame, timestamp_ns)
                with self._state_lock:
                    self._error_message = ""
            except Exception as exc:
                self._writer.set_status(STATUS_ERROR, timestamp_ns=time.time_ns())
                with self._state_lock:
                    self._error_message = str(exc)
                time.sleep(0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone PointGrey capture service")
    parser.add_argument("--width", type=int, default=2448, help="Requested image width")
    parser.add_argument("--height", type=int, default=2048, help="Requested image height")
    parser.add_argument("--fps", type=int, default=30, help="Requested acquisition frame rate")
    parser.add_argument("--serial", type=str, default=None, help="Optional PointGrey camera serial")
    parser.add_argument(
        "--socket-path",
        type=str,
        default=DEFAULT_POINTGREY_SOCKET_PATH,
        help="Unix socket path for control requests",
    )
    parser.add_argument(
        "--shm-prefix",
        type=str,
        default=DEFAULT_POINTGREY_SHM_PREFIX,
        help="Shared-memory object name prefix",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = PointGreyCaptureService(
        width=args.width,
        height=args.height,
        fps=args.fps,
        device_serial=args.serial,
        socket_path=args.socket_path,
        shm_prefix=args.shm_prefix,
    )
    try:
        service.start()
        service.serve_forever()
        return 0
    except Exception as exc:
        print(f"[PointGreyService] Failed to start: {exc}", file=sys.stderr)
        service.stop()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
