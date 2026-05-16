#!/usr/bin/env python3
import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk


class EpisodeViewer:
    def __init__(self, root: tk.Tk, json_path: Path) -> None:
        self.root = root
        self.root.title("Spatial Episode Viewer")

        self.json_path = json_path
        self.video_path: Path | None = None
        self.episode = {}
        self.frames = []
        self.frame_sources = []
        self.grid_width = 128
        self.grid_height = 128
        self.video_fps = 5.0

        self.cap: cv2.VideoCapture | None = None
        self.video_frame_count = 0
        self.spatial_frame_count = 0
        self.current_index = 0
        self.playing = False
        self.after_id: str | None = None

        self.video_panel: ttk.Label
        self.map_panel: ttk.Label
        self.frame_slider: ttk.Scale
        self.status_var = tk.StringVar()
        self.play_button: ttk.Button

        self.video_photo: ImageTk.PhotoImage | None = None
        self.map_photo: ImageTk.PhotoImage | None = None

        self._load_episode(json_path)
        self._build_ui()
        self._render_frame(0)

    def _load_episode(self, json_path: Path) -> None:
        with json_path.open("r", encoding="utf-8") as f:
            self.episode = json.load(f)

        self.frames, self.frame_sources = self._extract_frame_data(self.episode)
        if not self.frames:
            raise ValueError("JSON 中未找到可逐帧播放的 spatial 数据。")
        self.spatial_frame_count = len(self.frames)

        resolution = self.episode.get("spatial_config", {}).get("resolution_xyz", [128, 128, 1])
        if len(resolution) >= 2:
            self.grid_width = int(resolution[0])
            self.grid_height = int(resolution[1])

        self.video_fps = float(self.episode.get("rgb_video_fps") or 5.0)
        video_file = self.episode.get("rgb_video_file")
        if video_file:
            self.video_path = (json_path.parent / video_file).resolve()
        else:
            candidate = json_path.with_suffix(".mp4")
            self.video_path = candidate.resolve() if candidate.exists() else None

        if not self.video_path or not self.video_path.exists():
            raise FileNotFoundError("未找到与 JSON 对应的 MP4 文件。")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {self.video_path}")

        self.video_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.video_frame_count <= 0:
            self.video_frame_count = len(self.frames)

    def _extract_frame_data(self, episode: dict) -> tuple[list[dict], list[str]]:
        movement_frames = []
        movement_sources = []
        for movement_index, movement in enumerate(episode.get("movements", [])):
            for frame_index, frame in enumerate(movement.get("frames", [])):
                spatial = frame.get("spatial")
                if isinstance(spatial, dict) and spatial.get("tblock_coords") is not None:
                    movement_frames.append(spatial)
                    movement_sources.append(f"movement {movement_index + 1} / frame {frame_index + 1}")

        if movement_frames:
            return movement_frames, movement_sources

        spatial_history = episode.get("spatial_history", [])
        if spatial_history:
            return spatial_history, [f"spatial_history {index + 1}" for index in range(len(spatial_history))]

        return [], []

    def _build_ui(self) -> None:
        self.root.geometry("1480x900")
        self.root.minsize(1100, 700)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        self.video_panel = ttk.Label(main, anchor="center")
        self.video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.map_panel = ttk.Label(main, anchor="center")
        self.map_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        controls = ttk.Frame(main, padding=(0, 12, 0, 0))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.play_button = ttk.Button(controls, text="播放", command=self._toggle_play)
        self.play_button.grid(row=0, column=0, padx=(0, 8))

        self.frame_slider = ttk.Scale(
            controls,
            from_=0,
            to=max(self.spatial_frame_count - 1, 0),
            orient="horizontal",
            command=self._on_slider_change,
        )
        self.frame_slider.grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(controls, text="上一帧", command=self._prev_frame).grid(row=0, column=2, padx=8)
        ttk.Button(controls, text="下一帧", command=self._next_frame).grid(row=0, column=3, padx=8)
        ttk.Button(controls, text="打开 JSON", command=self._open_json).grid(row=0, column=4, padx=(8, 0))

        status = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.root.bind("<space>", lambda _event: self._toggle_play())
        self.root.bind("<Left>", lambda _event: self._prev_frame())
        self.root.bind("<Right>", lambda _event: self._next_frame())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.config(text="暂停" if self.playing else "播放")
        if self.playing:
            self._schedule_next_frame()
        else:
            self._cancel_schedule()

    def _schedule_next_frame(self) -> None:
        self._cancel_schedule()
        delay_ms = max(int(1000 / max(self.video_fps, 0.1)), 1)
        self.after_id = self.root.after(delay_ms, self._play_step)

    def _play_step(self) -> None:
        if not self.playing:
            return
        next_index = self.current_index + 1
        if next_index >= self.spatial_frame_count:
            next_index = self.spatial_frame_count - 1
        self._render_frame(next_index)
        self.frame_slider.set(next_index)
        if next_index >= self.spatial_frame_count - 1:
            self.playing = False
            self.play_button.config(text="播放")
            self._cancel_schedule()
        else:
            self._schedule_next_frame()

    def _cancel_schedule(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _prev_frame(self) -> None:
        self.playing = False
        self.play_button.config(text="播放")
        self._cancel_schedule()
        self._render_frame(max(self.current_index - 1, 0))
        self.frame_slider.set(self.current_index)

    def _next_frame(self) -> None:
        self.playing = False
        self.play_button.config(text="播放")
        self._cancel_schedule()
        self._render_frame(min(self.current_index + 1, self.spatial_frame_count - 1))
        self.frame_slider.set(self.current_index)

    def _on_slider_change(self, value: str) -> None:
        index = int(float(value))
        if index != self.current_index:
            self.playing = False
            self.play_button.config(text="播放")
            self._cancel_schedule()
            self._render_frame(index)

    def _read_video_frame(self, index: int) -> Image.Image:
        assert self.cap is not None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"读取视频第 {index} 帧失败。")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def _build_map_image(self, frame_data: dict, size: tuple[int, int] = (560, 560)) -> Image.Image:
        img = Image.new("RGB", size, color=(246, 248, 251))
        draw = ImageDraw.Draw(img)
        pad = 24
        cell_w = (size[0] - pad * 2) / max(self.grid_width, 1)
        cell_h = (size[1] - pad * 2) / max(self.grid_height, 1)

        draw.rectangle((pad, pad, size[0] - pad, size[1] - pad), outline=(140, 150, 165), width=2)

        for step in range(0, self.grid_width + 1, max(self.grid_width // 8, 1)):
            x = pad + step * cell_w
            draw.line((x, pad, x, size[1] - pad), fill=(220, 225, 232), width=1)
        for step in range(0, self.grid_height + 1, max(self.grid_height // 8, 1)):
            y = pad + step * cell_h
            draw.line((pad, y, size[0] - pad, y), fill=(220, 225, 232), width=1)

        self._draw_points(draw, frame_data.get("tblock_coords", []), pad, cell_w, cell_h, (227, 92, 74), 3)
        self._draw_points(draw, frame_data.get("pusher_coords", []), pad, cell_w, cell_h, (34, 116, 224), 6)
        self._draw_centroid(draw, frame_data.get("tblock_coords", []), pad, cell_w, cell_h, (139, 34, 20))
        self._draw_centroid(draw, frame_data.get("pusher_coords", []), pad, cell_w, cell_h, (12, 73, 156))

        draw.text((pad, 4), "tblock (current)", fill=(227, 92, 74))
        draw.text((pad + 160, 4), "pusher (current)", fill=(34, 116, 224))
        return img

    def _draw_points(
        self,
        draw: ImageDraw.ImageDraw,
        points: list[list[int]],
        pad: int,
        cell_w: float,
        cell_h: float,
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        for x, y in points:
            cx = pad + (y + 0.5) * cell_w
            cy = pad + (x + 0.5) * cell_h
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=color)

    def _draw_centroid(
        self,
        draw: ImageDraw.ImageDraw,
        points: list[list[int]],
        pad: int,
        cell_w: float,
        cell_h: float,
        color: tuple[int, int, int],
    ) -> None:
        if not points:
            return
        cx0 = sum(y for _, y in points) / len(points)
        cy0 = sum(x for x, _ in points) / len(points)
        cx = pad + (cx0 + 0.5) * cell_w
        cy = pad + (cy0 + 0.5) * cell_h
        r = 8
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
        draw.line((cx - r - 2, cy, cx + r + 2, cy), fill=color, width=2)
        draw.line((cx, cy - r - 2, cx, cy + r + 2), fill=color, width=2)

    def _fit_image(self, image: Image.Image, bounds: tuple[int, int]) -> Image.Image:
        target = image.copy()
        target.thumbnail(bounds, Image.Resampling.LANCZOS)
        return target

    def _render_frame(self, index: int) -> None:
        if index < 0 or index >= self.spatial_frame_count:
            return

        frame_data = self.frames[index]
        video_index = index
        video_img = self._read_video_frame(video_index)
        map_img = self._build_map_image(frame_data)

        fitted_video = self._fit_image(video_img, (920, 760))
        fitted_map = self._fit_image(map_img, (520, 520))

        self.video_photo = ImageTk.PhotoImage(fitted_video)
        self.map_photo = ImageTk.PhotoImage(fitted_map)
        self.video_panel.config(image=self.video_photo)
        self.map_panel.config(image=self.map_photo)

        self.current_index = index
        self.status_var.set(
            f"JSON: {self.json_path.name} | 视频: {self.video_path.name if self.video_path else '-'} | "
            f"帧 {index + 1}/{self.spatial_frame_count} | {self.frame_sources[index]} | "
            f"tblock 点数: {len(frame_data.get('tblock_coords', []))} | "
            f"pusher 点数: {len(frame_data.get('pusher_coords', []))}"
        )

    def _open_json(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 episode JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self.json_path.parent),
        )
        if not path:
            return
        self._cancel_schedule()
        self.playing = False
        self.play_button.config(text="播放")
        if self.cap is not None:
            self.cap.release()
        try:
            self.json_path = Path(path).resolve()
            self._load_episode(self.json_path)
            self.frame_slider.configure(to=max(self.spatial_frame_count - 1, 0))
            self.frame_slider.set(0)
            self._render_frame(0)
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _on_close(self) -> None:
        self._cancel_schedule()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化播放 spatial episode 的视频与逐帧坐标。")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="/Users/wuminye/code/new_cap/spatial_episode_20260513_181635_580.json",
        help="episode JSON 文件路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json_path).expanduser().resolve()
    if not json_path.exists():
        print(f"JSON 文件不存在: {json_path}", file=sys.stderr)
        return 1

    root = tk.Tk()
    try:
        EpisodeViewer(root, json_path)
    except Exception as exc:
        messagebox.showerror("启动失败", str(exc))
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
