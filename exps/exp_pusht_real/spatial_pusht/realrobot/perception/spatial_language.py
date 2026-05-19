"""Helpers for replay-time voxel discretization and spatial-language strings.

The token serialization below is adapted from the GraspGPT Push-T implementation:
- https://github.com/wuminye/GraspGPT/blob/pusht/graspGPT/model/pushT_dataset.py
- https://github.com/wuminye/GraspGPT/blob/pusht/graspGPT/model/parser_and_serializer.py

Only the minimal STATE -> PUSHER/TBAR sentence path is copied here so the replay
GUI can reuse the original sentence format without importing the full training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import trimesh


Coord2D = tuple[int, int]


@dataclass(frozen=True)
class CB:
    coord: Coord2D


@dataclass(frozen=True)
class Pusher:
    cbs: list[CB]


@dataclass(frozen=True)
class TBar:
    cbs: list[CB]


@dataclass(frozen=True)
class Goal:
    cbs: list[CB]


@dataclass(frozen=True)
class Movement:
    target: CB


@dataclass(frozen=True)
class State:
    pusher: Pusher
    tbar: TBar
    goal: Goal | None = None


@dataclass(frozen=True)
class Seq:
    items: list[State | Movement]


class Serializer:
    """Minimal serializer copied from GraspGPT's parser_and_serializer.py."""

    @staticmethod
    def serialize(seq: Seq) -> list[str | Coord2D]:
        tokens: list[str | Coord2D] = []
        for item in seq.items:
            if isinstance(item, State):
                tokens.extend(Serializer.serialize_state(item))
            elif isinstance(item, Movement):
                tokens.extend(Serializer.serialize_movement(item))
        return tokens

    @staticmethod
    def serialize_state(state: State) -> list[str | Coord2D]:
        tokens: list[str | Coord2D] = ["state", "pusher"]
        tokens.extend(cb.coord for cb in state.pusher.cbs)
        tokens.append("tbar")
        tokens.extend(cb.coord for cb in state.tbar.cbs)
        if state.goal is not None:
            tokens.append("goal")
            tokens.extend(cb.coord for cb in state.goal.cbs)
        return tokens

    @staticmethod
    def serialize_movement(movement: Movement) -> list[str | Coord2D]:
        return ["moveto", movement.target.coord]


@dataclass(frozen=True)
class SpatialLanguageResult:
    available: bool
    sentence: str
    status: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    resolution_xyz: np.ndarray
    voxel_size_xyz: np.ndarray
    tblock_voxels_3d: np.ndarray
    pusher_voxels_3d: np.ndarray
    tblock_voxels_2d_full: np.ndarray
    tblock_voxels_2d: np.ndarray
    pusher_voxels_2d: np.ndarray
    processing_time_ms: float | None = None


def default_bbox() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([-0.05, -0.1, 0.0], dtype=np.float64),
        np.array([0.45, 0.45, 0.1], dtype=np.float64),
    )


def format_tokens(tokens: Iterable[str | Coord2D]) -> str:
    return " ".join(
        f"({int(token[0])},{int(token[1])})" if isinstance(token, tuple) else str(token)
        for token in tokens
    )


def build_state_sentence(
    pusher_coords: Iterable[Coord2D],
    tbar_coords: Iterable[Coord2D],
    goal_coords: Iterable[Coord2D] | None = None,
) -> str:
    pusher = [CB(coord=coord) for coord in sorted(set(tuple(coord) for coord in pusher_coords))]
    tbar = [CB(coord=coord) for coord in sorted(set(tuple(coord) for coord in tbar_coords))]
    goal = None
    if goal_coords is not None:
        goal_cbs = [CB(coord=coord) for coord in sorted(set(tuple(coord) for coord in goal_coords))]
        if goal_cbs:
            goal = Goal(goal_cbs)
    if not pusher or not tbar:
        return ""
    return format_tokens(
        Serializer.serialize_state(State(Pusher(pusher), TBar(tbar), goal=goal))
    )


def build_moveto_sentence(movement_coords: Iterable[Coord2D]) -> str:
    movements = [Movement(target=CB(coord=tuple(coord))) for coord in movement_coords]
    return format_tokens(Serializer.serialize(Seq(items=movements)))


def append_moveto_phrases(
    base_sentence: str,
    movement_coords: Iterable[Coord2D],
) -> str:
    base_sentence = str(base_sentence).strip()
    movement_sentence = build_moveto_sentence(movement_coords).strip()
    if base_sentence and movement_sentence:
        return f"{base_sentence} {movement_sentence}"
    return base_sentence or movement_sentence


def build_composited_sentence(
    *,
    pusher_coords: Iterable[Coord2D],
    tbar_coords: Iterable[Coord2D],
    goal_coords: Iterable[Coord2D] | None = None,
    movement_coords: Iterable[Coord2D] = (),
) -> str:
    pusher = [CB(coord=coord) for coord in sorted(set(tuple(coord) for coord in pusher_coords))]
    tbar = [CB(coord=coord) for coord in sorted(set(tuple(coord) for coord in tbar_coords))]
    if not pusher or not tbar:
        return ""
    goal = None
    if goal_coords is not None:
        goal_cbs = [CB(coord=coord) for coord in sorted(set(tuple(coord) for coord in goal_coords))]
        if goal_cbs:
            goal = Goal(goal_cbs)
    items: list[State | Movement] = [State(Pusher(pusher), TBar(tbar), goal=goal)]
    items.extend(
        Movement(target=CB(coord=tuple(coord)))
        for coord in movement_coords
    )
    return format_tokens(Serializer.serialize(Seq(items=items)))


def project_point_to_spatial_coord(
    point_world: np.ndarray,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution_xyz: np.ndarray,
) -> Coord2D | None:
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    resolution_xyz = np.maximum(np.asarray(resolution_xyz, dtype=np.int32).reshape(3), 1)
    extent = bbox_max - bbox_min
    if np.any(extent <= 0.0):
        raise ValueError("Bounding box max must be greater than min on all axes.")
    voxel_size_xyz = extent / resolution_xyz.astype(np.float64)
    point_voxels_3d = _points_to_voxel_indices(
        np.asarray(point_world, dtype=np.float64).reshape(1, 3),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size_xyz=voxel_size_xyz,
        resolution_xyz=resolution_xyz,
    )
    point_voxels_2d = _orthographic_project_xy(point_voxels_3d, resolution_xyz=resolution_xyz)
    if len(point_voxels_2d) == 0:
        return None
    coord = point_voxels_2d[0]
    return (int(coord[0]), int(coord[1]))


def compute_spatial_language(
    *,
    mesh_vertices_world: np.ndarray,
    mesh_faces: np.ndarray,
    pusher_point_world: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution_xyz: np.ndarray,
) -> SpatialLanguageResult:
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    resolution_xyz = np.maximum(np.asarray(resolution_xyz, dtype=np.int32).reshape(3), 1)
    extent = bbox_max - bbox_min
    if np.any(extent <= 0.0):
        raise ValueError("Bounding box max must be greater than min on all axes.")

    voxel_size_xyz = extent / resolution_xyz.astype(np.float64)
    tblock_voxels_3d = _voxelize_mesh_to_grid(
        mesh_vertices_world=np.asarray(mesh_vertices_world, dtype=np.float64),
        mesh_faces=np.asarray(mesh_faces, dtype=np.int64),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size_xyz=voxel_size_xyz,
        resolution_xyz=resolution_xyz,
    )
    pusher_voxels_3d = _points_to_voxel_indices(
        np.asarray(pusher_point_world, dtype=np.float64).reshape(1, 3),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size_xyz=voxel_size_xyz,
        resolution_xyz=resolution_xyz,
    )
    tblock_voxels_2d_full = _orthographic_project_xy(tblock_voxels_3d, resolution_xyz=resolution_xyz)
    tblock_voxels_2d = _extract_2d_boundary_voxels(
        tblock_voxels_2d_full,
        resolution_xy=resolution_xyz[:2],
    )
    pusher_voxels_2d = _orthographic_project_xy(pusher_voxels_3d, resolution_xyz=resolution_xyz)
    sentence = build_state_sentence(
        [tuple(coord) for coord in pusher_voxels_2d.tolist()],
        [tuple(coord) for coord in tblock_voxels_2d.tolist()],
    )
    status = (
        f"Tblock voxels: {len(tblock_voxels_3d)} | "
        f"Pusher voxels: {len(pusher_voxels_3d)} | "
        f"2D cells: T={len(tblock_voxels_2d)}, P={len(pusher_voxels_2d)}"
    )
    if not sentence:
        missing_parts = []
        if len(tblock_voxels_2d) == 0:
            missing_parts.append("Tblock")
        if len(pusher_voxels_2d) == 0:
            missing_parts.append("pusher")
        if missing_parts:
            status = (
                f"No projected {'/'.join(missing_parts)} voxels inside the bounding box. "
                f"{status}"
            )
        else:
            status = f"Unable to build a spatial sentence. {status}"
    return SpatialLanguageResult(
        available=bool(sentence),
        sentence=sentence,
        status=status,
        processing_time_ms=None,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution_xyz=resolution_xyz,
        voxel_size_xyz=voxel_size_xyz,
        tblock_voxels_3d=tblock_voxels_3d,
        pusher_voxels_3d=pusher_voxels_3d,
        tblock_voxels_2d_full=tblock_voxels_2d_full,
        tblock_voxels_2d=tblock_voxels_2d,
        pusher_voxels_2d=pusher_voxels_2d,
    )


def unavailable_result(
    status: str,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution_xyz: np.ndarray,
) -> SpatialLanguageResult:
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    resolution_xyz = np.maximum(np.asarray(resolution_xyz, dtype=np.int32).reshape(3), 1)
    voxel_size_xyz = (bbox_max - bbox_min) / resolution_xyz.astype(np.float64)
    empty_3d = np.zeros((0, 3), dtype=np.int32)
    empty_2d = np.zeros((0, 2), dtype=np.int32)
    return SpatialLanguageResult(
        available=False,
        sentence="",
        status=status,
        processing_time_ms=None,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution_xyz=resolution_xyz,
        voxel_size_xyz=voxel_size_xyz,
        tblock_voxels_3d=empty_3d,
        pusher_voxels_3d=empty_3d,
        tblock_voxels_2d_full=empty_2d,
        tblock_voxels_2d=empty_2d,
        pusher_voxels_2d=empty_2d,
    )


def render_projected_voxels_image(
    result: SpatialLanguageResult,
    *,
    scale: int = 16,
) -> np.ndarray:
    resolution_x = max(int(result.resolution_xyz[0]), 1)
    resolution_y = max(int(result.resolution_xyz[1]), 1)
    image = np.full((resolution_y, resolution_x, 3), 255, dtype=np.uint8)
    for x_idx, y_idx in result.tblock_voxels_2d:
        image[resolution_y - 1 - int(y_idx), int(x_idx)] = np.array([80, 170, 255], dtype=np.uint8)
    for x_idx, y_idx in result.pusher_voxels_2d:
        row = resolution_y - 1 - int(y_idx)
        col = int(x_idx)
        image[row, col] = np.array([255, 110, 80], dtype=np.uint8)
    scale = max(int(scale), 1)
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def build_voxel_mesh(
    indices_xyz: np.ndarray,
    *,
    bbox_min: np.ndarray,
    voxel_size_xyz: np.ndarray,
    color_rgba: tuple[int, int, int, int],
) -> trimesh.Trimesh | None:
    indices_xyz = np.asarray(indices_xyz, dtype=np.int32).reshape(-1, 3)
    if len(indices_xyz) == 0:
        return None
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    voxel_size_xyz = np.asarray(voxel_size_xyz, dtype=np.float64).reshape(3)
    centers = bbox_min + (indices_xyz.astype(np.float64) + 0.5) * voxel_size_xyz
    extents = voxel_size_xyz * 0.92
    meshes = []
    for center in centers:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = center
        box = trimesh.creation.box(extents=extents, transform=transform)
        box.visual.face_colors = np.tile(
            np.asarray(color_rgba, dtype=np.uint8).reshape(1, 4),
            (len(box.faces), 1),
        )
        meshes.append(box)
    return trimesh.util.concatenate(meshes)


def _orthographic_project_xy(indices_xyz: np.ndarray, *, resolution_xyz: np.ndarray) -> np.ndarray:
    indices_xyz = np.asarray(indices_xyz, dtype=np.int32).reshape(-1, 3)
    if len(indices_xyz) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    resolution_xyz = np.maximum(np.asarray(resolution_xyz, dtype=np.int32).reshape(3), 1)
    occupancy = np.zeros(tuple(resolution_xyz.tolist()), dtype=bool)
    occupancy[indices_xyz[:, 0], indices_xyz[:, 1], indices_xyz[:, 2]] = True
    occupancy_xy = np.any(occupancy, axis=2)
    indices_xy = np.argwhere(occupancy_xy)
    order = np.lexsort((indices_xy[:, 1], indices_xy[:, 0]))
    return indices_xy[order].astype(np.int32)


def _extract_2d_boundary_voxels(
    indices_xy: np.ndarray,
    *,
    resolution_xy: np.ndarray,
) -> np.ndarray:
    indices_xy = np.asarray(indices_xy, dtype=np.int32).reshape(-1, 2)
    if len(indices_xy) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    resolution_xy = np.maximum(np.asarray(resolution_xy, dtype=np.int32).reshape(2), 1)
    occupancy = np.zeros((int(resolution_xy[0]), int(resolution_xy[1])), dtype=bool)
    occupancy[indices_xy[:, 0], indices_xy[:, 1]] = True
    boundary = np.zeros_like(occupancy)
    for x_idx, y_idx in indices_xy:
        x_idx = int(x_idx)
        y_idx = int(y_idx)
        if (
            x_idx == 0
            or y_idx == 0
            or x_idx == int(resolution_xy[0]) - 1
            or y_idx == int(resolution_xy[1]) - 1
        ):
            boundary[x_idx, y_idx] = True
            continue
        if not (
            occupancy[x_idx - 1, y_idx]
            and occupancy[x_idx + 1, y_idx]
            and occupancy[x_idx, y_idx - 1]
            and occupancy[x_idx, y_idx + 1]
        ):
            boundary[x_idx, y_idx] = True
    boundary_indices = np.argwhere(boundary)
    order = np.lexsort((boundary_indices[:, 1], boundary_indices[:, 0]))
    return boundary_indices[order].astype(np.int32)


def _voxelize_mesh_to_grid(
    *,
    mesh_vertices_world: np.ndarray,
    mesh_faces: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size_xyz: np.ndarray,
    resolution_xyz: np.ndarray,
) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=mesh_vertices_world, faces=mesh_faces, process=False)
    sample_points = [mesh.vertices]
    try:
        pitch = float(max(np.min(voxel_size_xyz), 1e-4))
        voxel_grid = mesh.voxelized(pitch=pitch)
        sample_points.append(np.asarray(voxel_grid.points, dtype=np.float64))
    except Exception:
        pass
    try:
        surface_samples = max(len(mesh.faces) * 12, 4096)
        sampled_points, _ = trimesh.sample.sample_surface(mesh, surface_samples)
        sample_points.append(np.asarray(sampled_points, dtype=np.float64))
    except Exception:
        pass
    merged = np.concatenate([pts for pts in sample_points if len(pts) > 0], axis=0)
    return _points_to_voxel_indices(
        merged,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size_xyz=voxel_size_xyz,
        resolution_xyz=resolution_xyz,
    )


def _points_to_voxel_indices(
    points_world: np.ndarray,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size_xyz: np.ndarray,
    resolution_xyz: np.ndarray,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    upper = bbox_max + 1e-9
    inside = np.all((points_world >= bbox_min) & (points_world <= upper), axis=1)
    if not np.any(inside):
        return np.zeros((0, 3), dtype=np.int32)
    selected = points_world[inside]
    indices = np.floor((selected - bbox_min) / voxel_size_xyz).astype(np.int32)
    resolution_xyz = np.maximum(np.asarray(resolution_xyz, dtype=np.int32).reshape(3), 1)
    indices = np.clip(indices, 0, resolution_xyz - 1)
    unique = np.unique(indices, axis=0)
    order = np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0]))
    return unique[order].astype(np.int32)
