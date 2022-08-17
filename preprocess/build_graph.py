from typing import Final, List, Optional, Sequence, Set, Tuple
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms
import io
import math
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType, TrackCategory
from av2.utils.typing import NDArrayFloat, NDArrayInt


_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_MOTORCYCLIST_LENGTH_M: Final[float] = 2.5
_ESTIMATED_MOTORCYCLIST_WIDTH_M: Final[float] = 1.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 1.8
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_ESTIMATED_BUS_LENGTH_M: Final[float] = 12.0
_ESTIMATED_BUS_WIDTH_M: Final[float] = 2.5
_PLOT_BOUNDS_BUFFER_M: Final[float] = 36.0

# Actor: 192 to 255
_PEDESTRIAN_COLOR: Final[str] = str((192 + 64 / 7 * 6) / 255)
_CYCLIST_COLOR: Final[str] = str((192 + 64 / 7 * 5) / 255)
_MOTORCYCLIST_COLOR: Final[str] = str((192 + 64 / 7 * 4) / 255)
_CAR_COLOR: Final[str] = str((192 + 64 / 7 * 3) / 255)
_BUS_COLOR: Final[str] = str((192 + 64 / 7 * 2) / 255)
_OBSTACLE_COLOR: Final[str] = str((192 + 64 / 7 * 1) / 255)

# Area: 128 to 191
_CROSSWALK_AREA_COLOR: Final[str] = str((128 + 64 / 3 * 2) / 255)
_DRIVABLE_AREA_COLOR: Final[str] = str((128 + 64 / 3 * 1) / 255)

# Line: 64 to 127
_DASHED_LINE_COLOR: Final[str] = str((64 + 64 / 4 * 3) / 255)
_SOLID_LINE_COLOR: Final[str] = str((64 + 64 / 4 * 2) / 255)
_DOUBLE_LINE_COLOR: Final[str] = str((64 + 64 / 4 * 1) / 255)

# Lane: 0 to 63
_VEHICLE_LANE_COLOR: Final[str] = str((64 / 4 * 3) / 255)
_BIKE_LANE_COLOR: Final[str] = str((64 / 4 * 2) / 255)
_BUS_LANE_COLOR: Final[str] = str((64 / 4 * 1) / 255)

_COLOR_DICT: Final[dict] = {
    "VEHICLE": _VEHICLE_LANE_COLOR,
    "BIKE": _BIKE_LANE_COLOR,
    "BUS": _BUS_LANE_COLOR,
    "DASH_SOLID_YELLOW": _SOLID_LINE_COLOR,
    "DASH_SOLID_WHITE": _SOLID_LINE_COLOR,
    "DASHED_WHITE": _DASHED_LINE_COLOR,
    "DASHED_YELLOW": _DASHED_LINE_COLOR,
    "DOUBLE_SOLID_YELLOW": _DOUBLE_LINE_COLOR,
    "DOUBLE_SOLID_WHITE": _DOUBLE_LINE_COLOR,
    "DOUBLE_DASH_YELLOW": _DASHED_LINE_COLOR,
    "DOUBLE_DASH_WHITE": _DASHED_LINE_COLOR,
    "SOLID_YELLOW": _SOLID_LINE_COLOR,
    "SOLID_WHITE": _SOLID_LINE_COLOR,
    "SOLID_DASH_WHITE": _DASHED_LINE_COLOR,
    "SOLID_DASH_YELLOW": _DASHED_LINE_COLOR,
    "SOLID_BLUE": _DOUBLE_LINE_COLOR,
    "NONE": _SOLID_LINE_COLOR,
    "UNKNOWN": _SOLID_LINE_COLOR,
}

_BOUNDING_BOX_ZORDER: Final[int] = 100  # Ensure actor bounding boxes are plotted on top of all map elements

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}


def build_graph(scenario: ArgoverseScenario, scenario_static_map: ArgoverseStaticMap, save_dir: Path) -> None:
    """Build dynamic visualization for all tracks and the local map associated with an Argoverse scenario.

    Args:
        scenario: Argoverse scenario to visualize.
        scenario_static_map: Local static map elements associated with `scenario`.
        save_dir: Folder where preprocessed trajectories, figure and graphs are saved.
    """
    # Build each frame for the video
    frames_global: List[NDArrayInt] = []
    frames_medium: List[NDArrayInt] = []
    frames_local: List[NDArrayInt] = []
    save_path_global = save_dir / "global"
    save_path_medium = save_dir / "medium"
    save_path_local = save_dir / "local"
    save_path_global.mkdir(parents=True,exist_ok=True)
    save_path_medium.mkdir(parents=True,exist_ok=True)
    save_path_local.mkdir(parents=True,exist_ok=True)

    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
        ratio = 1  # control figure size and resolution
        _, ax = plt.subplots(figsize=(0.72 * ratio, 0.24 * ratio), dpi=100)  # 72*24 pixels, fit map bounds 72*24, 36*12, 18*6

        # Plot static map elements and actor tracks
        _plot_static_map_elements(scenario_static_map)
        cur_plot_bounds, cur_heading = _plot_actor_tracks(ax, scenario, timestep)
        if cur_plot_bounds:
            plot_bounds = cur_plot_bounds
            rotate_angle = - cur_heading / math.pi * 180
        
        # Move and rotate the graph to focal-vehicle-centered
        transform = transforms.Affine2D().rotate_deg_around(plot_bounds[1], plot_bounds[3], rotate_angle)
        trans_data = transform + ax.transData
        # last 6 artists (TODO: confirm it) are not scenario elements, thus skip their rotation
        for artist in ax.get_children()[:-6]:
            artist.set_transform(trans_data)

        # Set map bounds to capture focal trajectory history (with fixed buffer in all directions)
        for line in plt.gca().lines:
            line.set_linewidth(0.1 * ratio)
            line.set_markersize(0.6 * ratio)
        plt.xlim(plot_bounds[1] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M)
        plt.ylim(plot_bounds[3] - _PLOT_BOUNDS_BUFFER_M / 3, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M / 3)
        plt.gca().set_aspect("equal", adjustable="box")

        # Minimize plot margins and make axes invisible
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Save plotted frame as array and image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        frame = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        frames_global.append(frame)
        cv2.imwrite(str(save_path_global / f'{timestep}.png'), frame)

        # Set map bounds, line width and marker size for medium map
        for line in plt.gca().lines:
            line.set_linewidth(0.2 * ratio)
            line.set_markersize(1.2 * ratio)
        plt.xlim(plot_bounds[1] - _PLOT_BOUNDS_BUFFER_M / 2, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M / 2)
        plt.ylim(plot_bounds[3] - _PLOT_BOUNDS_BUFFER_M / 6, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M / 6)

        # Save plotted frame as array and image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        frame = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        frames_medium.append(frame)
        cv2.imwrite(str(save_path_medium / f'{timestep}.png'), frame)

        # Set map bounds, line width and marker size for local map
        for line in plt.gca().lines:
            line.set_linewidth(0.4 * ratio)
            line.set_markersize(2.4 * ratio)
        plt.xlim(plot_bounds[1] - _PLOT_BOUNDS_BUFFER_M / 4, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M / 4)
        plt.ylim(plot_bounds[3] - _PLOT_BOUNDS_BUFFER_M / 12, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M / 12)

        # Save plotted frame as array and image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        frame = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        frames_local.append(frame)
        cv2.imwrite(str(save_path_local / f'{timestep}.png'), frame)
    
    np.savez_compressed(str(save_dir / "all.npz"), GLOBAL=frames_global, MEDIUM=frames_medium, LOCAL=frames_local)


def _plot_static_map_elements(static_map: ArgoverseStaticMap, show_ped_xings: bool = True) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        # Plot lanes
        lane_color = _COLOR_DICT[lane_segment.lane_type]
        _plot_polygons([lane_segment.polygon_boundary], alpha=1.0, color=lane_color)

        #Plot lane marks
        left_lane_mark_color = _COLOR_DICT[lane_segment.left_mark_type]
        _plot_polylines(
            [lane_segment.left_lane_boundary.xyz],
            line_width=1,
            color=left_lane_mark_color,
        )
        right_lane_mark_color = _COLOR_DICT[lane_segment.right_mark_type]
        _plot_polylines(
            [lane_segment.right_lane_boundary.xyz],
            line_width=1,
            color=right_lane_mark_color,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polygons([ped_xing.polygon], alpha=0.5, color=_CROSSWALK_AREA_COLOR)
            # _plot_polylines([ped_xing.edge1.xyz, ped_xing.edge2.xyz], alpha=1, color=_LANE_SEGMENT_COLOR)


def _plot_actor_tracks(ax: plt.Axes, scenario: ArgoverseScenario, timestep: int) -> Optional[_PlotBounds]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    for track in scenario.tracks:
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [object_state.timestep for object_state in track.object_states if object_state.timestep <= timestep]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        # Get actor trajectory and heading history
        actor_trajectory: NDArrayFloat = np.array(
            [list(object_state.position) for object_state in track.object_states if object_state.timestep <= timestep]
        )
        actor_headings: NDArrayFloat = np.array(
            [object_state.heading for object_state in track.object_states if object_state.timestep <= timestep]
        )

        if track.category == TrackCategory.FOCAL_TRACK:
            x_min, x_max = actor_trajectory[0, 0], actor_trajectory[-1, 0]
            y_min, y_max = actor_trajectory[0, 1], actor_trajectory[-1, 1]
            track_bounds = (x_min, x_max, y_min, y_max)
            cur_heading = actor_headings[-1]

        # Plot all actors
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                _CAR_COLOR,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif track.object_type == ObjectType.CYCLIST:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                _CYCLIST_COLOR,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        elif track.object_type == ObjectType.MOTORCYCLIST:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                _MOTORCYCLIST_COLOR,
                (_ESTIMATED_MOTORCYCLIST_LENGTH_M, _ESTIMATED_MOTORCYCLIST_WIDTH_M),
            )
        elif track.object_type == ObjectType.BUS:
            _plot_actor_bounding_box(
                ax,
                actor_trajectory[-1],
                actor_headings[-1],
                _BUS_COLOR,
                (_ESTIMATED_BUS_LENGTH_M, _ESTIMATED_BUS_WIDTH_M),
            )
        elif track.object_type == ObjectType.PEDESTRIAN:
            plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1, 1], "o", color=_PEDESTRIAN_COLOR, markersize=6)
        elif track.object_type in _STATIC_OBJECT_TYPES:
            plt.plot(actor_trajectory[-1, 0], actor_trajectory[-1, 1], "D", color=_OBSTACLE_COLOR, markersize=6)

    return track_bounds, cur_heading


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], style, linewidth=line_width, color=color, alpha=alpha)


def _plot_polygons(polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r") -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(
    ax: plt.Axes, cur_location: NDArrayFloat, heading: float, color: str, bbox_size: Tuple[float, float]
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width, np.degrees(heading), color=color, zorder=_BOUNDING_BOX_ZORDER
    )
    ax.add_patch(vehicle_bounding_box)
