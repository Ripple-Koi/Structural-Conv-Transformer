from typing import Final
from pathlib import Path
import pandas as pd
import argparse
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from typing import Final, List, Optional, Sequence, Set, Tuple
import numpy as np
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt

from build_graph import build_graph
from match_sv import match_sv

_DEFAULT_N_JOBS: Final[int] = 8


def generate_traj_and_graph(
    argoverse_scenario_dir: Path,
    output_dir: Path,
) -> None:
    """Generate and save 6-SV trajectories and 3 graphs for selected scenarios within `argoverse_scenario_dir`.

    Args:
        argoverse_scenario_dir: Path to local directory where Argoverse scenarios are stored.
        output_dir: Path to local directory where generated trajectories and graphs should be saved.
        num_scenarios: Maximum number of scenarios for which to generate trajectories and graphs.
    """
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    scenario_file_list = all_scenario_files
    graph_save_dir = output_dir / "graph"
    traj_save_dir = output_dir / "traj"
    graph_save_dir.mkdir(parents=True, exist_ok=True)
    traj_save_dir.mkdir(parents=True, exist_ok=True)


    # Build inner function to generate trajectories and graphs for a single scenario.
    def generate_scenario_traj_and_graph(
        scenario_path: Path
        ) -> Tuple[List[NDArrayInt], List[NDArrayInt], List[NDArrayInt], List[NDArrayFloat], List[NDArrayFloat]]:
        """Generate and save trajectories and graphs for a single Argoverse scenario.

        NOTE: This function assumes that the static map is stored in the same directory as the scenario file.

        Args:
            scenario_path: Path to the parquet file corresponding to the Argoverse scenario to visualize.
        """
        tracks = pd.read_parquet(scenario_path)
        focal = tracks[tracks['track_id']==tracks['focal_track_id']]
        if len(focal['timestep']) == 110:
            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
            graph_save_path = graph_save_dir / f"{scenario_id}.npz"
            traj_save_path = traj_save_dir / f"{scenario_id}.npz"
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            static_map = ArgoverseStaticMap.from_json(static_map_path)
            frames_global, frames_medium, frames_local = build_graph(scenario, static_map, graph_save_path)
            sv_traj, transformed_sv_traj = match_sv(tracks, traj_save_path)
        
            return frames_global, frames_medium, frames_local, sv_traj, transformed_sv_traj

    # Generate trajectories and graphs for each selected scenario in parallel
    with tqdm_joblib(desc='Preprocessing', total=25000) as progress_bar:
        outputs = Parallel(n_jobs=_DEFAULT_N_JOBS)(
            delayed(generate_scenario_traj_and_graph)(scenario_path) for scenario_path in scenario_file_list)

    frames_global = [output[0] for output in outputs if output is not None]
    frames_medium = [output[1] for output in outputs if output is not None]
    frames_local = [output[2] for output in outputs if output is not None]
    sv_traj = [output[3] for output in outputs if output is not None]
    transformed_sv_traj = [output[4] for output in outputs if output is not None]
    np.savez_compressed(
        str(output_dir), 
        GLOBAL=frames_global, 
        MEDIUM=frames_medium, 
        LOCAL=frames_local, 
        ORIGIN=sv_traj, 
        TRANSFORM=transformed_sv_traj)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Argoverse2 dataset')
    parser.add_argument("-i", "--argoverse_scenario_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    args = parser.parse_args()

    generate_traj_and_graph(Path(args.argoverse_scenario_dir), Path(args.output_dir))
