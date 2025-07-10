import torch
from environments.diverse_maze.utils import load_uniform
from environments.diverse_maze.data_generation.maze_stats import RENDER_STATS
import environments.diverse_maze.ant_draw as ant_draw
from environments.utils.normalizer import Normalizer
from environments.diverse_maze.utils import sample_nearby_grid_location_v2

import numpy as np


def sample_location(uniform_obs, anchor=None):
    return uniform_obs[np.random.randint(0, len(uniform_obs))]


class EnvsGenerator:
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        min_block_radius: int,
        max_block_radius: int,
        stack_states: int = 1,
        image_obs: bool = True,
        data_path: str = None,
        trials_path: str = None,
        unique_shortest_path: bool = False,
        normalizer: Normalizer = None,
    ):
        self.env_name = env_name
        self.n_envs = n_envs
        self.min_block_radius = min_block_radius
        self.max_block_radius = max_block_radius
        self.stack_states = stack_states
        self.image_obs = image_obs
        self.data_path = data_path
        self.trials_path = trials_path
        self.unique_shortest_path = unique_shortest_path
        self.normalizer = normalizer
        self.metadata = torch.load(f"{data_path}/metadata.pt")

    def __call__(self):
        envs = []

        env_name = self.env_name
        n_envs = self.n_envs
        min_block_radius = self.min_block_radius
        max_block_radius = self.max_block_radius
        data_path = self.data_path
        trials_path = self.trials_path

        uniform_ds = load_uniform(env_name, data_path)

        if "diverse" in env_name:
            map_path = f"{data_path}/train_maps.pt"
            map_layouts = torch.load(map_path)
            map_keys = list(map_layouts.keys())

            if trials_path is not None and trials_path:
                trials = torch.load(trials_path)
                n_envs = min(n_envs, len(trials["map_layouts"]))

                for i in range(n_envs):
                    env, _, _, _ = self._make_env(
                        start=trials["starts"][i] if "starts" in trials else None,
                        min_block_radius=min_block_radius,
                        max_block_radius=max_block_radius,
                        map_idx=i,
                        map_key=trials["map_layouts"][i],
                        ood_dist=(
                            trials["ood_distance"][i]
                            if "ood_distance" in trials
                            else None
                        ),
                        mode=trials["mode"][i] if "mode" in trials else "train",
                        target=trials["targets"][i] if "targets" in trials else None,
                        block_dist=(
                            trials["block_dists"][i]
                            if "block_dists" in trials
                            else None
                        ),
                        turns=trials["turns"][i] if "turns" in trials else None,
                        observations=(
                            trials["observations"][i]
                            if "observations" in trials
                            else None
                        ),
                        actions=trials["actions"][i] if "actions" in trials else None,
                    )
                    envs.append(env)
            else:
                trials = {
                    "starts": [],
                    "targets": [],
                    "map_layouts": [],
                    "block_dists": [],
                    "turns": [],
                }

                if n_envs <= len(map_keys):
                    env_keys = map_keys[:n_envs]
                else:
                    env_keys = np.resize(map_keys, n_envs)

                for i, map_idx in enumerate(env_keys):
                    start = sample_location(uniform_ds[map_idx])[:2]

                    env, target, block_dist, turns = self._make_env(
                        start=start,
                        min_block_radius=min_block_radius,
                        max_block_radius=max_block_radius,
                        map_idx=map_idx,
                        map_key=map_layouts[map_idx],
                        # block_dist=block_dist,
                        # turns=turns,
                    )
                    envs.append(env)

                    trials["starts"].append(start)
                    trials["targets"].append(target)
                    trials["map_layouts"].append(map_layouts[map_idx])
                    trials["block_dists"].append(block_dist)
                    trials["turns"].append(turns)

                # torch.save(trials, f"{data_path}/trials.pt")
        else:
            for i in range(n_envs):
                # there's just one layout. we give it 0 id by convention
                map_idx = 0

                env = self._make_env(
                    start=sample_location(uniform_ds[map_idx]),
                    target=sample_location(uniform_ds[map_idx]),
                )
                envs.append(env)

        return envs
