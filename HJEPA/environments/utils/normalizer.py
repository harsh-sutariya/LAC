from typing import NamedTuple

import torch
from tqdm import tqdm
from environments.ogbench.utils import PixelMapper as VisualAntPixelMapper
from environments.diverse_maze.utils import PixelMapper as D4RLPixelMapper


def create_pixel_mapper(env_name):
    if "ant" in env_name:
        # TODO: maybe better way to distinguish ogbench?
        pixel_mapper = VisualAntPixelMapper(env_name=env_name)
    elif "diverse" in env_name or "maze2d" in env_name:
        pixel_mapper = D4RLPixelMapper(env_name=env_name)
    else:

        class IdPixelMapper:
            def obs_coord_to_pixel_coord(self, x):
                return x

            def pixel_coord_to_obs_coord(self, x):
                return x

        pixel_mapper = IdPixelMapper()

    return pixel_mapper


STATS = {
    "WallDataset": {
        # NOTE: these values are in fact not correct but they work cuz random search
        # was performed using these. When using correct values performance got worse.
        # but should just perform random search over again on correct values
        "state_mean": torch.tensor([0.0002, 0.0014]),
        "state_std": torch.tensor([0.0034, 0.0112]),
        "action_mean": torch.tensor([0.0120, -0.0074]),
        "action_std": torch.tensor([0.7543, 0.7424]),
        "location_mean": torch.tensor([31.1224, 31.3396]),
        "location_std": torch.tensor([16.3134, 16.6708]),
        "proprio_pos_mean": torch.tensor([0, 1]),
        "proprio_pos_std": torch.tensor([0, 1]),
        "proprio_vel_mean": torch.tensor([0, 1]),
        "proprio_vel_std": torch.tensor([0, 1]),
    }
}


def get_nth_percentile(tensor, percentile):
    assert len(tensor.shape) == 1
    k = int(tensor.shape[0] * percentile)
    return tensor.kthvalue(k).values.item()


class Sample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]


class Normalizer:
    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        l2_action_mean: torch.Tensor,
        l2_action_std: torch.Tensor,
        location_mean: torch.Tensor,
        location_std: torch.Tensor,
        proprio_pos_mean: torch.Tensor,
        proprio_pos_std: torch.Tensor,
        proprio_vel_mean: torch.Tensor,
        proprio_vel_std: torch.Tensor,
        min_max_state: bool = False,
        image_based: bool = True,
        pixel_mapper=None,
    ):
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.l2_action_mean = l2_action_mean
        self.l2_action_std = l2_action_std
        self.location_mean = location_mean
        self.location_std = location_std
        self.proprio_pos_mean = proprio_pos_mean
        self.proprio_pos_std = proprio_pos_std
        self.proprio_vel_mean = proprio_vel_mean
        self.proprio_vel_std = proprio_vel_std
        self.min_max_state = min_max_state
        self.image_based = image_based
        self.pixel_mapper = pixel_mapper

    @staticmethod
    def _has_attr(sample, attr):
        return (
            hasattr(sample, attr)
            and getattr(sample, attr) is not None
            and bool(getattr(sample, attr).shape[-1])
        )

    @classmethod
    def build_normalizer(
        cls,
        dataset,
        l2_step_skip: int = 4,
        n_samples: int = 100,
        min_max_state: bool = False,
        normalizer_hardset: bool = False,
        image_based: bool = True,
    ):
        all_actions = []
        all_l2_actions = []
        all_locations = []
        all_states = []
        all_proprio_pos = []
        all_proprio_vel = []

        config = (
            dataset.dataset.config if hasattr(dataset, "dataset") else dataset.config
        )

        it = iter(dataset)
        for _i in tqdm(range(n_samples), desc="estimating mean stds"):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(dataset)
                sample = next(it)

            if cls._has_attr(sample, "states"):
                if len(sample.states.shape) == 5:  # image BxTxCxHxW
                    states = sample.states.float()  # convert to float in case of byte

                    if min_max_state:
                        states -= states.amin(dim=(3, 4), keepdim=True)
                        states /= states.amax(dim=(3, 4), keepdim=True)

                    states = states.flatten(start_dim=-2).flatten(end_dim=-3)
                    # make states CxBTHW
                    states = states.permute(1, 0, 2).flatten(start_dim=1)
                else:  # proprio BxTxD
                    if min_max_state:
                        raise NotImplementedError(
                            "min_max_state not implemented for proprio"
                        )
                    states = sample.states.permute(2, 1, 0).flatten(start_dim=1)
            else:
                states = torch.zeros([1, 1])

            actions = sample.actions

            if config.chunked_actions and not config.substitute_action == "direction":
                bs, T, chunk_size, action_dim = actions.shape
            else:
                bs, T, action_dim = actions.shape

            T = T // l2_step_skip * l2_step_skip
            l2_actions = actions[:, :T]

            if config.chunked_actions and not config.substitute_action == "direction":
                l2_actions = l2_actions.view(
                    bs, T // l2_step_skip, l2_step_skip, chunk_size, action_dim
                ).sum(dim=2)
            else:
                l2_actions = l2_actions.view(
                    bs, T // l2_step_skip, l2_step_skip, action_dim
                ).sum(dim=2)

            l2_actions = l2_actions.view(-1, action_dim)

            actions = sample.actions.view(-1, action_dim)

            locations = sample.locations.view(-1, sample.locations.shape[-1])

            all_actions.append(actions)
            all_l2_actions.append(l2_actions)
            all_locations.append(locations)
            all_states.append(states)

            if cls._has_attr(sample, "proprio_pos"):
                proprio_pos = sample.proprio_pos.view(-1, sample.proprio_pos.shape[-1])
            else:
                proprio_pos = torch.zeros([1, 2])
            all_proprio_pos.append(proprio_pos)

            if cls._has_attr(sample, "proprio_vel"):
                proprio_vel = sample.proprio_vel.view(-1, sample.proprio_vel.shape[-1])
            else:
                proprio_vel = torch.zeros([1, 2])
            all_proprio_vel.append(proprio_vel)

        # just for calculating stats for action binding during planning
        # l2_act = torch.cat(all_l2_actions)
        # l1_act = torch.cat(all_actions)
        # print(f'0.002 percentile l1 angle: {get_nth_percentile(l1_act[:, 0], 0.002)}')
        # print(f'0.998 percentile l1 angle: {get_nth_percentile(l1_act[:, 0], 0.998)}')
        # print(f'0.002 percentile l1 norm: {get_nth_percentile(l1_act[:, 1], 0.002)}')
        # print(f'0.998 percentile l1 norm: {get_nth_percentile(l1_act[:, 1], 0.998)}')
        # print(f'0.002 percentile l2 angle: {get_nth_percentile(l2_act[:, 0], 0.002)}')
        # print(f'0.998 percentile l2 angle: {get_nth_percentile(l2_act[:, 0], 0.998)}')
        # print(f'0.002 percentile l2 norm: {get_nth_percentile(l2_act[:, 1], 0.002)}')
        # print(f'0.998 percentile l2 norm: {get_nth_percentile(l2_act[:, 1], 0.998)}')

        # l1_act = torch.cat(all_actions)
        # l2_act = torch.cat(all_l2_actions)
        # print(f'0.002 pcercentile l1 norm: {get_nth_percentile(torch.norm(l1_act, dim=-1), 0.002)}')
        # print(f'0.998 pcercentile l1 norm: {get_nth_percentile(torch.norm(l1_act, dim=-1), 0.998)}')
        # print(f'0.002 pcercentile l2 norm: {get_nth_percentile(torch.norm(l2_act, dim=-1), 0.002)}')
        # print(f'0.998 pcercentile l2 norm: {get_nth_percentile(torch.norm(l2_act, dim=-1), 0.998)}')

        if hasattr(dataset, "config") and normalizer_hardset:
            ds_stats = STATS[dataset.__class__.__name__]
            total_state_mean = ds_stats["state_mean"].to(locations.device)
            total_state_std = ds_stats["state_std"].to(locations.device)
            total_action_mean = ds_stats["action_mean"].to(locations.device)
            total_action_std = ds_stats["action_std"].to(locations.device)
            total_location_mean = ds_stats["location_mean"].to(locations.device)
            total_location_std = ds_stats["location_std"].to(locations.device)
            total_proprio_pos_mean = ds_stats["proprio_pos_mean"].to(locations.device)
            total_proprio_pos_std = ds_stats["proprio_pos_std"].to(locations.device)
            total_proprio_vel_mean = ds_stats["proprio_vel_mean"].to(locations.device)
            total_proprio_vel_std = ds_stats["proprio_vel_std"].to(locations.device)
        else:
            total_state = torch.cat(all_states, dim=-1)
            total_state_mean = total_state.mean(dim=-1)
            total_state_std = total_state.std(dim=-1)

            total_action = torch.cat(all_actions)
            total_action_mean = total_action.mean(dim=0)
            total_action_std = total_action.std(dim=0)

            total_location = torch.cat(all_locations)
            total_location_mean = total_location.mean(dim=0)
            total_location_std = total_location.std(dim=0)

            total_proprio_pos = torch.cat(all_proprio_pos)
            total_proprio_pos_mean = total_proprio_pos.mean(dim=0)
            total_proprio_pos_std = total_proprio_pos.std(dim=0)

            total_proprio_vel = torch.cat(all_proprio_vel)
            total_proprio_vel_mean = total_proprio_vel.mean(dim=0)
            total_proprio_vel_std = total_proprio_vel.std(dim=0)

        # no need to hardset l2 actions
        total_l2_action_mean = torch.cat(all_l2_actions).mean(dim=0)
        total_l2_action_std = torch.cat(all_l2_actions).std(dim=0)

        pixel_mapper = create_pixel_mapper(getattr(dataset.config, "env_name", ""))

        return cls(
            total_state_mean,
            total_state_std,
            total_action_mean,
            total_action_std,
            total_l2_action_mean,
            total_l2_action_std,
            total_location_mean,
            total_location_std,
            total_proprio_pos_mean,
            total_proprio_pos_std,
            total_proprio_vel_mean,
            total_proprio_vel_std,
            min_max_state=min_max_state,
            image_based=image_based,
            pixel_mapper=pixel_mapper,
        )

    @classmethod
    def build_id_normalizer(cls):
        return cls(
            state_mean=torch.zeros(1),
            state_std=torch.ones(1),
            action_mean=torch.zeros(1),
            action_std=torch.ones(1),
            l2_action_mean=torch.zeros(1),
            l2_action_std=torch.ones(1),
            location_mean=torch.zeros(1),
            location_std=torch.ones(1),
            proprio_pos_mean=torch.zeros(1),
            proprio_pos_std=torch.ones(1),
            proprio_vel_mean=torch.zeros(1),
            proprio_vel_std=torch.ones(1),
            min_max_state=False,
        )

    def min_max_normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) >= 3:
            state = state - state.amin(dim=(-2, -1), keepdim=True)
            state = state / (state.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            state = state - state.amin(dim=-1, keepdim=True)
            state = state / (state.amax(dim=-1, keepdim=True) + 1e-6)
        return state

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:

        if self.min_max_state:
            state = self.min_max_normalize_state(state)
        if self.image_based:  # if its image
            adapted_mean = self.state_mean.view(-1, 1, 1).to(state.device)
            adapted_std = self.state_std.view(-1, 1, 1).to(state.device) + 1e-6

            state_channels = state.shape[-3]  # [..., ch, w, h]

            # in case the stats are calculated over stacked obs, but state is unstacked:
            if state_channels < adapted_mean.shape[0] and not (
                adapted_mean.shape[0] % state_channels
            ):
                adapted_mean = adapted_mean[:state_channels]
                adapted_std = adapted_std[:state_channels]

            normalized_state = (state - adapted_mean) / adapted_std

            return normalized_state
        else:
            return (state - self.state_mean.to(state.device)) / self.state_std.to(
                state.device
            )

    def normalize_l2_action(self, l2_action: torch.Tensor) -> torch.Tensor:
        return (l2_action - self.l2_action_mean.to(l2_action.device)) / (
            self.l2_action_std.to(l2_action.device) + 1e-6
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean.to(action.device)) / (
            self.action_std.to(action.device) + 1e-6
        )

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )

    def normalize_proprio_pos(self, proprio_pos: torch.Tensor) -> torch.Tensor:
        return (proprio_pos - self.proprio_pos_mean.to(proprio_pos.device)) / (
            self.proprio_pos_std.to(proprio_pos.device) + 1e-6
        )

    def normalize_proprio_vel(self, proprio_vel: torch.Tensor) -> torch.Tensor:
        return (proprio_vel - self.proprio_vel_mean.to(proprio_vel.device)) / (
            self.proprio_vel_std.to(proprio_vel.device) + 1e-6
        )

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) >= 3:  # if it's image
            adapted_mean = self.state_mean.view(-1, 1, 1).to(state.device)
            adapted_std = self.state_std.view(-1, 1, 1).to(state.device)

            state_channels = state.shape[-3]  # [..., ch, w, h]

            # in case the stats are calculated over stacked obs, but state is unstacked:
            if state_channels < adapted_mean.shape[0] and not (
                adapted_mean.shape[0] % state_channels
            ):
                adapted_mean = adapted_mean[:state_channels]
                adapted_std = adapted_std[:state_channels]

            return state * adapted_std + adapted_mean
        else:
            return state * self.state_std.to(state.device) + self.state_mean.to(
                state.device
            )

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std.to(action.device) + self.action_mean.to(
            action.device
        )

    def unnormalize_l2_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.l2_action_std.to(action.device) + self.l2_action_mean.to(
            action.device
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return location * self.location_std.to(location.device) + self.location_mean.to(
            location.device
        )

    def unnormalize_proprio_pos(self, proprio_pos: torch.Tensor) -> torch.Tensor:
        return proprio_pos * self.proprio_pos_std.to(
            proprio_pos.device
        ) + self.proprio_pos_mean.to(proprio_pos.device)

    def unnormalize_proprio_vel(self, proprio_vel: torch.Tensor) -> torch.Tensor:
        return proprio_vel * self.proprio_vel_std.to(
            proprio_vel.device
        ) + self.proprio_vel_mean.to(proprio_vel.device)

    def normalize_sample(self, sample):
        replaced = {}
        if self._has_attr(sample, "states"):
            replaced["states"] = self.normalize_state(sample.states)
        if self._has_attr(sample, "locations"):
            replaced["locations"] = self.normalize_location(sample.locations)
        if self._has_attr(sample, "actions"):
            replaced["actions"] = self.normalize_action(sample.actions)
        if self._has_attr(sample, "goal"):
            replaced["goal"] = self.normalize_location(sample.goal)
        if self._has_attr(sample, "proprio_pos"):
            replaced["proprio_pos"] = self.normalize_proprio_pos(sample.proprio_pos)
        if self._has_attr(sample, "proprio_vel"):
            replaced["proprio_vel"] = self.normalize_proprio_vel(sample.proprio_vel)
        if self._has_attr(sample, "chunked_locations"):
            replaced["chunked_locations"] = self.normalize_location(
                sample.chunked_locations
            )
        if self._has_attr(sample, "chunked_proprio_pos"):
            replaced["chunked_proprio_pos"] = self.normalize_proprio_pos(
                sample.chunked_proprio_pos
            )
        if self._has_attr(sample, "chunked_proprio_vel"):
            replaced["chunked_proprio_vel"] = self.normalize_proprio_vel(
                sample.chunked_proprio_vel
            )

        if self._has_attr(sample, "l2_states"):
            replaced["l2_states"] = self.normalize_state(sample.l2_states)
        if self._has_attr(sample, "l2_locations"):
            replaced["l2_locations"] = self.normalize_location(sample.l2_locations)
        if self._has_attr(sample, "l2_proprio_pos"):
            replaced["l2_proprio_pos"] = self.normalize_proprio_pos(
                sample.l2_proprio_pos
            )
        if self._has_attr(sample, "l2_proprio_vel"):
            replaced["l2_proprio_vel"] = self.normalize_proprio_vel(
                sample.l2_proprio_vel
            )

        return sample._replace(**replaced)

    @torch.no_grad()
    def unnormalize_mse(self, mse, attribute="locations"):
        # unnormalize locations mse
        std_mapper = {
            "locations": self.location_std,
            "proprio_pos": self.proprio_pos_std,
            "proprio_vel": self.proprio_vel_std,
            "l2_locations": self.location_std,
            "l2_proprio_pos": self.proprio_pos_std,
            "l2_proprio_vel": self.proprio_vel_std,
        }

        return mse * std_mapper[attribute].to(mse.device) ** 2

    def to(self, device):
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        self.action_mean = self.action_mean.to(device)
        self.action_std = self.action_std.to(device)
        self.l2_action_mean = self.l2_action_mean.to(device)
        self.l2_action_std = self.l2_action_std.to(device)
        self.location_mean = self.location_mean.to(device)
        self.location_std = self.location_std.to(device)
        self.proprio_pos_mean = self.proprio_pos_mean.to(device)
        self.proprio_pos_std = self.proprio_pos_std.to(device)
        self.proprio_vel_mean = self.proprio_vel_mean.to(device)
        self.proprio_vel_std = self.proprio_vel_std.to(device)

    def save(self, path):
        torch.save(
            {
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "l2_action_mean": self.l2_action_mean,
                "l2_action_std": self.l2_action_std,
                "location_mean": self.location_mean,
                "location_std": self.location_std,
                "proprio_pos_mean": self.proprio_pos_mean,
                "proprio_pos_std": self.proprio_pos_std,
                "proprio_vel_mean": self.proprio_vel_mean,
                "proprio_vel_std": self.proprio_vel_std,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu")
        return cls(
            state["state_mean"],
            state["state_std"],
            state["action_mean"],
            state["action_std"],
            state["l2_action_mean"],
            state["l2_action_std"],
            state["location_mean"],
            state["location_std"],
            state["proprio_pos_mean"],
            state["proprio_pos_std"],
            state["proprio_vel_mean"],
            state["proprio_vel_std"],
        )
