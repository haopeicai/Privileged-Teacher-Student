"""Play a front-camera RGB + proprioceptive Student checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from student_teacher_utils import configure_observation_groups, load_observation_config  # noqa: E402


parser = argparse.ArgumentParser(description="Play a visual Student checkpoint.")
parser.add_argument("--task", default="OpenArm-Bimanual-Box-Stable-Lift-Compat-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--obs_config", default=str(SCRIPT_DIR / "student_teacher_obs_visual.yaml"))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--real-time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import openarm.tasks  # noqa: F401, E402


class VisualStudent(torch.nn.Module):
    def __init__(self, proprio_dim: int, num_actions: int):
        super().__init__()
        self.visual = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=2), torch.nn.ELU(),
            torch.nn.Conv2d(16, 32, 5, stride=2), torch.nn.ELU(),
            torch.nn.Conv2d(32, 64, 5, stride=2), torch.nn.ELU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)), torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 128), torch.nn.ELU(),
        )
        self.proprio = torch.nn.Sequential(
            torch.nn.Linear(proprio_dim, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(192, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, num_actions),
        )

    def forward(self, proprio: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.proprio(proprio), self.visual(image)), dim=-1))


class VectorNormalizer(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1.0e-2)


def _camera_cfg():
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=1.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.20, 0.0, 0.55),
            rot=(0.6223, 0.3357, 0.3357, 0.6223),
            convention="opengl",
        ),
    )


def _camera_rgb(raw_env, camera_name: str) -> torch.Tensor:
    output = raw_env.scene[camera_name].data.output
    if "rgb" not in output:
        raise RuntimeError(f"Camera '{camera_name}' has no RGB output; available: {list(output.keys())}")
    image = output["rgb"]
    if image.ndim != 4:
        raise RuntimeError(f"Expected camera tensor [N,H,W,C], got {tuple(image.shape)}")
    image = image[..., :3].float().permute(0, 3, 1, 2).contiguous()
    return image / 255.0 if image.max() > 1.5 else image


def main() -> None:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    obs_config = load_observation_config(args_cli.obs_config)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = metadata.get("seed", 42)
    configure_observation_groups(env_cfg, obs_config)
    env_cfg.scene.front_camera = _camera_cfg()
    env_cfg.rerender_on_reset = True
    env_cfg.sim.render.antialiasing_mode = "OFF"

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=1.0)
    raw_env = env.unwrapped

    observations = env.get_observations()
    proprio_dim = int(observations["policy"].shape[-1])
    if proprio_dim != metadata["student_proprio_dim"]:
        raise RuntimeError(
            f"Student proprio dimension mismatch: checkpoint={metadata['student_proprio_dim']}, env={proprio_dim}"
        )
    model = VisualStudent(proprio_dim, env.num_actions).to(env.device)
    normalizer = VectorNormalizer(proprio_dim).to(env.device)
    model.load_state_dict(payload["student_state_dict"])
    normalizer.load_state_dict(payload["student_obs_normalizer"])
    model.eval()
    normalizer.eval()
    dt = float(raw_env.step_dt)

    try:
        while simulation_app.is_running():
            start = time.time()
            with torch.inference_mode():
                rgb = _camera_rgb(raw_env, metadata.get("camera_name", "front_camera"))
                actions = model(normalizer(observations["policy"]), rgb)
                actions = torch.clamp(actions, -1.0, 1.0)
                observations, _, _, _ = env.step(actions)
            if args_cli.real_time:
                remaining = dt - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
