"""Evaluate the distilled Student under the same fixed condition as Teacher.

This evaluator mirrors the stable-lift task's success/termination bookkeeping and
also reports Student-vs-composed-Teacher action MSE/MAE on the Student rollout.
It is intentionally kept in the artifact directory so the external OpenArm
repository does not need to be modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from student_teacher_utils import configure_observation_groups, compute_residual_gate, load_observation_config  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluate Student with fixed stable-lift conditions.")
parser.add_argument("--task", default="OpenArm-Bimanual-Box-Stable-Lift-Compat-v0")
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--nominal_checkpoint", required=True)
parser.add_argument("--residual_checkpoint", required=True)
parser.add_argument("--obs_config", default=str(SCRIPT_DIR / "student_teacher_obs_visual.yaml"))
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episodes_per_seed", type=int, default=20)
parser.add_argument("--eval_seed", type=int, default=42)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=30.0,
    help="Episode timeout in seconds; success criteria remain unchanged.",
)
parser.add_argument("--base_box_mass_kg", type=float, default=0.14)
parser.add_argument("--residual_scale", type=float, default=0.36)
parser.add_argument("--residual_lift_gate", type=float, default=0.06)
parser.add_argument("--residual_lift_gate_width", type=float, default=0.06)
parser.add_argument("--steady_lift_threshold_m", type=float, default=0.08)
parser.add_argument("--reference_lift_height_m", type=float, default=0.14)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
command_argv = sys.argv.copy()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import openarm.tasks  # noqa: F401, E402
from openarm.tasks.manager_based.openarm_manipulation.bimanual.box_stable_lift.mdp.events import (  # noqa: E402
    randomize_box_color_and_mass_on_reset,
)


class VisualStudent(torch.nn.Module):
    """CNN + proprioceptive Student architecture used by the released checkpoint."""

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
        prim_path="{ENV_REGEX_NS}/FrontCamera", update_period=0.0, height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, focus_distance=1.0, horizontal_aperture=24.0,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.20, 0.0, 0.55), rot=(0.6223, 0.3357, 0.3357, 0.6223),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_teacher_policy(env, agent_cfg, checkpoint: Path):
    cfg = copy.deepcopy(agent_cfg.to_dict())
    cfg["obs_groups"] = {"policy": ["teacher"], "critic": ["teacher"]}
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=env.device)
    runner.load(str(checkpoint), load_optimizer=False, map_location=env.device)
    runner.eval_mode()
    return runner.get_inference_policy(device=env.device)


def _force_fixed_mass(raw_env, mass: float) -> None:
    env_ids = torch.arange(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
    randomize_box_color_and_mass_on_reset(raw_env, env_ids, mass_range=(mass, mass))
    metadata = getattr(raw_env, "openarm_bimanual_box_randomization", None)
    if metadata is None or "object_mass" not in metadata:
        raise RuntimeError("Task did not expose object mass metadata.")
    if not torch.allclose(metadata["object_mass"], torch.full_like(metadata["object_mass"], mass), atol=1e-6, rtol=0.0):
        raise RuntimeError("Fixed mass assignment failed.")


def _state(raw_env):
    obj = raw_env.scene["object"]
    lift = obj.data.root_pos_w[:, 2] - obj.data.default_root_state[:, 2]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(obj.data.root_quat_w)
    tilt = torch.maximum(torch.abs(math_utils.wrap_to_pi(roll)), torch.abs(math_utils.wrap_to_pi(pitch)))
    return lift.detach(), tilt.detach()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.episodes_per_seed <= 0 or args_cli.episodes_per_seed > args_cli.num_envs:
        raise ValueError("episodes_per_seed must be in [1, num_envs]")
    if args_cli.episode_length_s <= 0.0:
        raise ValueError("episode_length_s must be positive")
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    student_path = Path(args_cli.student_checkpoint).expanduser().resolve()
    nominal_path = Path(args_cli.nominal_checkpoint).expanduser().resolve()
    residual_path = Path(args_cli.residual_checkpoint).expanduser().resolve()
    for path in (student_path, nominal_path, residual_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    obs_config = load_observation_config(args_cli.obs_config)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.eval_seed
    env_cfg.episode_length_s = args_cli.episode_length_s
    configure_observation_groups(env_cfg, obs_config)
    env_cfg.scene.front_camera = _camera_cfg()
    env_cfg.rerender_on_reset = True
    env_cfg.sim.render.antialiasing_mode = "OFF"
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = vec_env.unwrapped

    student_payload = torch.load(student_path, map_location="cpu", weights_only=False)
    metadata = student_payload["metadata"]
    obs = vec_env.get_observations()
    student = VisualStudent(metadata["student_proprio_dim"], vec_env.num_actions).to(vec_env.device)
    student_normalizer = VectorNormalizer(metadata["student_proprio_dim"]).to(vec_env.device)
    student.load_state_dict(student_payload["student_state_dict"])
    student_normalizer.load_state_dict(student_payload["student_obs_normalizer"])
    student.eval()
    student_normalizer.eval()
    teacher_nominal = _load_teacher_policy(vec_env, agent_cfg, nominal_path)
    teacher_residual = _load_teacher_policy(vec_env, agent_cfg, residual_path)

    # Match evaluate_stable_lift.py: effective seed = eval_seed * 1000 + batch_index(0).
    effective_env_seed = args_cli.eval_seed * 1000
    vec_env.seed(effective_env_seed)
    obs, _ = vec_env.reset()
    _force_fixed_mass(raw_env, args_cli.base_box_mass_kg)
    completed = torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)
    accum = [
        {"steps": 0, "reward": 0.0, "max_lift": -math.inf, "max_tilt": 0.0, "mse": [], "mae": []}
        for _ in range(raw_env.num_envs)
    ]
    rows = []
    max_steps = int(raw_env.max_episode_length) + 5
    with torch.inference_mode():
        for _ in range(max_steps):
            lift, tilt = _state(raw_env)
            for index in range(raw_env.num_envs):
                if not completed[index]:
                    accum[index]["steps"] += 1
                    accum[index]["max_lift"] = max(accum[index]["max_lift"], float(lift[index]))
                    accum[index]["max_tilt"] = max(accum[index]["max_tilt"], float(tilt[index]))
            teacher_nominal_action = teacher_nominal(obs)
            teacher_residual_action = teacher_residual(obs)
            teacher_action = teacher_nominal_action + compute_residual_gate(
                vec_env, args_cli.residual_lift_gate, args_cli.residual_lift_gate_width
            ).unsqueeze(-1) * args_cli.residual_scale * teacher_residual_action
            teacher_action = torch.clamp(teacher_action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            rgb = _camera_rgb(raw_env, metadata.get("camera_name", "front_camera"))
            student_action = torch.clamp(
                student(student_normalizer(obs["policy"]), rgb),
                -agent_cfg.clip_actions,
                agent_cfg.clip_actions,
            )
            error = student_action - teacher_action
            for index in range(raw_env.num_envs):
                if not completed[index]:
                    accum[index]["mse"].append(float(torch.mean(error[index] ** 2)))
                    accum[index]["mae"].append(float(torch.mean(torch.abs(error[index]))))
            obs, rewards, dones, _ = vec_env.step(student_action)
            for index in range(raw_env.num_envs):
                if not completed[index]:
                    accum[index]["reward"] += float(rewards[index])
            done_mask = dones.bool()
            if torch.any(done_mask):
                flags = {name: raw_env.termination_manager.get_term(name).detach().clone() for name in raw_env.termination_manager.active_terms}
                for index in torch.logical_and(done_mask, ~completed).nonzero(as_tuple=True)[0].cpu().tolist():
                    active = sorted(name for name, values in flags.items() if bool(values[index]))
                    success = bool(flags.get("success", torch.zeros_like(done_mask))[index])
                    explicit_timeout = bool(flags.get("time_out", torch.zeros_like(done_mask))[index])
                    timeout = explicit_timeout and not any(name != "time_out" for name in active)
                    rows.append({
                        "episode": len(rows),
                        "success": success,
                        "timeout": timeout,
                        "drop": bool(flags.get("box_dropped", torch.zeros_like(done_mask))[index]) or bool(flags.get("box_fell_after_lift", torch.zeros_like(done_mask))[index]),
                        "safety_termination": any(name in {"arm_collision", "joint_out_of_limits"} for name in active),
                        "terminated_reason": active,
                        "steps": accum[index]["steps"],
                        "duration_s": accum[index]["steps"] * float(raw_env.step_dt),
                        "reward_sum": accum[index]["reward"],
                        "max_lift_mm": accum[index]["max_lift"] * 1000.0,
                        "max_tilt_deg": math.degrees(accum[index]["max_tilt"]),
                        "teacher_action_mse": sum(accum[index]["mse"]) / max(len(accum[index]["mse"]), 1),
                        "teacher_action_mae": sum(accum[index]["mae"]) / max(len(accum[index]["mae"]), 1),
                    })
                completed |= done_mask
            if bool(torch.all(completed)):
                break
    if len(rows) < args_cli.episodes_per_seed:
        raise RuntimeError(f"Only completed {len(rows)} episodes; expected {args_cli.episodes_per_seed}.")
    rows = rows[: args_cli.episodes_per_seed]
    for row in rows:
        row.update({"method_id": "student", "task": args_cli.task, "eval_seed": args_cli.eval_seed, "mass_kg": args_cli.base_box_mass_kg})
    (output_dir / "episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "method_id": "student",
        "task": args_cli.task,
        "eval_seed": args_cli.eval_seed,
        "effective_env_seed": effective_env_seed,
        "mass_kg": args_cli.base_box_mass_kg,
        "episodes": len(rows),
        "success_count": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / len(rows),
        "timeout_rate": sum(int(row["timeout"]) for row in rows) / len(rows),
        "drop_rate": sum(int(row["drop"]) for row in rows) / len(rows),
        "safety_termination_rate": sum(int(row["safety_termination"]) for row in rows) / len(rows),
        "mean_episode_steps": sum(row["steps"] for row in rows) / len(rows),
        "mean_reward": sum(row["reward_sum"] for row in rows) / len(rows),
        "mean_teacher_action_mse": sum(row["teacher_action_mse"] for row in rows) / len(rows),
        "mean_teacher_action_mae": sum(row["teacher_action_mae"] for row in rows) / len(rows),
        "checkpoints": {"student": {"sha256": _sha256(student_path)}, "teacher": "private (paths omitted)"},
        "command": "omitted from published artifacts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
