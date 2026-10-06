"""Distill the adaptive-residual Teacher into a proprioception+RGB Student."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from student_teacher_utils import compute_residual_gate, configure_observation_groups, load_observation_config  # noqa: E402


parser = argparse.ArgumentParser(description="Distill the adaptive-residual Teacher into a visual Student.")
parser.add_argument("--task", default="OpenArm-Bimanual-Box-Stable-Lift-Compat-v0")
parser.add_argument("--nominal_checkpoint", required=True)
parser.add_argument("--residual_checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--obs_config", default=str(SCRIPT_DIR / "student_teacher_obs_visual.yaml"))
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--steps_per_iteration", type=int, default=24)
parser.add_argument("--epochs_per_iteration", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--replay_iterations", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--residual_scale", type=float, default=0.36)
parser.add_argument("--residual_lift_gate", type=float, default=0.06)
parser.add_argument("--residual_lift_gate_width", type=float, default=0.06)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# RGB sensors require Isaac Sim camera extensions even in headless mode.
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import openarm.tasks  # noqa: F401, E402


class VisualStudent(torch.nn.Module):
    """Small RGB encoder fused with the 50-D proprioceptive observation."""

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

    def forward(self, x):
        return (x - self.mean) / (self.std + 1.0e-2)

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        old_count = int(self.count.item())
        total = old_count + batch_count
        if old_count == 0:
            self.mean.copy_(batch_mean); self.std.copy_(torch.sqrt(batch_var + 1.0e-6)); self.count.fill_(batch_count); return
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        old_var = self.std.square()
        m2 = old_var * old_count + batch_var * batch_count + delta.square() * old_count * batch_count / total
        self.mean.copy_(new_mean); self.std.copy_(torch.sqrt(m2 / total + 1.0e-6)); self.count.fill_(total)


def _checkpoint(path: str) -> str:
    value = str(Path(path).expanduser().resolve())
    if not Path(value).is_file():
        raise FileNotFoundError(value)
    return value


def _camera_cfg():
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera", update_period=0.0, height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, focus_distance=1.0, horizontal_aperture=24.0, clipping_range=(0.05, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.20, 0.0, 0.55), rot=(0.6223, 0.3357, 0.3357, 0.6223), convention="opengl"),
    )


def _camera_rgb(raw_env, camera_name: str) -> torch.Tensor:
    camera = raw_env.scene[camera_name]
    output = camera.data.output
    if "rgb" not in output:
        raise RuntimeError(f"Camera '{camera_name}' has no rgb output. Available: {list(output.keys())}")
    image = output["rgb"]
    if image.ndim != 4:
        raise RuntimeError(f"Expected camera RGB tensor [N,H,W,C], got {tuple(image.shape)}")
    image = image[..., :3].float().permute(0, 3, 1, 2).contiguous()
    return image / 255.0 if image.max() > 1.5 else image


def _teacher_policy(env, agent_cfg, checkpoint):
    cfg = copy.deepcopy(agent_cfg.to_dict())
    cfg["obs_groups"] = {"policy": ["teacher"], "critic": ["teacher"]}
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=env.device)
    runner.load(checkpoint, load_optimizer=False, map_location=env.device)
    runner.eval_mode()
    return runner.get_inference_policy(device=env.device)


def _save(path, model, normalizer, iteration, metadata):
    torch.save({"student_state_dict": model.state_dict(), "student_obs_normalizer": normalizer.state_dict(), "iter": iteration, "metadata": metadata}, path)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args_cli.seed)
    nominal = _checkpoint(args_cli.nominal_checkpoint)
    residual = _checkpoint(args_cli.residual_checkpoint)
    output = Path(args_cli.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "student_final.pt").exists() and not args_cli.overwrite:
        raise FileExistsError(f"{output / 'student_final.pt'} exists; use another --output_dir or --overwrite")
    obs_cfg = load_observation_config(args_cli.obs_config)
    env_cfg.scene.num_envs = args_cli.num_envs; env_cfg.seed = args_cli.seed; env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    configure_observation_groups(env_cfg, obs_cfg)
    env_cfg.scene.front_camera = _camera_cfg()
    env_cfg.rerender_on_reset = True
    env_cfg.sim.render.antialiasing_mode = "OFF"
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv): gym_env = multi_agent_to_single_agent(gym_env)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    nominal_policy = _teacher_policy(env, agent_cfg, nominal)
    residual_policy = _teacher_policy(env, agent_cfg, residual)
    observations = env.get_observations(); proprio = observations["policy"]; teacher_obs = observations["teacher"]
    if proprio.shape[-1] != 50 or teacher_obs.shape[-1] != 98: raise RuntimeError(f"Expected policy=50, teacher=98; got {proprio.shape[-1]}, {teacher_obs.shape[-1]}")
    image = _camera_rgb(raw_env, obs_cfg.get("camera_name", "front_camera"))
    device = env.device
    student = VisualStudent(50, env.num_actions).to(device); normalizer = VectorNormalizer(50).to(device); optimizer = torch.optim.AdamW(student.parameters(), lr=args_cli.learning_rate, weight_decay=args_cli.weight_decay)
    metadata = {"synthetic_reference": False, "created_at_utc": datetime.now(timezone.utc).isoformat(), "task": args_cli.task, "teacher_checkpoints": "private (paths omitted)", "student_terms": obs_cfg["student_terms"], "teacher_terms": obs_cfg["teacher_terms"], "student_proprio_dim": 50, "camera_name": obs_cfg.get("camera_name", "front_camera"), "camera_shape_chw": [3, 240, 320], "num_actions": int(env.num_actions), "architecture": "VisualStudent(CNN+proprio)", "residual_scale": args_cli.residual_scale, "residual_lift_gate": args_cli.residual_lift_gate, "residual_lift_gate_width": args_cli.residual_lift_gate_width, "seed": args_cli.seed}
    output.joinpath("observation_config.yaml").write_text(yaml.safe_dump(obs_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    output.joinpath("distillation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replay = deque(maxlen=max(args_cli.replay_iterations, 1)); start_time = time.time(); observations = env.get_observations()
    try:
        for iteration in range(1, args_cli.max_iterations + 1):
            proprio_batches=[]; image_batches=[]; action_batches=[]
            with torch.inference_mode():
                for _ in range(args_cli.steps_per_iteration):
                    proprio_now = observations["policy"].detach(); image_now = _camera_rgb(raw_env, obs_cfg.get("camera_name", "front_camera")).detach()
                    nominal_action = nominal_policy(observations); residual_action = residual_policy(observations)
                    if agent_cfg.clip_actions is not None:
                        nominal_action=torch.clamp(nominal_action,-agent_cfg.clip_actions,agent_cfg.clip_actions); residual_action=torch.clamp(residual_action,-agent_cfg.clip_actions,agent_cfg.clip_actions)
                    gate=compute_residual_gate(env,args_cli.residual_lift_gate,args_cli.residual_lift_gate_width); action=nominal_action+gate.unsqueeze(-1)*args_cli.residual_scale*residual_action
                    if agent_cfg.clip_actions is not None: action=torch.clamp(action,-agent_cfg.clip_actions,agent_cfg.clip_actions)
                    proprio_batches.append(proprio_now); image_batches.append(image_now); action_batches.append(action.detach()); observations,_,_,_=env.step(action)
            batch=(torch.cat(proprio_batches),torch.cat(image_batches),torch.cat(action_batches)); replay.append(batch)
            train_proprio=torch.cat([x[0] for x in replay]); train_image=torch.cat([x[1] for x in replay]); train_action=torch.cat([x[2] for x in replay]); normalizer.train(); normalizer.update(batch[0]); normalizer.eval(); student.train(); permutation=torch.randperm(train_proprio.shape[0],device=device); losses=[]
            for _ in range(args_cli.epochs_per_iteration):
                for start in range(0,train_proprio.shape[0],args_cli.batch_size):
                    idx=permutation[start:start+args_cli.batch_size]; pred=student(normalizer(train_proprio[idx]),train_image[idx]); loss=torch.nn.functional.mse_loss(pred,train_action[idx]); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
            if iteration==1 or iteration%10==0: print(f"[Visual Student] iteration={iteration:05d} loss={sum(losses)/max(len(losses),1):.6f} samples={train_proprio.shape[0]} elapsed={(time.time()-start_time)/60.0:.1f}min")
            if iteration%args_cli.save_interval==0: _save(output/f"student_{iteration:06d}.pt",student,normalizer,iteration,metadata)
        _save(output/"student_final.pt",student,normalizer,args_cli.max_iterations,metadata); print(f"[Visual Student] saved: {output/'student_final.pt'}")
    finally: env.close()


if __name__ == "__main__":
    try: main()
    finally: simulation_app.close()
