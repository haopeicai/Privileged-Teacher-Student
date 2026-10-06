"""Small, Isaac-Lab-independent helpers shared by the distillation scripts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("student_teacher_obs.yaml")


def load_observation_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the observation split configuration."""

    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    student_terms = config.get("student_terms")
    if not isinstance(student_terms, list) or not student_terms:
        raise ValueError("student_terms must be a non-empty YAML list")
    teacher_terms = config.get("teacher_terms", "all")
    if teacher_terms != "all" and (not isinstance(teacher_terms, list) or not teacher_terms):
        raise ValueError("teacher_terms must be 'all' or a non-empty YAML list")
    config["student_terms"] = [str(term) for term in student_terms]
    config["teacher_terms"] = teacher_terms if teacher_terms == "all" else [str(term) for term in teacher_terms]
    # Keep generated metadata portable and free of local filesystem paths.
    config["config_path"] = config_path.name
    return config


def configure_observation_groups(env_cfg: Any, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Split the task's original policy observation group into policy and teacher groups.

    The original task policy group is copied before filtering. This preserves exactly the
    privileged input expected by the private Teacher checkpoints for the teacher group.
    The Student group is filtered to the terms explicitly listed in the YAML config.
    """

    original_policy = copy.deepcopy(env_cfg.observations.policy)
    teacher_group = copy.deepcopy(original_policy)
    student_group = copy.deepcopy(original_policy)

    all_terms = []
    for name, value in original_policy.__dict__.items():
        if name in {
            "enable_corruption",
            "concatenate_terms",
            "history_length",
            "flatten_history_dim",
            "concatenate_dim",
        }:
            continue
        if value is not None:
            all_terms.append(name)

    teacher_terms = config.get("teacher_terms", "all")
    if teacher_terms == "all":
        resolved_teacher_terms = all_terms
    else:
        resolved_teacher_terms = list(teacher_terms)

    student_terms = list(config["student_terms"])
    unknown_student = sorted(set(student_terms).difference(all_terms))
    unknown_teacher = sorted(set(resolved_teacher_terms).difference(all_terms))
    if unknown_student:
        raise ValueError(f"Unknown Student observation terms: {unknown_student}; available: {all_terms}")
    if unknown_teacher:
        raise ValueError(f"Unknown Teacher observation terms: {unknown_teacher}; available: {all_terms}")

    for name in all_terms:
        if name not in student_terms:
            setattr(student_group, name, None)
        if name not in resolved_teacher_terms:
            setattr(teacher_group, name, None)

    # ObservationGroupCfg is deliberately a normal config object in Isaac Lab, and the
    # manager supports multiple groups on the same environment.
    env_cfg.observations.policy = student_group
    env_cfg.observations.teacher = teacher_group
    return student_terms, resolved_teacher_terms


def compute_residual_gate(env: Any, lift_gate: float, gate_width: float):
    """Compute the same object-lift residual gate used by adaptive playback/training."""

    import torch

    object_asset = env.unwrapped.scene["object"]
    lift = object_asset.data.root_pos_w[:, 2] - object_asset.data.default_root_state[:, 2]
    return torch.clamp((lift - lift_gate) / max(gate_width, 1.0e-6), 0.0, 1.0)
