# Student 蒸馏发布包

本目录发布的是视觉 Student 的推理权重、蒸馏脚本、回放脚本和评估脚本。Student 使用 50 维本体观测（`joint_pos`、`joint_vel`、上一时刻动作）与 320×240 RGB 图像，输出 14 维动作；网络结构为 CNN 与本体状态分支融合的 `VisualStudent`。

## 目录

- `student/visual/student_final.pt`：最终视觉 Student 权重。权重内部的 Teacher 路径 metadata 已清理。
- `student/visual/observation_config.yaml`：Student 观测与相机配置。
- `student/visual/distillation_metadata.json`：不含本地路径和实验结果的公开元数据。
- `code/distill_adaptive_residual_visual.py`：蒸馏代码。运行蒸馏时由使用者通过命令行提供自己的 Teacher checkpoint。
- `code/evaluate_visual_student.py`：视觉 Student 评估代码。评估输出由使用者指定到本地目录，不随本发布包提供。
- `code/play_visual_student.py`：视觉 Student 回放代码。
- `code/student_teacher_utils.py`、`code/student_teacher_obs_visual.yaml`：观测分组与辅助配置。

## 使用边界

本发布包不包含 Teacher 模型、Teacher 权重、Student 评估结果、训练日志、真机原始日志、私有标定数据、评估 JSON/CSV/JSONL、缓存目录或压缩归档。评估代码中保留了 Teacher 参考动作的计算接口，因此完整的 Student--Teacher 对比评估需要使用者自行提供可访问的 Teacher checkpoint；本包不声称能够仅凭公开文件重建 Teacher。

蒸馏和评估脚本依赖原项目的 Isaac Lab、Isaac Sim、RSL-RL 以及 OpenArm 任务注册。脚本中的所有路径均应由使用者按本机环境通过命令行参数指定，不应把本机路径写入公开仓库。

## 基本命令示例

```bash
# 回放 Student
python code/play_visual_student.py \
  --task OpenArm-Bimanual-Box-Stable-Lift-Compat-Play-v0 \
  --checkpoint student/visual/student_final.pt \
  --obs_config student/visual/observation_config.yaml \
  --num_envs 1 --real-time

# 评估 Student（结果写到本地临时目录，不要将结果目录加入发布包）
python code/evaluate_visual_student.py \
  --student_checkpoint student/visual/student_final.pt \
  --nominal_checkpoint /path/to/private/teacher_nominal.pt \
  --residual_checkpoint /path/to/private/teacher_residual.pt \
  --obs_config student/visual/observation_config.yaml \
  --output_dir /tmp/student_eval
```

评估结果是否公开由发布者另行决定；本目录仅发布生成评估结果所需的代码，不附带任何既有评估数字。
