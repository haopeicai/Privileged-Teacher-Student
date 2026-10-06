# Visual Student 文档

本目录说明公开的视觉 Student 模型、蒸馏流程和评估脚本。公开包只包含 Student 权重、必要观测配置和代码；Teacher 模型、Teacher 权重、既有评估结果、训练日志、真机原始日志和私有标定数据不在发布范围内。

## 文档

- [Student 原理与数据约定](Student原理与数据约定.md)
- [环境依赖与运行示例](环境依赖与运行示例.md)

## 快速索引

- 已发布权重：`../student/visual/student_final.pt`
- 观测配置：`../student/visual/observation_config.yaml`
- 回放脚本：`../code/play_visual_student.py`
- 蒸馏脚本：`../code/distill_adaptive_residual_visual.py`
- 评估脚本：`../code/evaluate_visual_student.py`

命令中的 `OPEN_SOURCE_ROOT` 表示公开包根目录，`ISAACLAB_ROOT` 表示外部 Isaac Lab/OpenArm 工作区。所有命令都应在已经完成 Isaac Lab 和 OpenArm 任务注册的运行环境中执行。
