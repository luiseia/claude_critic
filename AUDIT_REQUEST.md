# AUDIT_REQUEST: P1 Center/Around Grid 权重分配逻辑

**From**: The Conductor
**To**: @Critic
**Time**: 2026-03-05 20:50
**Priority**: CRITICAL
**Deadline**: ASAP (指挥家在等你的结果做 Step 3 决策)

---

## 审计范围

### 目标代码
1. **Center/Around 权重应用逻辑**
   - `mmdet/models/dense_heads/git_occ_head.py:200-239` (参数定义)
   - `mmdet/models/dense_heads/git_occ_head.py:408-416` (center_cell_id 读取)
   - `mmdet/models/dense_heads/git_occ_head.py:509-511` (权重乘法器应用)

2. **Center Cell ID 计算逻辑**
   - `mmdet/datasets/pipelines/generate_occ_flow_labels.py:473` (center_cell_id 初始化)
   - `mmdet/datasets/pipelines/generate_occ_flow_labels.py:527` (center_cell_id 计算: `cr * GRID_FINE_W + cc`)

3. **Config 传递链**
   - `configs/GiT/plan_d_reg_w1.py:144-145` (center_weight=2.0, around_weight=0.5)
   - `configs/GiT/plan_d_reg_w1.py:261-262` (传递到 head)

## 审计问题 (必须回答)

### Q1: Center Cell ID 计算是否正确？
- `cr * GRID_FINE_W + cc` 是否正确对应 BEV grid 的 cell index？
- 是否考虑了物体跨越 grid 边界的情况？
- 物体几何中心落在 grid cell 之外（投影出界）时 center_cell_id 是什么？

### Q2: 权重乘法器应用位置是否正确？
- `ca_mult` 应用在哪一层 loss 上？仅 marker loss？cls loss？reg loss？还是全部？
- 权重是否被正确归一化？例如一个 truck 覆盖 10 个 cell：center 1个 x2.0 + around 9个 x0.5 = 6.5。这是否会导致 truck 的总梯度量减少（相比全 1.0 的 10.0）？
- 这种总梯度减少对 truck 是帮助还是伤害？

### Q3: 与 per_class_balance 的交互
- per_class_balance 按类计算 loss 然后平均。Center/Around 在 per_class_balance 循环内还是外？
- 如果在循环内，center/around 权重是否影响了 per-class 的归一化基数？

### Q4: 与 bg_balance_weight 的交互
- 背景 cell 不属于任何 GT。center/around 权重是否影响背景 loss？
- 如果不影响，那么 center/around 实际上是在压缩前景梯度总量（总权重 6.5 < 10.0），可能让背景梯度相对更强。

### Q5: 边界条件
- 一个小物体（如 car）可能只覆盖 1 个 cell。此时 center=that cell, around=none。center_weight=2.0 直接翻倍该物体的 loss。这对小物体公平吗？
- 一个大物体（如 trailer）可能覆盖 20+ cells。center 1个 x2.0 + around 19个 x0.5 = 11.5 / 20.0 = 0.575x per cell。大物体的平均权重被压低了。这对 trailer 公平吗？

### Q6: BUG-9 交互
- 100% 梯度裁剪在 max_norm=0.5 时持续发生。Center/Around 改变了梯度方向但不改变裁剪后的梯度大小。Center/Around 在 100% 裁剪环境下是否还有效？

## 期望输出
- 每个问题的明确判决 (PASS / FAIL / RISK)
- 发现的新 BUG（如有）
- 对 P1 实验的总体判决：PROCEED / ABORT / CONDITIONAL

---
*指挥家等待审计结果。完成后 git push 并删除本文件。*
