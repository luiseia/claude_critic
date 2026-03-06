# 首席批判官审计报告：GiT 3D 占用预测项目

> **审计目标**: /home/UNT/yz0370/projects/GiT/
> **审计时间**: 2026-03-05
> **审计员**: Claude Opus 4.6 (首席批判官)
> **结论**: 研究员 1 号的实验存在 **至少 7 个未被发现的代码 bug**、**3 个逻辑谬误** 和 **多项缺失的消融实验**。以下每一条都可能是精度瓶颈的根本原因。

---

## SECTION A: 致命代码缺陷 (CRITICAL BUGS)

### BUG-1: theta_fine 的周期性 Loss 完全错误
**文件**: `git_occ_head.py:694`
**代码**:
```python
periodic_diff = torch.min(abs_diff, float(bin_count) - abs_diff)
```
**问题**: `_calculate_theta_loss` 对 theta_fine（0-9，表示组内 0-9 度）也使用了周期性损失。theta_fine 的 bin_count=10，所以 `min(|pred-target|, 10-|pred-target|)` 会把 **0 和 9 视为相邻**（周期距离=1），但实际上 0 和 9 在组内相差 9 度，是最远的两个端点。

**后果**: 模型在 theta_fine 维度上收到错误的梯度信号。预测值 0 和目标值 9 的 loss 仅为 1/10=0.1（应该是 9/10=0.9）。这直接腐蚀了角度回归精度。

**反驳研究员**: theta_group 的周期性是正确的（0 度和 350 度确实相邻），但研究员把同一个函数复用到了 theta_fine 上，没有区分这两者的本质区别。

---

### BUG-2: Per-class Balanced Loss 系统性压制背景梯度 -- 精度瓶颈的真正根因

**文件**: `git_occ_head.py:907-922` (CE mode marker loss)
**代码**:
```python
for c in range(self.num_classes):  # 4 个前景类
    mask_c = is_real_car_gt & (slot_class == c)
    ...
bg_mask = is_bg_gt & (marker_weight > 0)
if bg_mask.any():
    loss_marker = loss_marker + bg_loss
    n_cls_active += 1
# 最终: loss_marker = total / n_cls_active  (n_cls_active = 4+1 = 5)
```

**问题**: Plan B 的 per-class balanced loss 将背景视为与每个前景类等权的"第 5 类"。背景仅获得 1/5 = 20% 的梯度贡献。但背景槽占所有 1200 个槽的 ~75%！

**在 Plan B 之前**: 背景 loss 自然占据了梯度的 ~75%（按样本频率加权），模型有强烈的激励去正确预测 END marker。

**在 Plan B 之后**: 背景仅占 20%，模型对"此处无车"的判断变得迟钝。

**证据链**: false_alarm_rate 从 iter 500 的 0.194 **持续上升**到 iter 5000 的 0.317。Plan B 越训越多，false alarm 越高。研究员认为 Plan B "成功解决了振荡"，但他完全忽视了 false alarm 在持续恶化这一致命信号。

**后果**: 这是 avg_precision ~0.09（远低于目标 0.20）的核心原因之一。研究员花了整夜追踪 precision 问题，先归咎于推理端，后来承认是训练端，但始终没有找到这个根因。

---

### BUG-3: Score 传播链断裂 -- 整个"推理修复"是一场空

**文件**: `git_occ_head.py:1466` + `occ_2d_box_eval.py:73`

**研究员在 01:30 部署的"推理精度修复"**:
1. 在 `decoder_inference` 中计算 `grid_scores`
2. 在 `predict()` 中通过 `kwargs.get('grid_scores', None)` 传递
3. 在 `add_pred_to_datasample` 中注入 `sample.pred_grid_scores`
4. 在评估器中通过 `sample.get('pred_grid_scores', None)` 读取

**断裂点**: `predict()` 的 `grid_scores` 依赖调用者通过 `**kwargs` 传入。但如果测试管道（test.py）在调用 `predict()` 时没有显式传递 `grid_scores=xxx`，则 `kwargs.get('grid_scores', None)` 永远返回 None。

**研究员自己的证据** (14:51 快报): "p_scores 在整个 Plan B 训练验证期间一直是 None"。这证明 score_thr=0.3 **从未生效过**。iter 3500-10000 的所有"新评估器"结果，实际上和没有 score filtering 时完全一样。

**浪费的时间**: 研究员在 01:10-01:45 花了 35 分钟"修复"推理端 precision，又在 14:06-14:51 花了 45 分钟才发现修复从未生效。总计 ~80 分钟消耗在一个从未工作的 feature 上。

---

### BUG-4: 深度排序使用相机系 Z 轴而非 BEV 距离

**文件**: `generate_occ_flow_labels.py:487`
```python
depth_val = float(corners_cam[:, 2].mean())  # 相机系 Z 轴
```

**问题**: Marker token (NEAR/MID/FAR) 用于表示 BEV 空间中从近到远的排列。但排序依据是**相机坐标系的 Z 轴**（相机光轴方向），而不是 **BEV 空间的欧氏距离**。

对于非正前方的物体（例如在相机视角边缘的车），相机 Z 值较大但 BEV 距离可能较近。反之，正前方近处的物体相机 Z 小但 BEV 距离可能也小。这导致同一个 grid cell 内 slot 的 NEAR/MID/FAR 分配与 BEV 真实远近不一致。

**后果**: slot-aligned 评估中，如果预测的深度排序和 GT 不一致，正确的检测在错误的 slot 上 -> 同时产生 FP 和 FN。

---

### BUG-5: 投影公式在 Z 极小值时行为不一致

**文件**: `generate_occ_flow_labels.py:494-496`
```python
safe_Z = np.where(corners_cam[:, 2] < 1e-3, 1e-3, corners_cam[:, 2])
u_all = (K[0, 0] * corners_cam[:, 0] + K[0, 2] * corners_cam[:, 2]) / safe_Z
```

**问题**: 分母用 safe_Z（clamp 后），但分子中 `K[0, 2] * corners_cam[:, 2]` 仍使用原始 Z。当 Z 在 (0, 1e-3) 区间时，分子用真实微小值，分母用 1e-3，导致投影偏差。

---

### BUG-6: slot_class 对背景 slot 的错误赋值

**文件**: `git_occ_head.py:819`
```python
slot_class = (tgt_flat[:, 1] - cls_start).clamp(0, self.num_classes - 1)
```

**问题**: 背景 slot 的 `tgt_flat[:, 1]` 是 `ignore_id` (223)。计算结果: `(223 - 168).clamp(0, 3) = 3`。背景 slot 被错误地标记为 class 3 (trailer)。虽然 `is_real_car_gt` mask 在 marker loss 中过滤了背景，但如果任何下游逻辑使用 slot_class 而不先检查 is_real_car_gt，trailer 类的统计会被污染。

---

### BUG-7: loss_reg 中 theta 的 magic number 无依据

**文件**: `git_occ_head.py:995`
```python
loss_reg = l_gx + l_gy + l_dx + l_dy + l_w + l_h + (l_th_group * 1.2 + l_th_fine)
```

**问题**: theta_group 的权重被硬编码为 1.2x，而其他所有回归项权重为 1.0。没有任何文档、注释或消融实验解释为什么是 1.2 而不是 1.0 或 1.5 或 2.0。

---

## SECTION B: 逻辑谬误与错误结论

### 谬误-1: "震荡打破" -- 3 个 checkpoint 就敢下结论

**研究员在 Check #17 声称**: "OSCILLATION BROKEN -- Truck recall STABLE at 0.25-0.26 for 3 consecutive checkpoints"

**500 iter 后的现实**: iter 2500 时 truck 从 0.35 暴跌到 0.057。"震荡打破"只是延迟了震荡。

**统计学常识**: 3 个数据点无法证明趋势稳定性。baseline 的震荡周期约 1000 iter，至少需要 2000 iter（4 个 checkpoint）无振荡才能谨慎地说"缓解"，而非"打破"。

---

### 谬误-2: "精度是推理端问题" -- 被自己推翻的结论

**01:10**: "Precision Is an Inference/Evaluation Problem, Not a Training Problem"
**14:51 (13 小时后)**: "精度低是训练端问题，推理后处理无法解决"

这 13 小时内，研究员基于错误结论：编写修复代码 -> 重启训练 -> 等 7 小时 -> debug 发现修复无效 -> 推翻自己。

**成本**: ~10000 iter GPU 时间 + 13 小时实验周期。一个 10 分钟的 debug print 本可以避免全部浪费。

---

### 谬误-3: "Plan B 是正确方向" -- 忽视引入的新问题

研究员将 Plan B 定性为"成功解决类竞争"。但完全忽视了:

1. **false_alarm_rate 持续上升** (0.194 -> 0.317)，Plan B 在牺牲背景判别能力
2. **Precision 始终低于 0.20**，无改善趋势
3. **avg_recall 最佳仅 0.675**，未达到 0.70 目标
4. **iter 5000 后所有指标持续下行**，模型没有收敛到更优点

Plan B 解决了一个问题（前景类振荡），但创造了另一个问题（背景被压制），研究员只看到了前者。

---

## SECTION C: 缺失的消融实验

| # | 缺失的实验 | 重要性 | 潜在影响 |
|---|-----------|--------|---------|
| 1 | pos_cls_w_multiplier 系统消融 (2/4/5/10/20) | 致命 | 直接控制 P/R trade-off |
| 2 | 背景权重在 per-class balance 中的比例 | 致命 | 可能同时解决 recall 均衡和 precision |
| 3 | theta_fine 周期性 vs 非周期性 | 中 | 量化 BUG-1 的影响 |
| 4 | slot-aligned vs cell-only 评估 | 中 | 揭示评估指标本身是否是瓶颈 |
| 5 | LR schedule (延迟衰减/温和衰减/Cosine) | 中 | iter 5000 后模型退化 |
| 6 | BEV grid 分辨率 (10x10 vs 20x20 vs 5x5) | 低 | 10m cell 是否太粗 |

---

## SECTION D: 架构层面的质疑

### D-1: 323 张图 + ViT-Base = 统计学灾难
nuScenes-mini 仅 323 张图。即使冻结大部分参数，可训练参数量仍远超样本量。在 323 张图上的"最佳 recall"能说明什么？模型是否只是记住了这 323 张图的模式？

### D-2: Autoregressive + Slot-Aligned = 误差级联
自回归模型的误差在 30 token 序列中累积。第一个 slot 的 marker 错误会污染后续所有 token。slot-aligned 评估放大了这种级联效应，使 precision 天然被压低。

### D-3: 10x10 BEV Grid -- 每 cell 10m x 10m
一辆轿车在 BEV 中仅占 cell 面积的 ~0.8%。多辆车可能挤在同一个 cell。3 个 slot 是否足够？从未分析 cell 拥挤度。

---

## SECTION E: 判决

| 发现 | 严重性 | 研究员是否意识到 |
|------|--------|-----------------|
| BUG-1: theta_fine 周期性错误 | 中 | 否 |
| BUG-2: 背景梯度被压制 | **致命** | 否 |
| BUG-3: Score 传播链断裂 | 高 | 是（14:51 发现） |
| BUG-4: 深度排序坐标系错误 | 中 | 否 |
| BUG-5: 投影公式边界行为 | 低 | 否 |
| BUG-6: slot_class 背景赋值错误 | 低 | 否 |
| BUG-7: theta 1.2x 无依据 | 低 | 否 |
| 谬误-1: 过早宣布震荡打破 | 中 | 是（被打脸） |
| 谬误-2: 13 小时追错方向 | 高 | 是（自我纠正） |
| 谬误-3: 忽视 Plan B 副作用 | **致命** | 否 |
| 背景权重消融缺失 | **致命** | 否 |
| LR schedule 消融缺失 | 中 | 部分意识到 |

### 最终判决

研究员 1 号的"最佳结果"(iter 5000, avg_recall=0.675) 是一个**被 bug 和设计缺陷共同压低的伪最优值**。修复 BUG-1 和 BUG-2 后，真正的最优值可能显著不同。

**唯一确定的是：我们不知道真正的 baseline 在哪里。**

---

## SECTION F: 后续发现 (18:01-18:25 狙击报告)

### BUG-8: cls loss 遗漏 bg_balance_weight (High)
**文件**: `git_occ_head.py:871-881`
Marker loss 有 bg_balance_weight=3.0 加权背景，但 cls loss 的 per-class balance 循环仅遍历前景类，完全忽略背景。Marker head 和 cls head 的梯度信号不对称。

### BUG-9: 永久性梯度裁剪 — 100% clipping (Critical)
**文件**: 所有 Plan 配置 `clip_grad=dict(max_norm=0.5)`
Plan D 的 grad_norm 范围 3.85-59.55，**从未低于裁剪阈值 0.5**。100% 的迭代被裁剪。有效学习率在迭代间波动 15.5×。原始 GiT 标准任务使用 max_norm=0.1，occupancy 的梯度比正常大 40-120×，说明 loss function 有严重数值问题。

### BUG-10: 优化器冷启动 (High)
**文件**: Plan D 配置 `resume = False`
`load_from` + `resume=False` 重置 AdamW 的动量和方差估计。从 Plan C checkpoint 加载权重但优化器从零开始，导致前 100-200 步参数更新不稳定。

### BUG-11: 默认类别顺序地雷 (Medium)
**文件**: `generate_occ_flow_labels.py:77`
`__init__` 默认 `["car", "bus", "truck", "trailer"]`，但 config 传入 `["car", "truck", "bus", "trailer"]`。若遗漏 config 值，truck 和 bus 标签互换。

### 新发现: 2D AABB 旋转框面积过估 (Critical)
**文件**: `generate_occ_flow_labels.py:502-506`
Grid 分配使用投影角点的 2D AABB 而非旋转框面积。Truck 在 45° 旋转时 AABB 面积是真实面积的 3.5×，导致 cell 覆盖膨胀、IBW 权重稀释、边缘 cell 回归噪声。

### 新发现: 30-token 误差级联的定量影响
Slot 2 的有效 bg_recall 比 slot 0 低约 19%。自回归结构中，错误在 30 token 中像多米诺骨牌一样传播。

### 新发现: 梯度三重挤压
每个前景类在总分类梯度中仅占 ~6.8%（Plan C）。Truck 的实际贡献仅 2.1%，而背景拿走 14.3%。背景分类梯度是 truck 的 7 倍。

---

## 修订后的 BUG 清单 (截至 18:25)

| Bug | 严重性 | 状态 | 发现时间 |
|-----|--------|------|---------|
| BUG-1: theta_fine 周期性 | 中 | FIXED in Plan C | 15:30 |
| BUG-2: BG 梯度被压制 | 致命 | FIXED in Plan C | 15:30 |
| BUG-3: Score 传播断裂 | 高 | FIXED in Plan C | 15:30 |
| BUG-4: 深度排序坐标系 | 中 | DEFERRED | 15:30 |
| BUG-5: 投影 Z 边界 | 低 | DEFERRED | 15:30 |
| BUG-6: slot_class bg→trailer | 低 | DEFERRED | 15:30 |
| BUG-7: theta 1.2x magic | 低 | DEFERRED | 15:30 |
| BUG-8: cls loss 缺 bg_weight | 高 | UNPATCHED | 18:01 |
| BUG-9: 永久梯度裁剪 | **致命** | UNPATCHED | 18:15 |
| BUG-10: 优化器冷启动 | 高 | UNPATCHED | 18:15 |
| BUG-11: 类别顺序地雷 | 中 | UNPATCHED | 18:25 |

---

*本报告基于对项目代码、实验日志、监督报告的全量审计。每一条质疑均附有文件路径和行号，可直接验证。*
*最后更新: 2026-03-05 18:25*
