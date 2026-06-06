# 第九轮 VI 保护实验记录

生成时间：2026-06-06

## 目标

当前可上传保底仍是 `r4_02_micro_edge_balanced43`。它的结构、SSIM、Nabf 比较稳，但 `AG/SF/Qabf` 仍低于第一名画像。第九轮尝试从 `11_vi_only` 的高 Qabf/VIF 特征出发，只做低风险结构回注，目标是提高 Qabf 的同时不破坏 `SSIM/Nabf`。

## 生成内容

- `scripts/generate_round9_vi_protect_candidates.py`
  - 生成 18 个候选。
  - 方法包括 `r4_02 + vi11` 保护混合、门控 IR 边缘回注、source-pick 高频选择和分解式低频/高频组合。
  - 输出目录：`results/candidates_round9/`
  - 分数表：`results/round9_local_scores.csv`

- `scripts/generate_round9b_qabf_ceiling.py`
  - 生成 24 个候选。
  - 专门验证 `vi11` 高频上限：线性混合、只替换低频结构、保留 VI 高频。
  - 输出目录：`results/candidates_round9b/`
  - 分数表：`results/round9b_local_scores.csv`

## 代表结果

第九轮预测最好候选：

- 候选：`18_source_pick_s450`
- 本地校验：93 张完整、同名、同尺寸、可解码
- 校准预测插榜平均名次：`23.31`
- 预测指标摘要：
  - `AG=6.00`
  - `Qabf=0.536`
  - `SD=41.1`
  - `SF=7.23`
  - `Nabf=0.036`
  - `SSIM=0.743`

第九轮补充组预测最好候选：

- 候选：`07_raw_r4_vi11_w70`
- 本地校验：93 张完整、同名、同尺寸、可解码
- 校准预测插榜平均名次：`24.23`
- 预测指标摘要：
  - `AG=4.51`
  - `Qabf=0.523`
  - `SD=39.9`
  - `SF=6.06`
  - `Nabf=0.007`
  - `SSIM=0.773`

## 判断

第九轮没有得到超过 `r4_02_micro_edge_balanced43` 的候选。原因是：

- `source_pick` 能提升 `AG/SF`，但 Qabf 预测仍低于目标，且 Nabf 开始上升。
- `vi11` 高频保护能保持低 Nabf 和较高 SSIM，但 `AG/SF/CC/SCD` 不足，插榜预测更差。
- 只替换低频结构不能同时解决 Qabf 与结构指标之间的冲突。

因此本轮不生成新的官方 `.dt`，不加入上传队列。

## 下一步

1. 先上传当前队列首位 `next_upload_r4_02_micro_edge_balanced43.dt` 换取真实反馈。
2. 若真实反馈显示 `Qabf/AG/SF` 明显低于预测，再上传 `r5_08` 或 `r7_13` 做对照。
3. 远程 GPU 开机后，优先用深度模型输出作为 `10%-35%` 高频/色调辅料，重点寻找能提升 `Qabf` 且不引入高 Nabf 的模型输出。
