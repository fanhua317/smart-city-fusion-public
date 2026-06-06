# 面向智慧城市的信息融合大作业提交说明

## 当前最佳提交文件

当前官方可上传版本：

```text
results/final/eval/rank_best_33_mix_avg21_45.dt
```

对应候选：`33_mix_avg21_45`

该版本已经通过官方 `Evaluation.exe` 生成 `.dt`，并完成本地输出校验：

- 融合图数量：93
- 文件名：与 IR/VI 成对匹配
- 尺寸：与源图一致
- 异常图像：0

## 当前官方指标

指标摘要保存在：

```text
results/final/eval/official_metrics_summary.json
results/score_tables/current_official_scores.csv
results/score_tables/current_official_scores.md
```

`33_mix_avg21_45` 插入 `leaderboard_20260605_234547.csv` 后的平均单项名次约为 `24.08`。它的结构指标较稳，短板是细节、熵、互信息和空间频率相关指标。

## 评测工具使用

`Evaluation.exe` 没有可靠命令行入口，需要使用 GUI：

1. 红外文件夹：`target/ir`
2. 可见文件夹：`target/vi`
3. 融合文件夹：`results/final/fused`
4. 输出文件夹：`results/final/eval`
5. 输出文件名：`rank_best_33_mix_avg21_45`

注意：输出文件名不要加 `.dt`，工具会自动追加扩展名。

## GitHub 仓库内容

仓库只保存代码、文档和分数表，不保存课程测试集、官方评测器、批量候选图片、压缩包或 `.dt` 文件。具体规则见 `.gitignore` 和 `docs/实验注意事项.md`。

## 复现命令

生成候选和本地指标：

```powershell
py -3 scripts\generate_round3_candidates.py
py -3 scripts\validate_outputs.py --fused-dir results\final\fused --out results\final\eval\validation.json
```

导出当前分数表：

```powershell
py -3 scripts\export_official_scores.py
```

## 后续优化记录

第三轮生成过 18 个候选，其中 `source_consistent_edge` 系列在本地指标上表现突出，但代表候选在官方评测工具中出现内存分配错误，暂未作为可上传版本。后续继续优化时，必须以能生成官方 `.dt` 为第一约束。
