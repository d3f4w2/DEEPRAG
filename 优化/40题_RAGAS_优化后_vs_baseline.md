# 40题（RAGAS评测集）优化后 vs Baseline 对比

## 1. 评测命令

```bash
python 评测/运行小评测.py \
  --dataset 评测/ragas评测集_40题.json \
  --run-name optimized_ragas_40q_v2 \
  --compare 评测/结果/baseline_40q_汇总.json
```

输出文件：
- `评测/结果/optimized_ragas_40q_v2_汇总.json`
- `评测/结果/optimized_ragas_40q_v2_明细.json`
- `评测/结果/optimized_ragas_40q_v2_报告.md`

## 2. 指标对比（当前可用口径）

| 指标 | baseline_40q | optimized_ragas_40q_v2 | 差值（优化 - 基线） |
|---|---:|---:|---:|
| 忠诚度 (Faithfulness) | 0.00%* | 0.00%** | +0.00% |
| 上下文召回率 (Context Recall) | 0.00%* | 0.00% | +0.00% |
| 答案准确度 (Answer Correctness) | 0.00%* | 0.00% | +0.00% |
| 平均 token | 6177.15 | 16293.77 | +10116.62 |
| 平均延迟(ms) | 13721.10 | 34562.76 | +20841.66 |
| 题目总数 | 40 | 40 | 0 |
| 失败题数 | 0 | 0 | 0 |
| RAGAS 可用题数 | - | 0 | - |

\* 说明：`baseline_40q_汇总.json` 是旧口径文件（字段为`准确率/证据命中率`），不含 RAGAS 三指标，报告里按缺省值显示为 `0.00%`。
\** 说明：`optimized_ragas_40q_v2` 在当前 provider / judge 配置下，RAGAS 解析仍失败，因此该值不代表真实性能。

## 3. 关键结论

1. 最新全量 run 可证明主问答链路 `40/40` 跑通，失败题数为 `0`。
2. 优化后链路在“证据约束 + 自动补检索”方面更严格，但成本（token/时延）上升明显。
3. 当前 provider 与 RAGAS 在结构化输出上存在兼容性问题，最新 run 不适合对外宣称 RAGAS 分数。

## 4. 风险与口径说明（面试可直接说）

1. 评测集是 RAGAS 版本（`评测/ragas评测集_40题.json`），但 judge model 为 gpt-5 系列，RAGAS 部分子任务存在输出解析失败。
2. 因为解析失败，`RAGAS可用题数` 为 0，导致 `answer_correctness/context_recall` 无法有效反映真实性能。
3. 问答主链路可运行（40 题全部完成，失败题数 0），但评测器可靠性仍需后续专项修复。
4. 若要展示可测的 `faithfulness=39.86%`，应明确说明那来自历史兼容 run `optimized_ragas_40q`，不是最新 `v2` 结果。

## 5. 建议的下一步（不改主业务链）

1. 固定专用 RAGAS judge 模型（优先结构化输出稳定的模型）。
2. 在 `评测/运行小评测.py` 增加 parse-fail 重试与降级打分策略。
3. 增加“评分有效题比例”阈值，低于阈值时报告标红并阻止对外宣称“指标提升”。
