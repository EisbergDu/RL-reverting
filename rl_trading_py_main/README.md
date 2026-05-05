# RL Trading Main (Python Rewrite)

这个目录是对 `I:\RL-reverting\rl_trading copy.ipynb` 的 `.py` 重写版本，目标是：

- 仅保留主实验（不做 ablation、不做 sensitivity）。
- 使用 CPU 多进程训练。
- 将主结果数量控制在约 100（96 个训练任务 + 4 条 Analytic 基准）。

## Files

- `run_rl_trading_main.py`: 主脚本
- `output_highbudget/`: 默认输出目录（运行后生成）

## 任务规模（固定）

- 参数组：`baseline / slow_reversion / high_volatility / strong_regularization`（4）
- seed：`0, 1`（2）
- reward：`R1, R2`（2）
- setting：`oracle, sequential`（2）
- algo：`PPO, TD3, SAC`（3）

训练任务数：`4 * 2 * 2 * 2 * 3 = 96`

再加 4 条每参数组的 Analytic 基准，`results_main.csv` 通常为 100 行。

## Run

在 `I:\RL-reverting\rl_trading_py_main` 目录执行：

```powershell
python .\run_rl_trading_main.py --workers 8 --output-dir .\output_highbudget --plots on
```

参数说明：

- `--workers N`: 多进程 worker 数，默认 `cpu_count - 1`
- `--output-dir PATH`: 输出目录，默认脚本同级 `output_highbudget/`
- `--plots on|off`: 是否生成图，默认 `on`

## Outputs

脚本会在输出目录下生成：

- `results_main.csv`
- `fig_sanity_trajectory.png`
- `fig_training_curves.png`
- `fig_reward1_policy_comparison.png`
- `fig_reward1_policy_heatmap.png`
- `fig_reward2_policy_heatmap.png`

附加中间产物：

- `models/*.pt`：从 96 个任务里挑选出的关键模型 checkpoint（用于画图，不额外重训）
- `logs/*.csv`：关键训练曲线日志

## Notes

- 脚本固定使用 CPU（`torch.device("cpu")`）。
- Windows 下使用 `spawn` 启动多进程。
- 若你需要进一步压缩单任务训练开销，可在脚本内调整 `MAIN_BUDGET`。
