# 强化学习在均值回复股票交易中的应用

日期：2026-05-06  
数据来源：`results_main.csv` 和同目录下的输出图片

## 摘要

本项目用一个模拟的均值回复股票市场展示强化学习在金融交易中的基本应用。交易者在每个时间点观察价格状态，选择连续持仓 \(A_t\)，并通过强化学习算法学习从状态到交易动作的策略。实验比较了三种常见连续动作强化学习算法：PPO、TD3 和 SAC。

核心结论是：在当前实验设置下，off-policy actor-critic 方法 TD3 和 SAC 通常比 PPO 获得更高的 Monte Carlo 平均折扣总奖励；PPO 的策略更保守、换手率更低，但收益较弱。Reward (1) 提供了解析最优策略，可以作为判断 learned policy 是否合理的 benchmark。Reward (2) 引入换仓惩罚，更接近交易成本设定，但不同算法的 turnover 行为差异更明显。

## 1. 问题背景

金融交易可以自然地写成强化学习问题：智能体根据市场状态决定仓位，并从未来价格变化中获得收益。这里的目标不是预测单期涨跌，而是学习一个交易策略，使长期折扣总奖励最大。

本项目使用模拟数据，而不使用真实股票数据。这样做的好处是价格动态已知，可以控制参数并验证算法是否学到正确的均值回复结构。

## 2. 市场模型

股票价格为 \(S_t\)，初始价格 \(S_0=1\)。定义对数收益率 \(L_t\)，它服从一维均值回复过程：

```text
L_{t+1} = (1-kappa) L_t + sigma Z_t,  Z_t ~ N(0, 1)
S_{t+1} = S_t exp(L_{t+1})
```

状态主要包括当前价格水平和收益率：

```text
Reward (1): state = (log S_t, L_t)
Reward (2): state = (log S_t, L_t, A_{t-1})
```

动作 \(A_t\) 表示当前持有的股票数量，可以为正也可以为负。正数表示做多，负数表示做空。实现中对动作做了截断，避免神经网络训练中出现极端持仓。

## 3. Reward 设计

本项目比较两种奖励函数。

Reward (1) 惩罚持仓大小：

```text
R_t = A_t(S_{t+1} - S_t) - lambda A_t^2
```

这个 reward 的惩罚项只和当前仓位大小有关。由于动作不影响未来价格过程，Reward (1) 可以推导解析最优策略。因此它适合作为 RL 策略的 benchmark。

Reward (2) 惩罚仓位变化：

```text
R_t = A_t(S_{t+1} - S_t) - lambda(A_t - A_{t-1})^2 S_t
```

这个 reward 更像 stylized transaction cost，因为它惩罚换仓幅度。为了让状态满足 Markov property，Reward (2) 的 state 必须包含上一期动作 \(A_{t-1}\)。

## 4. 算法

PPO 是 on-policy 方法。它用当前策略采样 rollout，然后通过 clipped objective 更新 actor 和 value function。PPO 通常训练稳定，但样本效率不如 off-policy 方法。

TD3 是 off-policy deterministic actor-critic。它使用 replay buffer、twin critics、target policy smoothing 和 delayed policy update 来减少 Q 值过估计。它适合连续动作问题。

SAC 是 entropy-regularized stochastic actor-critic。它在最大化 reward 的同时鼓励策略保持一定随机性，因此探索更充分，也适合连续动作交易问题。

## 5. 实验设置

结果来自 `outputs/results_main.csv`。主要设置如下：

| 项目 | 设置 |
| --- | --- |
| 参数组 | baseline, slow_reversion, high_volatility, strong_regularization |
| Seeds | 0, 1 |
| Rewards | R1, R2 |
| Data settings | simulation oracle, sequential |
| Algorithms | PPO, TD3, SAC |
| Monte Carlo evaluation paths | 200 |
| Evaluation horizon | 80 |

报告的主要性能指标是 `mean_return`，即独立 Monte Carlo evaluation 得到的平均折扣总 reward。它不是股票收益率，而是：

```text
E[sum_t gamma^t R_t]
```

## 6. 输出图片

### Sanity check

![Sanity trajectory](fig_sanity_trajectory.png)

这张图检查了模拟价格路径、均值回复收益率和随机动作轨迹，说明环境本身能正常生成数据。

### Training curves

![Training curves](fig_training_curves.png)

训练曲线用于观察 PPO、TD3、SAC 在 baseline R1 oracle setting 下的训练表现。它主要展示训练过程是否正常，不直接等同于最终泛化表现。

### Reward (1) policy comparison

![Reward 1 policy comparison](fig_reward1_policy_comparison.png)

Reward (1) 有解析最优策略，因此可以把 learned policies 和 analytic policy 放在同一张图中比较。越接近解析策略，说明算法越可能学到正确的交易结构。

### Reward (1) policy heatmap

![Reward 1 policy heatmap](fig_reward1_policy_heatmap.png)

Heatmap 展示不同状态下策略选择的仓位。它比单独看 reward 更直观，因为金融 RL 的训练 reward 可能噪声很大。

### Reward (2) policy heatmap

![Reward 2 policy heatmap](fig_reward2_policy_heatmap.png)

Reward (2) 的策略依赖上一期仓位 \(A_{t-1}\)。如果交易成本惩罚有效，策略通常会表现出一定平滑性，避免频繁大幅换仓。

## 7. Baseline 结果

下面表格聚合了 baseline 参数组下两个 seed 的平均结果。

| Reward | Setting | Method | Mean return | Std. error | Turnover | Avg. abs. position | Relative to analytic | Policy MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | oracle | TD3 | 1302.14 | 914.74 | 0.88 | 9.30 | 68.71% | 164.87 |
| R1 | oracle | SAC | 1212.14 | 866.67 | 1.27 | 6.01 | 63.96% | 93.92 |
| R1 | oracle | PPO | 329.81 | 189.67 | 0.13 | 1.66 | 17.40% | 66.09 |
| R1 | sequential | TD3 | 1381.05 | 936.12 | 0.13 | 10.00 | 72.87% | 160.91 |
| R1 | sequential | PPO | 94.46 | 183.26 | 0.34 | 1.30 | 4.98% | 47.77 |
| R1 | sequential | SAC | -1411.78 | 936.10 | 0.20 | 7.38 | -74.49% | 117.81 |
| R2 | oracle | SAC | 1295.27 | 935.50 | 12.07 | 9.92 | N/A | N/A |
| R2 | oracle | TD3 | 1166.50 | 858.66 | 2.53 | 5.93 | N/A | N/A |
| R2 | oracle | PPO | -112.12 | 188.76 | 0.14 | 1.40 | N/A | N/A |
| R2 | sequential | TD3 | 1399.72 | 936.12 | 0.12 | 10.00 | N/A | N/A |
| R2 | sequential | SAC | 98.82 | 911.99 | 0.23 | 8.50 | N/A | N/A |
| R2 | sequential | PPO | -220.01 | 159.74 | 0.24 | 1.52 | N/A | N/A |

在 baseline R1 oracle setting 中，TD3 的平均 reward 最高，约达到解析最优策略的 68.71%。SAC 接近 TD3，约为解析最优的 63.96%。PPO 明显更保守，平均持仓和 turnover 较低，但收益也更弱。

在 R2 oracle setting 中，SAC 的 mean return 最高，但 turnover 明显较高。这说明 SAC 在该设定下获得了较高 reward，但策略交易频率也更激进。TD3 的收益略低，但 turnover 更低。PPO 仍然最保守，收益为负。

## 8. 解析 benchmark 与参数组

Reward (1) 的解析策略提供了理论 benchmark。不同参数组下解析策略的 Monte Carlo 表现如下。

| Parameter set | Analytic mean return | Std. error | Turnover | Avg. abs. position |
| --- | --- | --- | --- | --- |
| baseline | 1895.18 | 1530.11 | 2.49 | 4.86 |
| slow_reversion | 5.224e+16 | 5.224e+16 | 1.43 | 5.32 |
| high_volatility | 8.111e+08 | 8.063e+08 | 2.68 | 5.56 |
| strong_regularization | 1880.77 | 1529.97 | 1.40 | 2.76 |

slow_reversion 和 high_volatility 下的 reward 数值非常大，主要原因是价格模型使用 \(S_t=\exp(\log S_t)\)，在长 horizon 和高波动/慢回复时可能产生极端价格路径。因此这些参数组更适合说明环境敏感性，而不适合过度解释为真实金融收益。

## 9. R1 Oracle 下的算法对比

下面表格聚合了 R1 + simulation oracle 下不同参数组的结果。

| Parameter set | Method | Mean return | Relative to analytic | Turnover | Policy MSE |
| --- | --- | --- | --- | --- | --- |
| baseline | TD3 | 1302.14 | 68.71% | 0.88 | 164.87 |
| baseline | SAC | 1212.14 | 63.96% | 1.27 | 93.92 |
| baseline | PPO | 329.81 | 17.40% | 0.13 | 66.09 |
| high_volatility | TD3 | 4.507e+08 | 55.57% | 0.97 | 80.86 |
| high_volatility | SAC | 4.505e+08 | 55.55% | 0.60 | 153.34 |
| high_volatility | PPO | 1.835e+08 | 22.62% | 0.23 | 66.03 |
| slow_reversion | TD3 | 2.904e+16 | 55.59% | 0.26 | 167.10 |
| slow_reversion | SAC | 2.904e+16 | 55.59% | 0.91 | 176.01 |
| slow_reversion | PPO | -1.265e+16 | -24.22% | 0.13 | 74.69 |
| strong_regularization | SAC | 1193.37 | 63.45% | 0.88 | 56.77 |
| strong_regularization | TD3 | 1133.18 | 60.25% | 1.32 | 61.79 |
| strong_regularization | PPO | 420.36 | 22.35% | 0.15 | 5.60 |

整体上，TD3 和 SAC 在多数参数组里明显优于 PPO。PPO 的优势是动作更保守，策略和解析策略的 MSE 有时较低，但它没有充分利用可交易信号，因此最终 reward 较低。

## 10. 讨论

第一，强化学习可以自然地用于交易策略学习。状态包含市场信息，动作是仓位，reward 是交易收益减去风险或交易成本惩罚。

第二，reward design 很关键。R1 惩罚持仓大小，数学结构更简单，也有解析 benchmark。R2 惩罚仓位变化，更接近交易成本，但策略解释更复杂。

第三，off-policy 方法在本实验中更有优势。TD3 和 SAC 可以重复利用 replay buffer 中的数据，因此样本效率通常高于 PPO。PPO 更稳定但更保守。

第四，训练结果的统计不确定性较大。`std_error` 在不少设置中很高，说明 Monte Carlo 路径下的 reward 方差很大。因此结果更适合作为算法行为展示，而不是精确排名。

## 11. 局限性

本项目使用模拟市场，不包含真实交易中的 bid-ask spread、market impact、资金限制、风险控制和非平稳性。

实验只使用两个 seed，因此结论仍然受随机性影响。更严谨的研究应使用更多 seed 和更多 evaluation paths。

slow_reversion 和 high_volatility 参数组会产生很大的 reward 数值，说明该价格模型在部分参数下可能出现极端路径。后续可以加入 price clipping、risk-adjusted metric 或 Sharpe ratio。

本报告没有单独做 learning-rate ablation。当前结果已经包含多个市场参数组，但超参数消融仍可以作为后续扩展。

## 12. 结论

本项目展示了强化学习在金融交易中的一个最小完整流程：构造市场环境，定义 state/action/reward，实现 PPO、TD3 和 SAC，最后用独立 Monte Carlo 路径评估 learned policy。

实验表明，在这个连续动作均值回复交易任务中，TD3 和 SAC 通常比 PPO 获得更高 reward。PPO 更保守，但收益较弱。Reward (1) 的解析 benchmark 对验证 learned policy 非常有帮助，而 Reward (2) 说明交易成本会显著改变策略行为。

因此，本项目的主要价值不是提出可直接交易的策略，而是展示如何把金融交易问题形式化为强化学习问题，并比较不同 RL 算法在同一环境中的行为差异。
