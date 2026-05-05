# 强化学习金融交易项目报告与讲解稿（High Budget 结果版）

日期：2026-05-06  
数据来源：`output_highbudget/results_main.csv`、训练日志和同目录下生成的图片  
项目主题：用模拟均值回复股票市场展示强化学习在金融交易中的应用，并比较 PPO、TD3、SAC 三种连续动作强化学习算法

---

# 第一部分：如何讲解这份报告

这一部分用于 presentation 或口头讲解。你可以把它当成讲稿框架。整体讲法是：先讲为什么金融交易适合用强化学习建模，再讲环境、reward 和算法，最后用 high budget 的结果解释 PPO、TD3、SAC 的差异。

## 1. 一句话概括项目

可以这样开场：

> 本项目把一个模拟的均值回复股票交易问题建模成强化学习任务，让智能体根据市场状态选择连续持仓，并比较 PPO、TD3 和 SAC 三种算法在不同 reward、数据协议和市场参数下的表现。

这句话要突出四点：

- 这是一个金融强化学习项目。
- 市场是模拟的 mean-reverting stock market。
- 动作是连续仓位，而不是简单买入/卖出分类。
- 重点是比较 PPO、TD3、SAC 三个算法的行为差异。

## 2. 建议讲解顺序

推荐按以下顺序讲：

1. 金融交易为什么是 sequential decision-making。
2. 本项目如何定义 state、action、reward。
3. R1 和 R2 的区别。
4. PPO、TD3、SAC 的核心机制。
5. Oracle 与 sequential data 的区别。
6. 用图片解释 learned policy。
7. 用表格解释不同算法表现。
8. 总结局限和未来扩展。

## 3. 5 分钟讲稿

### 第 1 分钟：背景

金融交易不是单次预测问题，而是连续决策问题。交易者需要在每个时间点根据市场状态调整仓位，并在未来价格变化中获得收益或承担损失。因此，交易问题可以自然地写成强化学习问题。

在本项目中，state 是价格状态，action 是持仓 \(A_t\)，reward 是交易收益减去惩罚项。目标是学习一个 policy，让长期折扣总 reward 最大。

### 第 2 分钟：环境与 reward

市场价格由一个均值回复模型生成：

```text
L_{t+1} = (1-kappa)L_t + sigma Z_t
S_{t+1} = S_t exp(L_{t+1})
```

本项目有两个 reward。R1 惩罚持仓大小：

```text
R_t = A_t(S_{t+1}-S_t) - lambda A_t^2
```

R2 惩罚仓位变化：

```text
R_t = A_t(S_{t+1}-S_t) - lambda(A_t-A_{t-1})^2 S_t
```

R1 有解析最优策略，所以可以作为 benchmark；R2 更接近 transaction cost，因为它惩罚频繁换仓。

### 第 3 分钟：算法

PPO 是 on-policy 方法。它使用当前 policy 采样 rollout，并通过 clipped objective 限制策略更新幅度。优点是稳定，缺点是样本效率较低。

TD3 是 off-policy deterministic actor-critic。它使用 replay buffer、twin critics、target policy smoothing 和 delayed policy update，适合连续动作控制。

SAC 是 off-policy stochastic actor-critic，并加入 entropy regularization。它不仅追求 reward，也鼓励探索，因此在噪声较大的交易环境中可能更灵活。

### 第 4 分钟：主要结果

High budget 结果中，baseline R1 oracle 下 SAC 和 TD3 表现接近。SAC 的 mean return 是 485.16，达到解析 benchmark 的 52.27%；TD3 的 mean return 是 479.98，达到 51.71%；PPO 的 mean return 是 -158.29，说明在这个设置下 PPO 没有学到有效交易策略。

在 baseline R1 sequential 下，TD3 表现最好，mean return 为 601.52，并且 turnover 只有 0.13。说明 TD3 在固定 sequential data setting 中仍然较稳定。

在 baseline R2 oracle 下，SAC 的 mean return 为 464.66，略高于 TD3 的 453.52，但 SAC 的 turnover 为 6.98，明显高于 TD3 的 2.80。这说明 SAC 更激进，可能通过更频繁换仓获得 reward。

### 第 5 分钟：总结

整体来看，TD3 和 SAC 明显优于 PPO。PPO 更保守，但在当前训练预算和任务结构下收益较弱。TD3 更稳定，特别是在 sequential data 和 R2 setting 下有较好表现。SAC 在 oracle setting 下有较强收益，但 turnover 偏高。

本项目的主要价值不是提出真实交易策略，而是展示金融交易如何被形式化为 RL 问题，以及不同 RL 算法在相同市场环境中的策略差异。

## 4. 可能被问到的问题

**Q1：这里的 mean_return 是真实投资收益率吗？**

不是。它是 Monte Carlo 平均折扣总 reward：

```text
E[sum_t gamma^t R_t]
```

它包含交易利润，也包含 reward 中的惩罚项。

**Q2：为什么 R1 有 analytic benchmark？**

因为 R1 中动作不会影响未来价格过程，也不依赖上一期动作，所以可以对每一期单独求最优动作。

**Q3：为什么 R2 要把 \(A_{t-1}\) 放进 state？**

因为 R2 的交易成本项依赖 \(A_t-A_{t-1}\)。如果 state 里没有上一期动作，智能体无法知道换仓成本。

**Q4：为什么 off-policy 方法更好？**

TD3 和 SAC 可以重复利用 replay buffer 中的数据，因此在有限训练预算下通常比 PPO 更 sample efficient。

**Q5：这能直接用于真实交易吗？**

不能。真实交易需要考虑手续费、滑点、market impact、资金约束、风险控制和非平稳市场。这个项目是教学和方法展示。

---

# 第二部分：详细项目报告

## 摘要

本项目研究强化学习在金融交易中的简化应用。我们构造了一个模拟均值回复股票市场，让智能体观察市场状态并选择连续持仓。项目比较 PPO、TD3 和 SAC 三种强化学习算法，并在 Reward (1)、Reward (2)、simulation oracle 和 sequential data setting 下进行 Monte Carlo evaluation。

High budget 结果使用 1000 条独立 evaluation paths，评估 horizon 为 80。结果显示：TD3 和 SAC 整体优于 PPO；PPO 的策略更保守，但收益较弱；TD3 在 sequential data setting 下更稳定；SAC 在 oracle setting 下表现较强，但在 R2 下 turnover 明显偏高。

## 1. 问题背景

金融交易是典型的动态决策问题。交易者不是只预测下一期价格，而是需要在多个时间点连续调整仓位。当前的交易动作会影响当前收益、未来风险暴露和后续交易成本。因此，交易问题适合用强化学习描述。

强化学习的核心要素可以对应到交易问题：

- State：当前市场信息，例如价格水平和收益率。
- Action：当前仓位，例如持有多少股票。
- Reward：交易收益减去风险或成本惩罚。
- Policy：从状态到动作的规则。
- Objective：最大化长期折扣总 reward。

本项目使用模拟市场，而不直接使用真实股票数据。这样可以明确价格动态，控制参数组，并使用独立模拟路径评价最终策略。对于 Reward (1)，还可以推导解析最优策略，作为 learned policy 的 benchmark。

## 2. 市场模型

股票价格为 \(S_t\)，初始价格为 \(S_0=1\)。对数收益率 \(L_t\) 服从：

```text
L_{t+1} = (1-kappa)L_t + sigma Z_t,  Z_t ~ N(0,1)
S_{t+1} = S_t exp(L_{t+1})
```

其中：

- \(kappa\)：均值回复速度。
- \(sigma\)：价格波动强度。
- \(Z_t\)：标准正态噪声。

动作 \(A_t\) 是连续持仓。正值表示做多，负值表示做空。实现中会对动作做 clipping，防止神经网络输出过大仓位。

## 3. MDP 定义

对于 Reward (1)，状态为：

```text
s_t = (log S_t, L_t)
```

对于 Reward (2)，状态为：

```text
s_t = (log S_t, L_t, A_{t-1})
```

Reward (2) 必须包含 \(A_{t-1}\)，因为它惩罚仓位变化。如果不把上一期动作放入 state，就无法满足 Markov property。

目标函数为：

```text
maximize E[sum_t gamma^t R_t]
```

报告中的 `mean_return` 就是该目标在独立 Monte Carlo paths 上的估计。

## 4. Reward 设计

### 4.1 Reward (1)：持仓惩罚

```text
R_t = A_t(S_{t+1}-S_t) - lambda A_t^2
```

第一项是持仓带来的价格变动收益。第二项惩罚绝对仓位大小，避免策略无限加杠杆。

Reward (1) 的优点是结构简单，而且可以推导解析最优策略。它适合用来检查 RL 是否学到了正确方向。

### 4.2 Reward (2)：换仓惩罚

```text
R_t = A_t(S_{t+1}-S_t) - lambda(A_t-A_{t-1})^2 S_t
```

这个 reward 惩罚仓位变化，更接近 transaction cost。它鼓励策略不要频繁大幅换仓，因此 learned policy 应该更依赖上一期仓位。

## 5. 数据协议

### 5.1 Simulation Oracle

Oracle setting 中，算法可以不断从模拟器生成新 episode。它相当于拥有一个已知市场模型，可以无限采样。这个 setting 更理想，也更容易训练。

### 5.2 Sequential Data

Sequential setting 中，训练数据来自固定价格路径。它更接近真实历史数据训练，因为真实市场不能无限生成独立路径。该 setting 下数据相关性更强，样本覆盖更有限，因此训练更困难。

## 6. 算法详细介绍

### 6.1 PPO

PPO 是 on-policy policy gradient 方法。它使用当前策略采样 rollout，并用 clipped objective 控制新旧策略差异：

```text
min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)
```

PPO 的优势是训练稳定，策略更新不会过大。缺点是样本效率低，因为 rollout 数据通常只能用于当前更新，不能像 replay buffer 那样长期复用。

在本实验中，PPO 的策略通常较保守。它的 turnover 和平均仓位较低，但 mean return 也较低。High budget baseline R1 oracle 下 PPO 的 mean return 为 -158.29，明显低于 SAC 和 TD3。

### 6.2 TD3

TD3 是 off-policy deterministic actor-critic。它的 actor 输出确定性动作，critic 估计 \(Q(s,a)\)。TD3 的关键机制包括：

- Twin critics：用两个 Q 网络减少过估计。
- Delayed policy update：critic 更新多次后再更新 actor。
- Target policy smoothing：给 target action 加噪声，使 Q 估计更平滑。
- Replay buffer：重复利用历史 transition，提高样本效率。

High budget 结果中，TD3 在 baseline R1 oracle 下 mean return 为 479.98，达到解析 benchmark 的 51.71%。在 baseline R1 sequential 下 mean return 为 601.52，是该 setting 下最好的方法。在 R2 sequential 下，TD3 的 mean return 为 620.20，turnover 只有 0.16，也表现最好。

### 6.3 SAC

SAC 是 off-policy stochastic actor-critic。它在 reward 目标之外加入 entropy term：

```text
expected reward + alpha * entropy
```

这使得 SAC 不会过早变成完全确定性的策略，而是保留探索能力。对于金融环境这种噪声较大、局部最优较多的问题，entropy regularization 可能有帮助。

High budget 结果中，SAC 在 baseline R1 oracle 下 mean return 为 485.16，略高于 TD3；在 baseline R2 oracle 下 mean return 为 464.66，也略高于 TD3。但 SAC 在 R2 oracle 下 turnover 为 6.98，说明策略更频繁换仓，交易行为更激进。

## 7. 实验设置

High budget 结果来自 `output_highbudget/results_main.csv`。

| 项目 | 设置 |
| --- | --- |
| 参数组 | baseline, slow_reversion, high_volatility, strong_regularization |
| Seeds | 0, 1 |
| Rewards | R1, R2 |
| Data settings | simulation oracle, sequential |
| Algorithms | PPO, TD3, SAC |
| Monte Carlo evaluation paths | 1000 |
| Evaluation horizon | 80 |

主要指标：

- `mean_return`：平均折扣总 reward。
- `std_error`：Monte Carlo 标准误。
- `turnover`：平均换仓幅度。
- `avg_abs_position`：平均绝对仓位。
- `relative_to_analytic_pct`：R1 learned policy 相对解析策略的百分比。
- `policy_mse_to_analytic`：R1 learned policy 与解析策略的 MSE。

## 8. 图片解释

### 8.1 Sanity trajectory

![Sanity trajectory](fig_sanity_trajectory.png)

这张图验证模拟环境是否正常。它展示了样本价格路径、均值回复收益率和随机动作。它的作用是确认价格过程和 action clipping 没有明显错误。

### 8.2 Training curves

![Training curves](fig_training_curves.png)

训练曲线展示 baseline R1 oracle 下 PPO、TD3、SAC 的训练过程。曲线用于观察训练是否稳定、是否出现 reward 崩溃，以及不同算法在训练阶段的波动。

训练曲线不是最终性能指标。最终性能应以独立 Monte Carlo evaluation 的 `mean_return` 为准。

### 8.3 Reward (1) policy comparison

![Reward 1 policy comparison](fig_reward1_policy_comparison.png)

这张图比较 learned policy 和 Reward (1) 的 analytic policy。它能直观看出算法是否学到了均值回复交易结构。

如果 learned policy 和 analytic curve 越接近，说明策略越符合理论最优结构。如果曲线过平，说明策略过于保守；如果长期贴近 action bound，说明策略可能过于激进。

### 8.4 Reward (1) policy heatmap

![Reward 1 policy heatmap](fig_reward1_policy_heatmap.png)

Heatmap 展示不同 \(\log S_t\) 和 \(L_t\) 下的动作选择。它显示策略如何根据市场状态改变仓位。平滑的 heatmap 通常说明策略较稳定；极端颜色较多可能说明策略频繁满仓。

### 8.5 Reward (2) policy heatmap

![Reward 2 policy heatmap](fig_reward2_policy_heatmap.png)

Reward (2) 的策略依赖上一期仓位 \(A_{t-1}\)。因此这张图可以观察策略是否具有 inertia，即是否避免不必要的大幅换仓。

## 9. Baseline 结果

下表聚合 baseline 参数组下两个 seed 的平均结果。

| Reward | Setting | Method | Mean return | Std. error | Turnover | Avg. abs. position | Relative to analytic | Policy MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | oracle | SAC | 485.16 | 192.69 | 1.66 | 5.72 | 52.27% | 25.29 |
| R1 | oracle | TD3 | 479.98 | 177.91 | 1.39 | 4.61 | 51.71% | 85.64 |
| R1 | oracle | PPO | -158.29 | 79.53 | 0.18 | 2.62 | -17.05% | 65.40 |
| R1 | sequential | TD3 | 601.52 | 212.87 | 0.13 | 10.00 | 64.80% | 160.91 |
| R1 | sequential | PPO | -86.47 | 60.74 | 0.56 | 1.34 | -9.32% | 37.41 |
| R1 | sequential | SAC | -135.90 | 199.27 | 0.37 | 4.70 | -14.64% | 132.93 |
| R2 | oracle | SAC | 464.66 | 202.54 | 6.98 | 6.41 | N/A | N/A |
| R2 | oracle | TD3 | 453.52 | 174.68 | 2.80 | 4.96 | N/A | N/A |
| R2 | oracle | PPO | -36.66 | 113.17 | 0.18 | 3.88 | N/A | N/A |
| R2 | sequential | TD3 | 620.20 | 212.87 | 0.16 | 9.34 | N/A | N/A |
| R2 | sequential | PPO | -195.06 | 76.71 | 0.41 | 2.43 | N/A | N/A |
| R2 | sequential | SAC | -222.96 | 210.86 | 2.33 | 7.09 | N/A | N/A |

### 9.1 R1 oracle

R1 oracle 是最干净的算法比较，因为有解析 benchmark。SAC 和 TD3 表现接近，分别达到 analytic benchmark 的 52.27% 和 51.71%。PPO 的平均 return 为负，说明在 high budget 这版结果里，PPO 没有学到有效的 R1 oracle 策略。

SAC 的 policy MSE 为 25.29，低于 TD3 的 85.64，说明在曲线形状上 SAC 更接近 analytic policy。TD3 的 mean return 与 SAC 接近，但策略形状和 analytic policy 差异更大。

### 9.2 R1 sequential

R1 sequential 下 TD3 表现最好，mean return 为 601.52。PPO 和 SAC 都为负。这个结果说明在固定 sequential data 下，TD3 的 deterministic actor 和 off-policy replay buffer 更稳定。

### 9.3 R2 oracle

R2 oracle 下 SAC 的 mean return 最高，为 464.66；TD3 为 453.52，接近 SAC。PPO 为 -36.66。

但 SAC 的 turnover 为 6.98，远高于 TD3 的 2.80 和 PPO 的 0.18。因此，SAC 虽然收益略高，但策略换仓更频繁。如果真实交易成本更高，SAC 的优势可能会下降。

### 9.4 R2 sequential

R2 sequential 下 TD3 明显最好，mean return 为 620.20，turnover 为 0.16。PPO 和 SAC 都为负。这说明在有换仓惩罚和固定数据路径的情况下，TD3 的策略更稳健。

## 10. R1 Oracle 下的参数组比较

| Parameter set | Method | Mean return | Relative to analytic | Turnover | Policy MSE |
| --- | --- | --- | --- | --- | --- |
| baseline | SAC | 485.16 | 52.27% | 1.66 | 25.29 |
| baseline | TD3 | 479.98 | 51.71% | 1.39 | 85.64 |
| baseline | PPO | -158.29 | -17.05% | 0.18 | 65.40 |
| high_volatility | SAC | 1.250e+08 | 58.67% | 0.47 | 152.27 |
| high_volatility | PPO | -9.820e+06 | -4.61% | 0.35 | 74.73 |
| high_volatility | TD3 | -1.359e+07 | -6.38% | 0.41 | 166.72 |
| slow_reversion | TD3 | 5.868e+15 | 55.76% | 0.44 | 106.65 |
| slow_reversion | SAC | 5.863e+15 | 55.71% | 0.83 | 81.61 |
| slow_reversion | PPO | -9.220e+14 | -8.76% | 0.11 | 74.08 |
| strong_regularization | TD3 | 477.50 | 52.32% | 1.24 | 22.86 |
| strong_regularization | SAC | 397.42 | 43.54% | 1.60 | 56.66 |
| strong_regularization | PPO | 181.68 | 19.91% | 0.16 | 5.53 |

这个表说明算法表现对市场参数很敏感。在 baseline 和 slow_reversion 下，TD3/SAC 都明显强于 PPO。在 high_volatility 下，SAC 表现最好，而 TD3 和 PPO 为负。strong_regularization 下 TD3 最好，PPO 虽然为正但仍明显落后。

## 11. 解析 benchmark

| Parameter set | Analytic mean return | Std. error | Turnover | Avg. abs. position |
| --- | --- | --- | --- | --- |
| baseline | 928.21 | 334.39 | 2.48 | 4.92 |
| slow_reversion | 1.052e+16 | 1.045e+16 | 1.45 | 5.36 |
| high_volatility | 2.131e+08 | 1.652e+08 | 2.68 | 5.61 |
| strong_regularization | 912.69 | 334.31 | 1.44 | 2.86 |

slow_reversion 和 high_volatility 的数值非常大。这主要来自 \(S_t=\exp(\log S_t)\) 的指数价格模型。当收益率长期偏离或波动率较高时，价格路径可能出现极端值。因此这些结果应更多理解为模型敏感性，而不是现实市场中的可实现收益。

## 12. 不同算法对比

### PPO

PPO 的特点是稳定和保守。它通过 clipped objective 限制策略更新幅度，因此不容易剧烈震荡。但因为 PPO 是 on-policy，它不能充分重复利用历史数据，样本效率较低。

在 high budget 结果中，PPO 在 baseline 下表现较弱。R1 oracle、R1 sequential、R2 oracle 和 R2 sequential 的 mean return 都为负。它的 turnover 通常较低，说明策略确实保守，但也因此没有充分利用交易机会。

### TD3

TD3 是本结果中最稳健的算法。它在 baseline R1 sequential 和 R2 sequential 中都是最优，在 R1 oracle 和 R2 oracle 中也接近 SAC。

TD3 的优势在于 replay buffer 和 twin critics。Replay buffer 提高样本利用率，twin critics 降低 Q-value overestimation。对于连续仓位控制任务，TD3 的 deterministic policy 也比较直接。

### SAC

SAC 在 oracle setting 中表现强。baseline R1 oracle 下它略高于 TD3，R2 oracle 下也略高于 TD3。这说明 entropy-regularized stochastic policy 有助于探索交易策略。

但 SAC 的缺点是 turnover 可能偏高。baseline R2 oracle 下 SAC turnover 为 6.98，明显高于 TD3 的 2.80。这意味着如果真实交易成本更高，SAC 的实际净收益可能会受到影响。

### 综合判断

如果只看 high budget 的结果：

- R1 oracle：SAC 略优于 TD3。
- R1 sequential：TD3 最优。
- R2 oracle：SAC 略优于 TD3，但 turnover 高。
- R2 sequential：TD3 明显最优。
- PPO：整体最保守，但收益最弱。

因此，本项目可以总结为：TD3 更稳健，SAC 在 oracle 环境下有竞争力但更激进，PPO 在当前训练设置下不足以学到强交易策略。

## 13. 局限性与未来展望

### 13.1 模拟市场简化

本项目使用一维均值回复价格模型。真实金融市场包含趋势、波动聚集、跳跃、流动性变化和结构性变化。因此，本项目是方法展示，不是可直接交易的系统。

### 13.2 Reward 与真实交易成本仍有差距

R2 用二次换仓惩罚近似 transaction cost，但真实交易成本包括 bid-ask spread、commission、slippage 和 market impact。真实成本可能随成交量和市场流动性变化。

### 13.3 风险指标不足

当前主要指标是 mean discounted total reward。真实策略还应报告 Sharpe ratio、maximum drawdown、tail risk、VaR/CVaR 等指标。高 reward 不一定意味着风险可接受。

### 13.4 Seed 数量有限

虽然 high budget 使用了 1000 条 evaluation paths，但训练 seed 只有 0 和 1。更严谨的实验应使用更多 seed，以判断结果是否稳定。

### 13.5 参数组中存在极端数值

slow_reversion 和 high_volatility 的 reward 数值很大，说明模型在部分参数下会出现极端路径。未来可以加入 price clipping、log-return based reward 或 risk-adjusted reward。

### 13.6 未来扩展

后续可以继续做：

- 加入真实股票或 ETF 数据。
- 使用 walk-forward validation。
- 增加更多训练 seed。
- 加入 realistic transaction cost。
- 比较更多算法，如 DDPG、A2C、离散动作 DQN。
- 做 hyperparameter ablation，例如 learning rate、batch size、network width。
- 加入 risk-adjusted objective，例如 Sharpe ratio 或 mean-variance reward。

## 14. 总结

本项目展示了金融交易如何被形式化为强化学习问题。智能体通过观察市场状态选择仓位，并用长期折扣 reward 学习交易策略。

High budget 结果表明，TD3 和 SAC 在大多数设置中明显优于 PPO。TD3 在 sequential data 下更稳健，SAC 在 oracle setting 下更有竞争力但 turnover 更高，PPO 更保守但收益较弱。

从课程项目角度看，这个实验完整展示了金融 RL 的关键流程：环境建模、reward 设计、算法比较、Monte Carlo evaluation、policy visualization 和局限性分析。它不是一个真实交易系统，而是一个清晰的金融强化学习 prototype。
