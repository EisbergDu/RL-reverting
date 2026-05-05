# 强化学习方法在模拟均值回复股票交易中的比较

日期：2026-05-06  
结果目录：`output_highbudget/`  
主要数据文件：`results_main.csv`

## 摘要

本项目研究强化学习方法在模拟均值回复股票交易问题中的应用。我们将交易过程建模为一个连续动作 Markov Decision Process，其中智能体观察市场状态，选择当前持仓 \(A_t\)，并通过交易收益与惩罚项构成的 reward 学习交易策略。项目比较了三种常见连续控制强化学习算法：PPO、TD3 和 SAC，并在两种 reward 设计、两种数据协议以及多组市场参数下评估 learned policy。

实验使用模拟价格模型生成股票路径。对数收益率 \(L_t\) 服从均值回复过程，价格由 \(S_{t+1}=S_t\exp(L_{t+1})\) 更新。项目考虑两个 reward：Reward (1) 惩罚绝对持仓大小，Reward (2) 惩罚仓位变化以近似交易成本。对于 Reward (1)，由于动作不影响未来价格过程，可以推导解析最优策略，因此它被用作 learned policy 的理论 benchmark。

High budget 结果使用 1000 条独立 Monte Carlo evaluation paths，评估 horizon 为 80。实验结果表明，TD3 和 SAC 整体显著优于 PPO。Baseline R1 oracle setting 下，SAC 的 mean return 为 485.16，TD3 为 479.98，分别达到解析 benchmark 的 52.27% 和 51.71%；PPO 的 mean return 为 -158.29。Baseline R2 oracle setting 下，SAC 的 mean return 为 464.66，略高于 TD3 的 453.52，但 SAC turnover 为 6.98，高于 TD3 的 2.80，说明 SAC 策略更激进。Baseline R2 sequential setting 下，TD3 的 mean return 为 620.20，显著优于 PPO 和 SAC，并保持较低 turnover。

总体而言，本项目说明强化学习可以自然地用于交易策略学习，但算法表现高度依赖 reward design、数据协议和市场参数。TD3 在当前实验中最稳健，SAC 在 oracle setting 下有竞争力但换手率较高，PPO 更保守且样本效率较低。

## 1. 问题背景

金融交易是典型的序列决策问题。交易者需要在多个时间点连续观察市场、调整仓位并承担未来价格波动带来的收益或损失。与单纯预测下一期价格不同，交易策略学习需要决定在当前状态下应该持有多少仓位，并综合考虑收益、风险、交易成本和未来机会。因此，金融交易问题可以自然地写成强化学习中的 state-action-reward 框架。

在强化学习中，智能体与环境交互。环境给出状态，智能体选择动作，环境返回 reward 和下一状态。对应到本项目：

- 状态是当前市场信息，例如价格水平和收益率。
- 动作是当前持仓 \(A_t\)，可以为正、负或零。
- Reward 是交易利润减去风险或交易成本惩罚。
- Policy 是从状态到动作的映射。
- 目标是最大化长期折扣总 reward。

本项目使用模拟市场，而不是直接使用真实股票数据。这样做有三个原因。第一，模拟环境可以明确控制市场参数，例如均值回复速度、波动率和惩罚系数。第二，模拟环境可以生成独立的 evaluation paths，避免训练路径和评估路径混淆。第三，Reward (1) 有解析最优策略，可以作为检查 RL 方法是否学到合理策略结构的 benchmark。

需要强调的是，本项目的目标不是构造可直接用于真实交易的策略，而是展示如何将金融交易形式化为强化学习问题，并比较不同 RL 算法在同一模拟环境中的行为差异。

## 2. 市场模型

股票价格记为 \(S_t\)，初始值为 \(S_0=1\)。定义对数收益率 \(L_t\)，其动态为：

```text
L_{t+1} = (1 - kappa) L_t + sigma Z_t,  Z_t ~ N(0, 1)
S_{t+1} = S_t exp(L_{t+1})
```

其中，\(\kappa\in(0,1)\) 控制均值回复速度，\(\sigma>0\) 控制波动率，\(Z_t\) 是独立标准正态随机变量。当 \(\kappa\) 较大时，收益率更快回到零；当 \(\sigma\) 较大时，价格路径波动更强。

交易动作 \(A_t\) 表示第 \(t\) 期持有的股票数量。正值代表做多，负值代表做空。由于动作是连续变量，本项目属于 continuous control 问题。为了避免神经网络输出极端仓位，实际实现中会将动作截断到有限区间。

## 3. MDP 形式化

对于 Reward (1)，状态定义为：

```text
s_t = (log S_t, L_t)
```

对于 Reward (2)，状态定义为：

```text
s_t = (log S_t, L_t, A_{t-1})
```

Reward (2) 必须包含上一期动作 \(A_{t-1}\)，因为其交易成本项依赖仓位变化 \(A_t-A_{t-1}\)。如果状态中不包含上一期动作，智能体无法仅根据当前状态判断换仓成本，从而破坏 Markov property。

目标函数为：

```text
maximize E[sum_t gamma^t R_t]
```

报告中的 `mean_return` 是该目标在独立 Monte Carlo evaluation paths 上的估计。它不是股票收益率，而是 reward 的折扣总和。

## 4. Reward 设计

### 4.1 Reward (1)：持仓惩罚

Reward (1) 定义为：

```text
R_t = A_t(S_{t+1} - S_t) - lambda A_t^2
```

第一项 \(A_t(S_{t+1}-S_t)\) 是交易利润。如果智能体做多并且价格上涨，则该项为正；如果智能体做空并且价格下跌，也可以获得正收益。第二项 \(\lambda A_t^2\) 惩罚绝对仓位大小，用来限制过度杠杆和极端持仓。

Reward (1) 的重要性质是动作不会影响未来价格过程。因此，在给定当前状态时，可以逐期求解最优动作。解析策略为 learned policy 提供了强 benchmark：如果 RL 策略学得合理，它的动作曲线应当接近解析策略。

### 4.2 Reward (2)：换仓惩罚

Reward (2) 定义为：

```text
R_t = A_t(S_{t+1} - S_t) - lambda(A_t - A_{t-1})^2 S_t
```

与 Reward (1) 不同，Reward (2) 不直接惩罚仓位大小，而是惩罚仓位变化。它更接近 stylized transaction cost，因为真实交易中频繁买卖会产生手续费、滑点和市场冲击。

Reward (2) 的策略通常应更平滑。若上一期已有较大仓位，除非当前市场信号足够强，否则策略不应频繁大幅换仓。

## 5. 数据协议

### 5.1 Simulation Oracle

在 simulation oracle setting 中，算法可以不断从已知模拟器生成新的独立 episode。这是一个理想化设定，相当于智能体拥有可无限采样的市场模型。该 setting 数据覆盖更充分，训练更容易稳定。

### 5.2 Sequential Data

在 sequential data setting 中，训练数据来自固定价格路径。这更接近真实历史数据场景，因为现实中无法随意生成无限独立市场路径。Sequential data 的难点是样本相关性强、覆盖有限，因此对算法的数据效率和稳定性要求更高。

## 6. 强化学习算法

### 6.1 PPO

PPO，即 Proximal Policy Optimization，是一种 on-policy policy gradient 方法。它使用当前策略采样 rollout，并基于 advantage estimate 更新策略。PPO 的核心是 clipped objective：

```text
min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)
```

其中 \(r_t\) 是新旧策略概率比。clip 操作限制策略更新幅度，避免新策略偏离旧策略太远。PPO 的优点是训练稳定，缺点是样本效率较低，因为 rollout 数据通常只能用于当前策略更新，不能长期重复利用。

在本项目中，PPO 的 actor 输出连续动作分布，critic 估计 value function。结果显示 PPO 通常更保守，turnover 和平均仓位较低，但 mean return 也较弱。High budget baseline R1 oracle 下 PPO 的 mean return 为 -158.29，说明在当前训练设置下 PPO 没有学到有效交易策略。

### 6.2 TD3

TD3，即 Twin Delayed Deep Deterministic Policy Gradient，是一种 off-policy deterministic actor-critic 方法。它直接学习确定性策略 \(A_t=\mu(s_t)\)，并用 critic 估计 \(Q(s_t,A_t)\)。

TD3 有三个关键机制：

1. Twin critics：训练两个 Q 网络，并在 target value 中取较小值，减少 Q-value overestimation。
2. Delayed policy update：critic 更新多次后再更新 actor，提高 actor 更新质量。
3. Target policy smoothing：给 target action 加入截断噪声，避免策略过度利用 Q 函数局部尖峰。

TD3 使用 replay buffer，因此可以重复利用历史 transition，样本效率高于 on-policy 方法。High budget 结果显示，TD3 在 sequential setting 下尤其稳定。Baseline R1 sequential 下 TD3 的 mean return 为 601.52；baseline R2 sequential 下为 620.20，均为对应 setting 中最高。

### 6.3 SAC

SAC，即 Soft Actor-Critic，是一种 off-policy stochastic actor-critic 方法。它在最大化 reward 的同时加入 entropy regularization：

```text
expected reward + alpha * entropy
```

这种设计鼓励策略保持随机性和探索能力，避免过早收敛到单一确定性动作。对于金融市场这种噪声较大、局部最优较多的环境，SAC 的探索机制可能带来优势。

High budget 结果显示，SAC 在 oracle setting 下表现强。Baseline R1 oracle 下 SAC 的 mean return 为 485.16，略高于 TD3 的 479.98。Baseline R2 oracle 下 SAC 的 mean return 为 464.66，也略高于 TD3 的 453.52。但 SAC 在 R2 oracle 下 turnover 为 6.98，高于 TD3 的 2.80，说明它的策略更激进。

## 7. 实验设置

本报告使用 `output_highbudget/results_main.csv` 中的结果。

| 项目 | 设置 |
| --- | --- |
| 参数组 | baseline, slow_reversion, high_volatility, strong_regularization |
| Seeds | 0, 1 |
| Rewards | R1, R2 |
| Data settings | simulation oracle, sequential |
| Algorithms | PPO, TD3, SAC |
| Monte Carlo evaluation paths | 1000 |
| Evaluation horizon | 80 |

主要指标包括：

- `mean_return`：平均折扣总 reward。
- `std_error`：Monte Carlo 标准误。
- `turnover`：平均换仓幅度。
- `avg_abs_position`：平均绝对仓位。
- `relative_to_analytic_pct`：R1 learned policy 相对解析 benchmark 的百分比。
- `policy_mse_to_analytic`：R1 learned policy 与解析策略之间的 MSE。

## 8. 图像结果解释

### 8.1 环境 sanity check

![Sanity trajectory](fig_sanity_trajectory.png)

该图展示模拟价格路径、均值回复收益率和随机动作轨迹。它用于检查环境是否正常生成数据。价格路径应随收益率变化，\(L_t\) 应表现出均值回复特征，动作应在允许范围内。

### 8.2 训练曲线

![Training curves](fig_training_curves.png)

训练曲线展示 baseline R1 oracle setting 下 PPO、TD3 和 SAC 的训练过程。该图用于检查训练是否正常，例如 reward 是否出现明显崩溃、不同算法是否有学习趋势。最终性能仍以独立 Monte Carlo evaluation 的 `mean_return` 为准。

### 8.3 Reward (1) 策略对比

![Reward 1 policy comparison](fig_reward1_policy_comparison.png)

Reward (1) 有解析最优策略，因此可以将 learned policies 与 analytic policy 画在同一张图中。若 learned policy 与 analytic curve 越接近，说明算法越可能学到正确的均值回复交易结构。若策略曲线过平，说明算法过于保守；若长期贴近 action bound，说明策略可能过于激进。

### 8.4 Reward (1) 策略热力图

![Reward 1 policy heatmap](fig_reward1_policy_heatmap.png)

该图展示不同 \(\log S_t\) 和 \(L_t\) 下策略选择的仓位。颜色代表动作大小。平滑的 heatmap 表明策略对状态变化响应连续；大面积极端颜色可能表示策略倾向于满仓或满仓做空。

### 8.5 Reward (2) 策略热力图

![Reward 2 policy heatmap](fig_reward2_policy_heatmap.png)

Reward (2) 的动作依赖上一期仓位 \(A_{t-1}\)。该图用于观察策略是否体现交易成本下的 inertia。若交易成本惩罚有效，策略应避免从大多头快速切换到大空头，仓位变化应更平滑。

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

R1 oracle 是最适合比较算法学习能力的 setting，因为它有解析 benchmark。SAC 与 TD3 表现接近，分别达到解析 benchmark 的 52.27% 和 51.71%。PPO 的 mean return 为负，说明其策略没有充分利用均值回复信号。

SAC 的 policy MSE 为 25.29，低于 TD3 的 85.64，说明 SAC 在策略形状上更接近 analytic policy。TD3 的 return 与 SAC 接近，但策略形状偏差更大。

### 9.2 R1 sequential

R1 sequential 下 TD3 表现最好，mean return 为 601.52，turnover 仅为 0.13。PPO 和 SAC 均为负。该结果表明，在固定价格路径训练时，TD3 的 off-policy deterministic actor-critic 结构更稳定。

### 9.3 R2 oracle

R2 oracle 下 SAC 的 mean return 为 464.66，略高于 TD3 的 453.52。但 SAC turnover 为 6.98，明显高于 TD3 的 2.80 和 PPO 的 0.18。这说明 SAC 获得较高 reward 的同时，也采用了更频繁的仓位调整。

### 9.4 R2 sequential

R2 sequential 下 TD3 表现最好，mean return 为 620.20，turnover 为 0.16。PPO 和 SAC 均为负。该结果说明，在固定数据路径和换仓惩罚同时存在时，TD3 的策略最稳健。

## 10. 参数组比较

R1 oracle setting 下，不同参数组的算法表现如下。

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

该表说明算法表现对市场参数敏感。在 baseline、slow_reversion 和 strong_regularization 中，TD3 或 SAC 明显优于 PPO。在 high_volatility 下，SAC 表现最好，而 PPO 和 TD3 为负。这说明高波动环境更难训练，也更容易放大策略差异。

## 11. 解析 benchmark

Reward (1) 的 analytic policy 在不同参数组下的表现如下。

| Parameter set | Analytic mean return | Std. error | Turnover | Avg. abs. position |
| --- | --- | --- | --- | --- |
| baseline | 928.21 | 334.39 | 2.48 | 4.92 |
| slow_reversion | 1.052e+16 | 1.045e+16 | 1.45 | 5.36 |
| high_volatility | 2.131e+08 | 1.652e+08 | 2.68 | 5.61 |
| strong_regularization | 912.69 | 334.31 | 1.44 | 2.86 |

slow_reversion 和 high_volatility 的数值非常大，主要来自指数价格模型 \(S_t=\exp(\log S_t)\)。当波动较大或均值回复较慢时，价格路径可能出现极端值。因此这些数值不应解释为真实市场中的可实现收益，而应理解为模型敏感性的体现。

## 12. 算法对比总结

PPO 的主要优点是稳定和保守，但在当前实验中样本效率不足。它的 turnover 通常较低，但 mean return 较弱甚至为负。

TD3 是本实验中最稳健的方法。它在 sequential setting 下表现尤其强，说明 replay buffer、twin critics 和 deterministic policy 对固定数据路径的交易任务较有效。

SAC 在 oracle setting 中有竞争力，尤其在 baseline R1 oracle 和 R2 oracle 下略优于 TD3。但 SAC 在 R2 oracle 下 turnover 偏高，说明它更激进。如果引入更真实的交易成本，SAC 的优势可能减弱。

总体而言，TD3 更稳健，SAC 更有探索性但更激进，PPO 更保守但收益不足。

## 13. 局限性与未来工作

本项目使用模拟市场，因此忽略了真实交易中的 bid-ask spread、slippage、market impact、资金约束和非平稳性。真实市场中，策略还需要经过严格回测、交易成本建模和风险控制。

当前实验使用两个 training seeds。虽然 high budget 使用了 1000 条 evaluation paths，但训练随机性仍可能影响结论。更严谨的研究应增加 seed 数量并报告置信区间。

当前主要指标是 mean discounted total reward。真实投资策略还应关注 Sharpe ratio、maximum drawdown、tail risk、VaR/CVaR 等风险指标。

部分参数组产生极端 reward 数值，说明价格模型在高波动或慢均值回复情况下可能产生极端路径。未来可以加入 price clipping、log-return reward 或 risk-adjusted objective。

后续可以扩展到真实股票或 ETF 数据，使用 walk-forward validation，并比较更多算法，如 DDPG、A2C 或离散动作 DQN。同时，也可以系统研究 learning rate、network width、batch size 和 replay buffer size 等超参数。

## 14. 结论

本项目完整展示了金融强化学习实验流程：构造模拟市场环境，定义 state、action 和 reward，实现 PPO、TD3、SAC，使用 Monte Carlo paths 评估 learned policy，并通过 policy visualization 解释策略行为。

High budget 结果显示，TD3 和 SAC 整体明显优于 PPO。SAC 在 oracle setting 下略有优势，但 turnover 较高；TD3 在 sequential setting 下更稳健；PPO 策略较保守但收益较弱。

因此，本项目的主要贡献是清楚展示了如何将金融交易问题形式化为强化学习问题，并说明不同 RL 算法在相同交易环境中的学习能力和策略行为差异。
