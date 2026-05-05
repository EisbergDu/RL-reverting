# 强化学习金融交易项目报告与讲解稿（合并版）

日期：2026-05-06  
数据来源：`results_main.csv`、训练日志和同目录下生成的图片  
项目主题：用模拟均值回复股票市场展示强化学习在金融交易中的应用，并比较 PPO、TD3、SAC 三种连续动作强化学习算法

---

# 第一部分：如何讲解这份报告

这一部分是给 presentation 或口头讲解用的。它不是实验结果本身，而是帮助你把报告讲清楚的讲解稿。核心思路是：先说明金融交易为什么可以建模成强化学习问题，再说明本项目如何定义环境、reward 和算法，最后用结果解释三种算法的差异。

## 1. 一句话概括项目

可以这样开场：

> 本项目研究的是强化学习在金融交易中的一个简化应用。我们构造了一个模拟的均值回复股票市场，让智能体根据价格状态选择持仓，并比较 PPO、TD3 和 SAC 三种强化学习算法学到的交易策略。

这句话里面有四个关键词：

- 强化学习：不是直接预测价格，而是学习交易策略。
- 金融交易：动作是持仓，reward 是交易收益减去惩罚项。
- 均值回复股票市场：价格动态是可控的模拟环境。
- 算法比较：重点比较 PPO、TD3、SAC 在同一环境下的表现。

## 2. 建议讲解顺序

推荐按下面顺序讲：

1. 项目背景：为什么金融交易可以用 RL 表示。
2. 环境建模：state、action、reward 分别是什么。
3. Reward 设计：R1 和 R2 的区别。
4. 三个算法：PPO、TD3、SAC 的核心机制。
5. 实验设置：oracle/sequential、seed、Monte Carlo evaluation。
6. 结果解释：表格和图片说明了什么。
7. 总结：off-policy 方法更强，PPO 更保守，实验仍有局限。

## 3. 5 分钟口头讲稿

### 第 1 分钟：背景

金融交易问题可以看作一个 sequential decision-making problem。交易者不是只做一次预测，而是在每个时间点连续观察市场、调整仓位、获得收益。因此它很适合用强化学习描述。

在本项目中，我们不使用真实股票数据，而是使用一个模拟的均值回复市场。这样做的好处是价格过程已知，实验更可控，也可以用 Reward (1) 的解析最优策略作为 benchmark。

### 第 2 分钟：环境和 reward

状态包含当前的价格水平和收益率，也就是 \(\log S_t\) 和 \(L_t\)。动作 \(A_t\) 表示当前持有多少股票，可以做多也可以做空。目标是让长期折扣 reward 最大。

项目里有两个 reward。Reward (1) 惩罚持仓大小：

```text
R_t = A_t(S_{t+1} - S_t) - lambda A_t^2
```

Reward (2) 惩罚仓位变化：

```text
R_t = A_t(S_{t+1} - S_t) - lambda(A_t - A_{t-1})^2 S_t
```

R1 更简单，有解析最优策略；R2 更接近交易成本，因为频繁换仓会被惩罚。

### 第 3 分钟：算法

PPO 是 on-policy 算法。它只用当前策略采样出来的数据更新策略，通常更稳定，但样本效率较低。

TD3 是 off-policy deterministic actor-critic 算法。它使用 replay buffer，可以重复利用历史数据，并使用 twin critics 和 delayed update 减少 Q-value 过估计。

SAC 也是 off-policy actor-critic，但它是 stochastic policy，并加入 entropy regularization。也就是说，SAC 不仅追求高 reward，也鼓励策略保留探索性。

### 第 4 分钟：结果

从 baseline R1 oracle 结果看，TD3 的 mean return 最高，约为 1302，达到解析 benchmark 的 68.71%。SAC 接近 TD3，约为 1212，达到 63.96%。PPO 只有约 330，说明它更加保守，没有充分利用交易信号。

从 R2 结果看，SAC 在 oracle setting 下 mean return 最高，但 turnover 也很高。这说明它比较激进，频繁换仓。TD3 的收益也不错，而且在 sequential setting 下表现较强。PPO 的动作幅度小、turnover 低，但收益也较低。

### 第 5 分钟：总结和局限

这个项目说明了两点。第一，强化学习可以把金融交易自然地表示为 state-action-reward 的决策问题。第二，不同 RL 算法学到的交易行为差异很大：TD3 和 SAC 更激进、收益更高，PPO 更保守、收益较低。

不过这个项目仍然是简化实验。市场是模拟的，不包含真实交易中的 bid-ask spread、market impact、资金限制和非平稳性。实验 seed 也不多，所以结果更适合作为方法展示，而不能直接作为真实交易策略。

## 4. 可能被问到的问题

**Q1：这里的 mean_return 是股票收益率吗？**

不是。这里的 `mean_return` 是 Monte Carlo 估计的平均折扣总 reward：

```text
E[sum_t gamma^t R_t]
```

它包含交易利润，也包含 reward 里的惩罚项。

**Q2：R1 和 R2 的区别是什么？**

R1 惩罚持仓大小，鼓励策略不要持有太大的绝对仓位。R2 惩罚仓位变化，鼓励策略不要频繁换仓，因此更像 transaction cost。

**Q3：为什么 R1 有 analytic benchmark，R2 没有？**

R1 的动作不会影响未来价格过程，也没有依赖上一期动作，所以可以逐期求最优动作。R2 的 reward 依赖 \(A_{t-1}\)，策略具有路径依赖，解析解更复杂。

**Q4：为什么 TD3 和 SAC 比 PPO 好？**

主要原因是 TD3 和 SAC 是 off-policy，可以重复利用 replay buffer 中的数据，样本效率更高。PPO 是 on-policy，每次更新只使用当前策略采样的数据，因此在相同训练预算下可能学得更慢。

**Q5：这个策略能用于真实交易吗？**

不能直接用于真实交易。项目的目的是教学和方法展示。真实交易还需要考虑交易费用、滑点、市场冲击、风险约束、非平稳数据、回测偏差和资金管理。

---

# 第二部分：详细项目报告

## 摘要

本项目研究强化学习在金融交易中的一个简化应用。我们构建了一个模拟的均值回复股票市场，令智能体在每个时间点观察市场状态并选择连续持仓 \(A_t\)。项目比较了 PPO、TD3 和 SAC 三种连续动作强化学习算法，并在 Reward (1) 与 Reward (2)、simulation oracle 与 sequential data 两种数据协议下进行评估。

实验结果显示，在 baseline 参数组下，TD3 和 SAC 通常比 PPO 获得更高的 Monte Carlo 平均折扣总 reward。PPO 的策略更保守，平均仓位和 turnover 通常更低，但收益表现较弱。TD3 的表现较稳定，尤其在 R1 和 sequential setting 下表现较强。SAC 在部分 R2 oracle setting 下取得较高 reward，但 turnover 明显更高，说明它的策略更激进。Reward (1) 的解析最优策略为 learned policy 提供了重要 benchmark，而 Reward (2) 展示了交易成本型惩罚如何改变策略行为。

## 1. 问题背景

金融交易本质上是一个动态决策问题。交易者在每个时间点都需要根据市场状态决定是否买入、卖出、持有或做空。这个过程具有明显的 sequential structure：当前动作不仅影响当前收益，还会影响下一期持仓、交易成本和未来风险暴露。

传统金融建模常常关注价格预测，例如预测下一期价格涨跌或收益率大小。但交易策略学习不等同于价格预测。一个好的交易策略需要回答的问题是：在当前状态下应该持有多少仓位，才能让长期收益和风险之间达到较好平衡。因此，本项目将交易问题写成强化学习中的 Markov Decision Process。

在强化学习框架中：

- State 是交易者能观察到的市场信息。
- Action 是交易者选择的仓位。
- Reward 是该动作带来的交易收益扣除风险或成本后的结果。
- Policy 是从 state 到 action 的映射。
- Objective 是最大化长期折扣总 reward。

本项目选择模拟市场而不直接使用真实股票数据。原因有三点。第一，模拟环境可以让我们控制参数，例如均值回复速度、波动率和惩罚系数。第二，模拟环境可以生成独立 evaluation paths，避免训练路径和测试路径混在一起。第三，Reward (1) 可以推导解析最优策略，因此可以用理论 benchmark 判断 RL 策略是否学到合理结构。

## 2. 市场模型与交易环境

股票价格记为 \(S_t\)，初始价格为 \(S_0=1\)。定义对数收益率过程 \(L_t\)，它服从均值回复动态：

```text
L_{t+1} = (1 - kappa) L_t + sigma Z_t,  Z_t ~ N(0, 1)
S_{t+1} = S_t exp(L_{t+1})
```

其中，\(\kappa\) 控制均值回复速度，\(\sigma\) 控制波动率。如果 \(\kappa\) 较大，\(L_t\) 会更快回到 0；如果 \(\sigma\) 较大，价格路径会更加波动。

交易动作 \(A_t\) 表示第 \(t\) 期持有的股票数量。它是连续动作，可以为正也可以为负：

- \(A_t > 0\)：做多。
- \(A_t < 0\)：做空。
- \(A_t = 0\)：空仓。

为了保证训练稳定，代码中会把动作限制在固定区间内。否则神经网络可能输出极端仓位，导致 reward 和梯度变得非常不稳定。

## 3. MDP 形式化

本项目可以写成如下 MDP：

```text
State:  s_t
Action: A_t
Reward: R_t
Transition: price dynamics + next state update
Policy: pi(A_t | s_t)
Objective: maximize E[sum_t gamma^t R_t]
```

对于 Reward (1)，状态为：

```text
s_t = (log S_t, L_t)
```

对于 Reward (2)，状态为：

```text
s_t = (log S_t, L_t, A_{t-1})
```

Reward (2) 必须包含上一期动作 \(A_{t-1}\)，因为交易成本项依赖仓位变化 \(A_t - A_{t-1}\)。如果不把 \(A_{t-1}\) 放进状态，智能体就无法仅凭当前 state 判断换仓成本，这会破坏 Markov property。

## 4. Reward 设计

### 4.1 Reward (1)：持仓惩罚

Reward (1) 定义为：

```text
R_t = A_t(S_{t+1} - S_t) - lambda A_t^2
```

第一项 \(A_t(S_{t+1}-S_t)\) 是交易利润。如果持有正仓位且价格上涨，则收益为正；如果做空且价格下跌，也可以获得正收益。

第二项 \(\lambda A_t^2\) 是持仓惩罚。它限制策略持有过大的绝对仓位，避免智能体为了追求短期收益而无限加杠杆。

Reward (1) 的重要特点是动作 \(A_t\) 不会影响未来价格动态，因此可以推导解析最优动作。给定当前状态，最优动作大致与条件期望价格变化成正比，并受到 \(\lambda\) 控制。这个解析策略在实验中作为 benchmark。

### 4.2 Reward (2)：换仓惩罚

Reward (2) 定义为：

```text
R_t = A_t(S_{t+1} - S_t) - lambda(A_t - A_{t-1})^2 S_t
```

这里的惩罚项不再惩罚仓位大小，而是惩罚仓位变化。它更接近真实交易中的 transaction cost，因为真实市场中频繁买卖通常会产生手续费、滑点和冲击成本。

Reward (2) 的策略一般应该更平滑。也就是说，如果上一期已经持有某个仓位，除非市场信号足够强，否则策略不应该频繁大幅调整仓位。

## 5. 数据协议：Oracle 与 Sequential

本项目比较两种数据设定。

### 5.1 Simulation Oracle

在 oracle setting 中，训练算法可以不断从已知价格模型中生成新的 episode。每个 episode 都是独立模拟路径。这个设定比较理想化，相当于我们拥有一个可以无限采样的市场模拟器。

它的优点是数据覆盖更充分，训练更稳定。缺点是它比真实金融数据环境更乐观，因为现实中我们通常不能无限生成真实市场路径。

### 5.2 Sequential Data

在 sequential setting 中，训练数据来自固定的价格路径。这个设定更接近历史数据训练，因为真实市场数据通常是一条或少量时间序列，而不是无限独立样本。

Sequential setting 的难点是样本覆盖有限，而且数据之间高度相关。对 off-policy 方法来说，通常需要通过 behavior policy 收集 replay buffer，再进行离线或半离线更新。

## 6. 算法详细介绍

## 6.1 PPO

PPO，全称 Proximal Policy Optimization，是一种 on-policy policy gradient 方法。它直接学习 stochastic policy，也就是给定 state 后输出 action distribution。

PPO 的核心思想是：每次更新策略时，不允许新策略偏离旧策略太远。它通过 clipped objective 实现这一点：

```text
min(r_t advantage_t, clip(r_t, 1-epsilon, 1+epsilon) advantage_t)
```

其中 \(r_t\) 是新旧策略概率比。如果新策略改变太大，目标函数会被 clip，从而限制更新幅度。

在本项目中，PPO 使用 actor 网络输出连续动作分布，critic 网络估计 value function。它的优点是训练相对稳定，缺点是 on-policy 数据不能反复使用，因此样本效率较低。

从结果看，PPO 的策略通常更保守。它的 turnover 和平均绝对仓位较低，但 mean return 也较低。这说明 PPO 在当前训练预算下没有充分学到利用均值回复信号的激进交易策略。

## 6.2 TD3

TD3，全称 Twin Delayed Deep Deterministic Policy Gradient，是一种 off-policy deterministic actor-critic 算法。它适合连续动作问题。

TD3 有三个关键机制：

1. Twin critics：同时训练两个 Q 网络，target value 取二者较小值，从而减少 Q-value overestimation。
2. Delayed policy update：critic 更新多次后再更新 actor，使 Q 函数更稳定。
3. Target policy smoothing：给 target action 加噪声，减少策略对 Q 函数局部尖峰的过拟合。

在本项目中，TD3 的 actor 直接输出确定性仓位 \(A_t\)，critic 估计 \(Q(s_t, A_t)\)。由于 TD3 使用 replay buffer，它可以反复利用历史 transition，因此样本效率较高。

从结果看，TD3 在 baseline R1 oracle 下 mean return 为 1302.14，达到解析 benchmark 的 68.71%。在 baseline R1 sequential 下，TD3 的 mean return 为 1381.05，也明显优于 PPO 和 SAC。说明 TD3 在这个连续动作交易环境中表现较强。

## 6.3 SAC

SAC，全称 Soft Actor-Critic，也是一种 off-policy actor-critic 算法。和 TD3 不同，SAC 学习 stochastic policy，并在目标函数中加入 entropy term。

SAC 的目标不仅是最大化 reward，也要最大化策略熵：

```text
expected reward + alpha * entropy
```

这种设计鼓励策略保持探索，不要太快收敛到单一动作。对于金融交易这种噪声较大、局部最优较多的问题，entropy regularization 有助于探索更广泛的策略空间。

在本项目中，SAC 在 baseline R1 oracle 下 mean return 为 1212.14，接近 TD3。在 R2 oracle 下，SAC 的 mean return 为 1295.27，是该组最高，但 turnover 达到 12.07，明显高于 TD3 和 PPO。这说明 SAC 可能通过更频繁的仓位调整获得较高 reward，但交易行为更激进。

## 7. 实验设置

实验结果来自 `results_main.csv`。实验覆盖：

| 项目 | 设置 |
| --- | --- |
| 参数组 | baseline, slow_reversion, high_volatility, strong_regularization |
| Seeds | 0, 1 |
| Rewards | R1, R2 |
| Data settings | simulation oracle, sequential |
| Algorithms | PPO, TD3, SAC |
| Evaluation paths | 200 |
| Evaluation horizon | 80 |

主要指标如下：

- `mean_return`：Monte Carlo 平均折扣总 reward。
- `std_error`：Monte Carlo 标准误，反映评估不确定性。
- `turnover`：平均换手幅度，反映交易频率和仓位调整强度。
- `avg_abs_position`：平均绝对仓位，反映策略激进程度。
- `relative_to_analytic_pct`：只对 R1 有意义，表示 learned policy 达到解析 benchmark 的百分比。
- `policy_mse_to_analytic`：只对 R1 有意义，表示 learned policy 与解析策略曲线之间的均方误差。

需要强调的是，`mean_return` 不是股票收益率，而是：

```text
E[sum_t gamma^t R_t]
```

因此它包含交易利润，也包含 reward 中的惩罚项。

## 8. 图片解释

### 8.1 Sanity trajectory

![Sanity trajectory](fig_sanity_trajectory.png)

这张图用于检查环境是否正确。左图展示价格路径 \(S_t\)，中间展示均值回复收益率 \(L_t\)，右图展示随机策略下的动作。重点不是看收益，而是确认模拟环境能生成合理轨迹。

如果 \(L_t\) 长期偏离 0，就说明均值回复过程可能有问题；如果动作超过边界，就说明 action clipping 可能失效。本图显示环境基本能够正常运行。

### 8.2 Training curves

![Training curves](fig_training_curves.png)

训练曲线展示 baseline R1 oracle 下 PPO、TD3、SAC 的训练过程。它可以反映训练是否发生崩溃、是否有学习趋势、不同算法 reward 波动是否明显。

不过训练曲线不能作为最终性能判断。金融 RL 的训练 reward 噪声很大，而且训练路径和 evaluation paths 不完全相同。因此最终仍要看独立 Monte Carlo evaluation 的 `mean_return`。

### 8.3 Reward (1) policy comparison

![Reward 1 policy comparison](fig_reward1_policy_comparison.png)

这张图是本项目最重要的可视化之一。Reward (1) 有解析策略，所以图中可以比较 analytic policy 与 PPO、TD3、SAC 学到的策略。

在均值回复市场中，策略应该根据 \(L_t\) 调整仓位。如果当前收益率信号表示未来价格有上行机会，策略应增加做多仓位；如果信号相反，策略应减少仓位或做空。learned policy 越接近 analytic curve，说明算法越可能学到正确结构。

### 8.4 Reward (1) policy heatmap

![Reward 1 policy heatmap](fig_reward1_policy_heatmap.png)

Heatmap 展示不同 \(\log S_t\) 和 \(L_t\) 下策略选择的 \(A_t\)。颜色代表仓位大小。它比单条曲线更全面，因为策略可能同时依赖价格水平和收益率。

如果 heatmap 呈现平滑结构，说明策略对状态变化的响应较连续；如果出现大面积极端颜色，说明策略可能倾向于长期满仓或满仓做空。

### 8.5 Reward (2) policy heatmap

![Reward 2 policy heatmap](fig_reward2_policy_heatmap.png)

Reward (2) 的 heatmap 更复杂，因为 state 包含上一期仓位 \(A_{t-1}\)。因此该图展示的是当前收益率 \(L_t\)、上一期仓位和当前动作之间的关系。

如果交易成本惩罚有效，策略应该表现出 inertia，也就是不会轻易从大多头跳到大空头。换句话说，动作应该对 \(A_{t-1}\) 有依赖，仓位调整应更加平滑。

## 9. Baseline 结果解释

下面表格聚合 baseline 参数组下两个 seed 的平均结果。

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

### 9.1 R1 oracle

R1 oracle 是最标准的比较，因为它有 simulation oracle 和 analytic benchmark。在这个 setting 下，TD3 表现最好，SAC 次之，PPO 最弱。

TD3 的高 reward 说明它能有效利用均值回复信号。SAC 的 reward 接近 TD3，但 turnover 更高。PPO 的 turnover 和仓位都较低，说明策略更保守，但这也导致它无法获得足够交易收益。

### 9.2 R1 sequential

Sequential setting 更接近历史数据训练。结果中 TD3 仍然表现最好，PPO 有较低正收益，而 SAC 在 baseline 下为负。

这说明 sequential setting 下算法对数据覆盖和 replay buffer 质量更敏感。SAC 的随机策略和 entropy 机制在固定数据分布下可能更难稳定。

### 9.3 R2 oracle

R2 oracle 下 SAC 的 mean return 最高，但 turnover 也最高。它可能通过频繁调整仓位获得较高 reward，但这也说明策略行为较激进。

TD3 的收益略低，但 turnover 明显低于 SAC，因此更平衡。PPO 仍然最保守，收益为负。

### 9.4 R2 sequential

R2 sequential 下 TD3 表现最好，mean return 为 1399.72，turnover 只有 0.12。这个结果说明 TD3 在固定数据环境和换仓惩罚下仍然能找到较稳定的策略。

SAC 在该 setting 下平均收益较低，PPO 仍为负。总体上，R2 sequential 更强调稳定换仓，而 TD3 的 deterministic policy 在这里更有优势。

## 10. 不同算法对比

### PPO

PPO 的特点是稳定、保守、更新幅度受限制。它适合需要稳定训练的任务，但在本项目中收益明显较低。原因可能是训练预算有限，on-policy 方法样本效率不足。PPO 每次只能用当前策略采样出来的数据更新，历史数据不能像 TD3/SAC 那样反复利用。

从结果看，PPO 的平均仓位较小，turnover 通常也较低。这符合它保守的行为特征。但在交易任务中，过度保守会导致策略无法充分捕捉价格均值回复带来的机会。

### TD3

TD3 在多数设置中表现最稳定。它在 baseline R1 oracle 和 R1 sequential 下都是最佳或接近最佳。在 R2 sequential 下，TD3 也取得最高 mean return，同时 turnover 很低。

TD3 的优势来自 replay buffer 和 twin critics。Replay buffer 提高样本利用率，twin critics 减少过估计，使 actor 更新更可靠。它输出 deterministic action，也让策略在交易问题中更直接。

### SAC

SAC 的优势是探索能力强。它通过 entropy regularization 避免策略过早收敛。在 R1 oracle 下，SAC 接近 TD3。在 R2 oracle 下，SAC 取得最高 mean return。

但 SAC 的问题是策略可能更激进。例如 baseline R2 oracle 下，SAC 的 turnover 达到 12.07，远高于 TD3 的 2.53 和 PPO 的 0.14。这意味着它的交易频率高，实际市场中如果交易成本更真实，表现可能会下降。

### 总体比较

综合来看：

- TD3 是本实验中最稳健的算法。
- SAC 在部分 setting 下收益很高，但 turnover 可能过大。
- PPO 最保守，训练稳定但样本效率不足。

如果目标是教学展示，三者对比非常清楚：PPO 代表稳定 on-policy 方法，TD3 代表 deterministic off-policy 方法，SAC 代表 entropy-regularized stochastic off-policy 方法。

## 11. 参数组结果与环境敏感性

Reward (1) 的解析 benchmark 如下：

| Parameter set | Analytic mean return | Std. error | Turnover | Avg. abs. position |
| --- | --- | --- | --- | --- |
| baseline | 1895.18 | 1530.11 | 2.49 | 4.86 |
| slow_reversion | 5.224e+16 | 5.224e+16 | 1.43 | 5.32 |
| high_volatility | 8.111e+08 | 8.063e+08 | 2.68 | 5.56 |
| strong_regularization | 1880.77 | 1529.97 | 1.40 | 2.76 |

slow_reversion 和 high_volatility 下 reward 非常大，原因是价格模型是指数形式：

```text
S_t = exp(log S_t)
```

当收益率长时间偏离或波动率较大时，价格可能产生极端路径。这说明模型对参数较敏感，也提醒我们不能把这些数值直接理解为真实投资收益。

在 R1 oracle 下，各参数组算法表现如下：

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

可以看到，TD3 和 SAC 在大部分参数组下显著优于 PPO。PPO 的 relative performance 通常只有 17% 到 22% 左右，说明它在有限训练预算下没有学到足够强的交易策略。

## 12. 局限性与未来展望

### 12.1 模拟市场过于简化

本项目使用的是一维均值回复模型。真实金融市场远比这个模型复杂，包含趋势、跳跃、波动聚集、流动性变化、宏观事件冲击和非平稳性。因此，本项目结果只能说明算法在可控环境中的行为，不能直接推广到真实市场。

### 12.2 交易成本不够真实

Reward (2) 用二次换仓惩罚近似交易成本，但真实交易成本包括 bid-ask spread、commission、slippage 和 market impact。这些成本通常不是简单二次函数，也会随市场流动性变化。

### 12.3 风险指标不足

当前主要指标是 mean discounted total reward。真实投资策略还需要关注 Sharpe ratio、maximum drawdown、tail risk、VaR、CVaR 和资金利用率。一个 mean return 高的策略，如果回撤过大，也可能不可接受。

### 12.4 随机性和统计显著性有限

实验只使用两个 seed，虽然可以展示算法行为，但不足以给出非常稳健的统计结论。后续应增加 seed 数量，并报告置信区间。

### 12.5 价格路径可能产生极端值

由于 \(S_t=\exp(\log S_t)\)，在 high_volatility 和 slow_reversion 参数组下，reward 数值可能非常大。这是模型本身的数值特性，不应解释为真实市场中的可实现收益。未来可以加入 log-price clipping 或直接在 return space 中定义 reward。

### 12.6 未来工作

后续可以从以下方向改进：

- 使用真实股票或 ETF 数据进行 backtesting。
- 加入 transaction cost、slippage 和 capital constraints。
- 增加更多 seed 和 evaluation paths。
- 比较更多算法，例如 DDPG、A2C、DQN 离散仓位版本。
- 加入 risk-adjusted objective，例如 Sharpe ratio 或 mean-variance reward。
- 做系统的 hyperparameter ablation，例如 learning rate、network size、batch size 和 replay buffer size。
- 使用 walk-forward validation 检查策略在不同市场时期的稳定性。

## 13. 总结

本项目展示了如何把金融交易问题形式化为强化学习问题。智能体观察市场状态，选择连续仓位，并通过 reward 学习长期交易策略。通过比较 PPO、TD3 和 SAC，可以看到不同 RL 算法在同一交易环境下会产生明显不同的行为。

实验的主要结论是：

1. TD3 和 SAC 在多数设置下明显优于 PPO。
2. PPO 更保守，但样本效率较低，收益较弱。
3. TD3 在 R1 和 R2 的多个 setting 下表现稳定。
4. SAC 有较强探索能力，部分 setting 下收益高，但 turnover 可能过大。
5. Reward 设计会显著影响策略行为。
6. Analytic benchmark 对验证 RL 策略是否合理非常重要。

从课程项目角度看，这个实验完整展示了金融 RL 的基本流程：环境建模、reward 设计、算法实现、Monte Carlo evaluation、policy visualization 和结果解释。它不是一个可直接用于真实交易的系统，而是一个清晰的研究和教学 prototype。
