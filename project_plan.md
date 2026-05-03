# 项目计划：Comparing RL Methods in Simulated Mean-Reverting Stock

## 1. 项目目标

本项目比较不同强化学习方法在模拟均值回复股票交易环境中的表现。主实验完全使用题目给定的价格模型生成模拟数据，不使用真实股票数据。

核心比较维度包括：

- 两种奖励函数：`Reward (1)` 和 `Reward (2)`。
- 两种数据设置：`simulation oracle` 和 `sequential data`。
- 两类 RL 方法：on-policy 方法 `PPO` 和 off-policy 方法 `TD3`。
- `Reward (1)` 的解析最优策略与 learned policy 的对比。
- 使用独立模拟路径进行 Monte Carlo 评估。
- 可视化 learned policy，并完成至少一个训练超参数 ablation study。

最终交付物：

- Jupyter notebook：包含可运行代码、实验结果表格和图。
- Written report：解释模型、方法、实验设计、结果和结论。

## 2. Assignment Compliance Matrix

| 作业要求 | 本项目中的对应设计 |
| --- | --- |
| 描述 simulation oracle 和 sequential data 下的 trajectory sampling | 第 6 节定义两种数据协议，并说明 PPO/TD3 的数据收集方式。 |
| 若使用 off-policy 方法，说明 data collection policy | 第 6.3 节定义 TD3 在 sequential data 下的 behavior policy。 |
| 描述 RL 方法和函数逼近结构 | 第 8 节和第 9 节分别说明 PPO、TD3 的 actor/critic 输入、输出、网络和更新过程。 |
| 分别处理 Reward (1) 和 Reward (2) | 第 4 节定义两个 reward，第 11 节给出完整实验矩阵。 |
| 对 Reward (1) 推导理论最优策略 | 第 7 节给出 closed-form analytic benchmark。 |
| 测试多组参数 `(kappa, sigma, lambda, gamma)` | 第 10 节给出 baseline 和 parameter sensitivity sets。 |
| 使用独立模拟路径评估最终策略 | 第 12 节规定 evaluation paths 必须独立于训练路径。 |
| 报告 Monte Carlo mean discounted truncated total reward | 第 12 节定义 `mean_return` 和 `std_error`。 |
| 可视化 learned policy | 第 14 节定义 Reward (1) policy curve/heatmap 和 Reward (2) heatmap。 |
| 完成至少一个 ablation study | 第 15 节定义 learning rate ablation。 |

## 3. 市场模型

### 3.1 价格动态

股票价格为 $S_t$，初始价格为：

$$
S_0 = 1.
$$

定义对数收益率：

$$
L_t = \log \frac{S_t}{S_{t-1}}.
$$

题目给定 $L_t$ 服从一维均值回复过程：

$$
L_{t+1} = L_t - \kappa L_t + \sigma Z_t
= (1-\kappa)L_t + \sigma Z_t,
$$

其中：

$$
\kappa \in (0,1), \qquad \sigma > 0, \qquad Z_t \sim N(0,1),
$$

且 $Z_t$ 相互独立。价格更新为：

$$
S_{t+1} = S_t e^{L_{t+1}}.
$$

初始状态采用：

$$
S_0 = 1, \qquad L_0 = 0, \qquad A_{-1} = 0.
$$

### 3.2 动作空间

动作 $A_t$ 表示第 $t$ 期持有的股票数量。题目允许 fractional shares，因此理论上：

$$
A_t \in \mathbb{R}.
$$

为保证神经网络训练稳定，实际实现中使用 bounded continuous action：

$$
A_t \in [-A_{\max}, A_{\max}].
$$

默认设置：

$$
A_{\max} = 10.
$$

可选 sensitivity analysis 使用：

$$
A_{\max} \in \{5, 10, 20\}.
$$

## 4. Reward 设定

### 4.1 Reward (1)：Position Regularization

第一种单期奖励为：

$$
R_t = A_t(S_{t+1}-S_t) - \lambda A_t^2,
$$

其中 $\lambda > 0$ 惩罚过大仓位。由于 $A_t$ 不影响未来价格过程，`Reward (1)` 可以推导解析最优策略，作为 RL benchmark。

### 4.2 Reward (2)：Transaction Cost

第二种单期奖励为：

$$
R_t = A_t(S_{t+1}-S_t) - \lambda(A_t-A_{t-1})^2S_t,
$$

其中：

$$
A_{-1}=0.
$$

该 reward 惩罚仓位变化，近似 stylized transaction cost。为了保持 Markov property，`Reward (2)` 的状态必须包含上一期动作 $A_{t-1}$。

## 5. 优化目标与截断

目标是找到策略 $\pi$，最大化折扣总收益：

$$
\max_{\pi}
\mathbb{E}^{\pi}\left[
\sum_{t=0}^{\infty}\gamma^t R_t
\mid S_0=1, L_0=0, A_{-1}=0
\right],
$$

其中：

$$
\gamma \in (0,1).
$$

实现中使用 finite horizon approximation。推荐截断长度：

$$
T = \left\lceil \frac{\log(10^{-4})}{\log(\gamma)} \right\rceil.
$$

默认计算预算：

| Discount factor | Recommended horizon |
| ---: | ---: |
| $\gamma = 0.95$ | `T = 200` |
| $\gamma = 0.99$ | `T = 500` for fast runs, `T = 1000` for final runs |

## 6. 状态与数据协议

### 6.1 State Representation

`Reward (1)` 使用：

$$
x_t = (\log S_t, L_t).
$$

Notebook 中对应：

```text
state_reward1 = [log_S_t, L_t]
```

`Reward (2)` 使用：

$$
x_t = (\log S_t, L_t, A_{t-1}).
$$

Notebook 中对应：

```text
state_reward2 = [log_S_t, L_t, A_prev]
```

使用 $\log S_t$ 而不是 $S_t$ 可以减少价格尺度漂移带来的数值问题。

### 6.2 Simulation Oracle Setting

在 `simulation oracle` setting 中，训练算法可以随时调用模型生成新的独立 episode。每个 episode 重置为：

$$
S_0 = 1,\qquad L_0 = 0,\qquad A_{-1}=0.
$$

采样流程：

```text
for each episode:
    reset S_0 = 1, L_0 = 0, A_{-1} = 0
    for t = 0, ..., T - 1:
        choose action A_t from current policy
        sample Z_t ~ N(0, 1)
        update L_{t+1}
        update S_{t+1}
        compute R_t
        store transition
```

PPO 在该 setting 下使用当前 policy 收集 on-policy rollout。TD3 使用当前 deterministic actor 加 exploration noise 与模拟器交互，并将 transitions 存入 replay buffer。

### 6.3 Sequential Data Setting

在 `sequential data` setting 中，训练时不能任意 reset，也不能生成无限独立 trajectories。训练数据来自一条或少量固定长路径：

$$
\{(S_t,L_t)\}_{t=0}^{T_{\text{data}}}.
$$

默认：

```text
T_data = 100000
```

PPO sequential protocol：

- 沿固定价格路径按时间顺序收集 contiguous rollout。
- 当前 policy 只能在给定路径上选择动作并获得 reward。
- 一个 rollout 到达路径末端后才允许回到路径起点或切换到另一条预先生成的固定路径。

TD3 sequential protocol：

- 先用 behavior policy 沿固定路径收集 transitions。
- 将 transitions 存入 replay buffer。
- TD3 只从 replay buffer 更新，不调用 fresh simulation。

默认 behavior policy：

$$
A_t^b = \operatorname{clip}(cL_t + \epsilon_t, -A_{\max}, A_{\max}),
$$

其中：

$$
c = 2,\qquad \epsilon_t \sim N(0,\sigma_A^2),\qquad \sigma_A = 1.
$$

如果该 behavior policy 覆盖不足，可增加 noise 或改用：

$$
A_t^b = \operatorname{clip}(\epsilon_t, -A_{\max}, A_{\max}).
$$

## 7. Reward (1) 的理论最优策略

`Reward (1)` 为：

$$
R_t = A_t(S_{t+1}-S_t) - \lambda A_t^2.
$$

由于动作不影响未来价格动态，每期可以单独最大化条件期望 reward：

$$
\max_{A_t}
\mathbb{E}_t\left[
A_t(S_{t+1}-S_t)-\lambda A_t^2
\right].
$$

由模型可知：

$$
L_{t+1}\mid L_t \sim N((1-\kappa)L_t,\sigma^2),
$$

且：

$$
S_{t+1} = S_t e^{L_{t+1}}.
$$

因此：

$$
\mathbb{E}_t[S_{t+1}]
= S_t\mathbb{E}_t[e^{L_{t+1}}]
= S_t e^{(1-\kappa)L_t+\frac{1}{2}\sigma^2}.
$$

所以：

$$
\mathbb{E}_t[S_{t+1}-S_t]
= S_t\left(e^{(1-\kappa)L_t+\frac{1}{2}\sigma^2}-1\right).
$$

令：

$$
m_t = S_t\left(e^{(1-\kappa)L_t+\frac{1}{2}\sigma^2}-1\right).
$$

需要最大化：

$$
A_tm_t - \lambda A_t^2.
$$

一阶条件为：

$$
m_t - 2\lambda A_t = 0.
$$

理论最优策略为：

$$
A_t^* =
\frac{
S_t\left(e^{(1-\kappa)L_t+\frac{1}{2}\sigma^2}-1\right)
}{2\lambda}.
$$

考虑实际动作范围后，benchmark 使用 clipped analytic policy：

$$
A_{t,\text{clipped}}^*
= \operatorname{clip}(A_t^*, -A_{\max}, A_{\max}).
$$

该策略用于：

- `Reward (1)` 的 analytic benchmark。
- `Reward (1)` 的 policy visualization 对比。
- `Reward (1)` 的 relative performance、regret 和 policy MSE 计算。

## 8. RL 方法一：PPO

### 8.1 使用场景

PPO 作为 on-policy 方法，分别训练：

- `Reward (1), simulation oracle`
- `Reward (1), sequential data`
- `Reward (2), simulation oracle`
- `Reward (2), sequential data`

### 8.2 函数逼近结构

Actor 输入 state，输出 Gaussian policy 的 mean。Critic 输入 state，输出 state value。

输入维度：

| Reward | State | Dimension |
| --- | --- | ---: |
| Reward (1) | `[log_S_t, L_t]` | 2 |
| Reward (2) | `[log_S_t, L_t, A_prev]` | 3 |

Actor：

```text
state
-> Linear(hidden_dim)
-> Tanh
-> Linear(hidden_dim)
-> Tanh
-> Linear(1)
-> action mean
```

使用可学习的 `log_std`：

$$
A_t \sim N(\mu_\theta(x_t), \sigma_\theta^2),
$$

然后将动作 clip 到 $[-A_{\max}, A_{\max}]$。

Critic：

```text
state
-> Linear(hidden_dim)
-> Tanh
-> Linear(hidden_dim)
-> Tanh
-> Linear(1)
-> V_phi(state)
```

默认超参数：

| Hyperparameter | Value |
| --- | ---: |
| `hidden_dim` | 64 |
| `learning_rate` | `3e-4` |
| `batch_size` | 64 |
| `rollout_steps` | 2048 |
| `ppo_epochs` | 10 |
| `clip_epsilon` | 0.2 |
| `gae_lambda` | 0.95 |
| `entropy_coef` | 0.0 or 0.01 |
| `value_coef` | 0.5 |

### 8.3 PPO 更新过程

每次训练：

1. 用当前 policy 收集 rollout transitions。
2. 存储 `state, action, reward, next_state, log_prob, value, done`。
3. 计算 discounted return。
4. 使用 GAE 计算 advantage。
5. 标准化 advantage。
6. 多个 epoch 更新 actor 和 critic。

PPO clipped objective：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}\left[
\min\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right],
$$

其中：

$$
r_t(\theta)
=
\frac{\pi_\theta(A_t|x_t)}
{\pi_{\theta_{\text{old}}}(A_t|x_t)}.
$$

Critic loss：

$$
L_V(\phi)
=
\mathbb{E}\left[
(V_\phi(x_t)-\hat{G}_t)^2
\right].
$$

总 loss：

$$
L
=
-L^{\text{CLIP}}
+ c_v L_V
- c_e H(\pi_\theta).
$$

## 9. RL 方法二：TD3

### 9.1 使用场景

TD3 作为 off-policy 方法，分别训练：

- `Reward (1), simulation oracle`
- `Reward (1), sequential data`
- `Reward (2), simulation oracle`
- `Reward (2), sequential data`

SAC 可作为 optional extension，但不纳入主实验，避免项目范围膨胀。

### 9.2 函数逼近结构

TD3 包含：

- Actor network。
- Twin critics：`Q1` 和 `Q2`。
- Target actor。
- Target critics。
- Replay buffer。

Actor 输入 state，输出 deterministic action：

$$
A_t = \mu_\theta(x_t).
$$

Actor：

```text
state
-> Linear(hidden_dim)
-> ReLU
-> Linear(hidden_dim)
-> ReLU
-> Linear(1)
-> Tanh
-> multiply by A_max
```

Critic 输入 state-action pair，输出 Q value：

```text
[state, action]
-> Linear(hidden_dim)
-> ReLU
-> Linear(hidden_dim)
-> ReLU
-> Linear(1)
-> Q-value
```

默认超参数：

| Hyperparameter | Value |
| --- | ---: |
| `hidden_dim` | 256 |
| `actor_lr` | `1e-3` |
| `critic_lr` | `1e-3` |
| `batch_size` | 256 |
| `replay_buffer_size` | `1_000_000` |
| `tau` | 0.005 |
| `policy_delay` | 2 |
| `target_noise` | 0.2 |
| `noise_clip` | 0.5 |
| `exploration_noise` | `0.1 * A_max` |

### 9.3 TD3 更新过程

Replay buffer 存储：

```text
(state, action, reward, next_state, done)
```

每次更新：

1. 从 replay buffer 采样 batch。
2. 使用 target actor 计算 next action：

$$
A' = \mu_{\theta'}(x') + \epsilon,
$$

其中：

$$
\epsilon \sim \operatorname{clip}(N(0,\sigma_{\text{target}}^2), -c, c).
$$

3. 将 $A'$ clip 到动作范围。
4. 计算 target Q：

$$
y =
r + \gamma(1-d)
\min\left(
Q_{\phi_1'}(x',A'),
Q_{\phi_2'}(x',A')
\right).
$$

5. 更新两个 critic：

$$
L_Q =
(Q_{\phi_1}(x,A)-y)^2
+
(Q_{\phi_2}(x,A)-y)^2.
$$

6. 每隔 `policy_delay` 步更新 actor：

$$
L_{\text{actor}}
=
-\mathbb{E}\left[
Q_{\phi_1}(x,\mu_\theta(x))
\right].
$$

7. 软更新 target networks：

$$
\theta' \leftarrow \tau\theta + (1-\tau)\theta',
\qquad
\phi' \leftarrow \tau\phi + (1-\tau)\phi'.
$$

## 10. 实验参数与计算预算

### 10.1 Baseline 参数

默认参数：

$$
\kappa=0.2,\qquad
\sigma=0.1,\qquad
\lambda=0.01,\qquad
\gamma=0.95.
$$

### 10.2 Parameter Sensitivity Sets

不要运行过大的 full grid。采用 baseline 加一次改变一个参数的方式。

| Experiment | $\kappa$ | $\sigma$ | $\lambda$ | $\gamma$ | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.2 | 0.1 | 0.01 | 0.95 | 基准 |
| Slow mean reversion | 0.05 | 0.1 | 0.01 | 0.95 | 均值回复慢 |
| Fast mean reversion | 0.5 | 0.1 | 0.01 | 0.95 | 均值回复快 |
| Low volatility | 0.2 | 0.05 | 0.01 | 0.95 | 低波动 |
| High volatility | 0.2 | 0.2 | 0.01 | 0.95 | 高波动 |
| Strong regularization | 0.2 | 0.1 | 0.05 | 0.95 | 更强仓位惩罚 |
| High discount | 0.2 | 0.1 | 0.01 | 0.99 | 更重视长期收益 |

### 10.3 Reproducibility Defaults

| Item | Default |
| --- | --- |
| Random seeds | `[0, 1, 2]` for final tables; seed `0` for fast debugging |
| Device | CPU by default; GPU optional if available |
| `N_eval` | 1000 independent evaluation paths |
| Fast-debug training budget | 20% of final steps |
| Final training budget | enough for stable training curves under baseline |
| Saved tables | `results_main.csv`, `results_ablation.csv` |
| Saved figures | `fig_reward1_policy_comparison.png`, `fig_reward2_policy_heatmap.png`, `fig_training_curves.png`, `fig_ablation_learning_rate.png` |

## 11. 实验矩阵

### 11.1 Main Matrix

`Reward (1)`：

| Data setting | Method | Benchmark |
| --- | --- | --- |
| Simulation oracle | Analytic optimal policy | yes |
| Simulation oracle | PPO | compare to analytic |
| Simulation oracle | TD3 | compare to analytic |
| Sequential data | PPO | compare to analytic |
| Sequential data | TD3 | compare to analytic |

`Reward (2)`：

| Data setting | Method | Benchmark |
| --- | --- | --- |
| Simulation oracle | PPO | no analytic benchmark |
| Simulation oracle | TD3 | no analytic benchmark |
| Sequential data | PPO | no analytic benchmark |
| Sequential data | TD3 | no analytic benchmark |

### 11.2 Minimum Viable Matrix

如果计算时间有限，最小可行版本必须覆盖：

- 两种 reward。
- 两种 data setting。
- PPO 和 TD3。
- `Reward (1)` analytic benchmark。
- 独立 Monte Carlo evaluation。
- 至少一个 ablation study。

推荐最小参数组：

- Baseline。
- High volatility。
- Strong regularization。

### 11.3 Full Extension Matrix

时间允许时，扩展到第 10.2 节所有参数组，并对 `Reward (1)` 和 `Reward (2)` 分别报告 oracle/sequential、PPO/TD3 的结果。

## 12. Evaluation Protocol

最终 policy 评估必须使用独立生成的模拟路径，不能复用训练路径或 sequential training path。

每个训练好的 policy 生成：

```text
N_eval = 1000
```

如果计算资源允许：

```text
N_eval = 5000
```

每条评估路径长度为：

$$
T_{\text{eval}}
=
\left\lceil \frac{\log(10^{-4})}{\log(\gamma)} \right\rceil.
$$

每条路径计算 discounted return：

$$
G_i =
\sum_{t=0}^{T_{\text{eval}}-1}
\gamma^t R_t^{(i)}.
$$

Monte Carlo mean：

$$
\bar{G}
=
\frac{1}{N_{\text{eval}}}
\sum_{i=1}^{N_{\text{eval}}}
G_i.
$$

Standard error：

$$
SE
=
\frac{\operatorname{std}(G_i)}
{\sqrt{N_{\text{eval}}}}.
$$

`Reward (1)` 的 relative performance：

$$
\text{Relative Performance}
=
\frac{\bar{G}_{\text{RL}}}{\bar{G}_{\text{analytic}}}
\times 100\%.
$$

`Reward (1)` 的 regret：

$$
\text{Regret}
=
\bar{G}_{\text{analytic}} - \bar{G}_{\text{RL}}.
$$

额外指标：

- `return_std`：episode return 的标准差。
- `avg_abs_position`：平均绝对仓位。
- `turnover`：平均换手率。
- `sharpe_ratio`：可选。
- `max_drawdown`：可选。

Turnover 定义：

$$
\text{Turnover}
=
\frac{1}{T}
\sum_{t=0}^{T-1}
|A_t-A_{t-1}|.
$$

## 13. 结果表格设计

### 13.1 Main Results Table

| Column | Meaning |
| --- | --- |
| `reward_type` | `R1` or `R2` |
| `parameter_set` | Baseline, High volatility, etc. |
| `method` | Analytic, PPO, TD3 |
| `train_setting` | Oracle or Sequential |
| `eval_setting` | Independent simulation paths |
| `seed` | Random seed |
| `num_eval_paths` | Number of MC paths |
| `T_eval` | Evaluation horizon |
| `mean_return` | Mean discounted truncated return |
| `std_error` | Monte Carlo standard error |
| `relative_to_benchmark` | For Reward (1), analytic benchmark = 100% |
| `regret` | For Reward (1), analytic minus RL return |
| `turnover` | Average turnover |
| `policy_mse_to_analytic` | For Reward (1), MSE against analytic policy |

### 13.2 Parameter Sensitivity Table

| `reward_type` | `method` | `train_setting` | `kappa` | `sigma` | `lambda` | `gamma` | `mean_return` | `std_error` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 | PPO | Oracle | ... | ... | ... | ... | ... | ... |
| R1 | TD3 | Oracle | ... | ... | ... | ... | ... | ... |
| R2 | PPO | Oracle | ... | ... | ... | ... | ... | ... |
| R2 | TD3 | Oracle | ... | ... | ... | ... | ... | ... |

## 14. Policy Visualization

### 14.1 Reward (1) Policy Curve

固定：

$$
S_t = 1.
$$

绘制：

- x-axis：$L_t$。
- y-axis：$A_t$。
- curves：analytic optimal policy、PPO learned policy、TD3 learned policy。

该图用于直接检验 learned policy 是否接近：

$$
A_t^* =
\frac{
S_t\left(e^{(1-\kappa)L_t+\frac{1}{2}\sigma^2}-1\right)
}{2\lambda}.
$$

### 14.2 Reward (1) Heatmap

绘制：

$$
A_t = \pi(\log S_t,L_t).
$$

图像设定：

- x-axis：$L_t$。
- y-axis：$\log S_t$。
- color：action $A_t$。

分别绘制 analytic policy、PPO policy、TD3 policy。

### 14.3 Reward (2) Heatmap

固定：

$$
S_t = 1.
$$

绘制：

$$
A_t = \pi(L_t,A_{t-1}).
$$

图像设定：

- x-axis：$L_t$。
- y-axis：$A_{t-1}$。
- color：action $A_t$。

该图用于观察 transaction cost 是否使策略更平滑。合理预期是：$\lambda$ 越大，$A_t$ 越接近 $A_{t-1}$，平均 turnover 越低。

## 15. Ablation Study

主 ablation 使用 learning rate，因为它容易实现、解释清楚，并直接影响 PPO/TD3 训练稳定性。

默认设置：

- `Reward (1)`。
- `simulation oracle`。
- Baseline model parameters。
- 使用 PPO 作为主 ablation 方法。

比较：

$$
\text{learning rate} \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}\}.
$$

报告：

- Training curve。
- Final Monte Carlo evaluation reward。
- `Reward (1)` learned policy 与 analytic policy 的 policy MSE。

Policy MSE 定义：

$$
\text{Policy MSE}
=
\frac{1}{M}
\sum_{j=1}^M
\left(
\pi_\theta(x_j)-A^*(x_j)
\right)^2.
$$

默认 grid：

```text
S_t = 1
L_t evenly spaced from -0.5 to 0.5
```

可选 ablation：

- Network size：`[32, 32]`, `[64, 64]`, `[128, 128]`。
- Action bound：`A_max = 5, 10, 20`。
- PPO rollout length。
- TD3 replay buffer size。

## 16. Sanity Checks

实现完成后先运行以下 sanity checks，再扩展到完整实验：

- 价格路径检查：确认 $L_t$ 服从均值回复结构，$S_t = S_{t-1}e^{L_t}$。
- Reward 检查：确认 `Reward (1)` 和 `Reward (2)` 的 reward 计算与公式一致。
- 动作范围检查：所有 policy 输出或采样动作都被 clip 到 $[-A_{\max}, A_{\max}]$。
- `A_prev` 检查：`Reward (2)` 中 state 的 `A_prev` 在每次 step 后正确更新。
- Analytic policy 检查：`Reward (1)` analytic policy 的 Monte Carlo return 应明显优于随机策略。
- PPO baseline 检查：先只跑 `Reward (1), oracle, PPO, baseline`，确认 learned policy 曲线接近 analytic policy 后再扩展。
- TD3 baseline 检查：确认 replay buffer 中 transitions 的 reward、next_state、done 与环境一致。

## 17. Notebook 结构

Notebook 按以下顺序组织：

1. Imports and random seeds。
2. Environment implementation。
3. Analytic solution for `Reward (1)`。
4. Data generation：oracle sampler、sequential path generator、behavior policy。
5. PPO implementation。
6. TD3 implementation。
7. Training functions。
8. Evaluation functions。
9. Main experiments。
10. Policy visualization。
11. Ablation study。
12. Summary tables and saved figures。

关键函数：

- `MeanRevertingStockEnv`
- `analytic_policy_reward1`
- `train_ppo_oracle`
- `train_ppo_sequential`
- `train_td3_oracle`
- `train_td3_sequential`
- `evaluate_policy_mc`
- `compute_policy_mse_reward1`

## 18. Written Report 结构

Written report 建议结构：

1. Introduction：项目目标、价格模型、reward、比较维度。
2. Market Environment：价格动态、state/action、discounting、truncation。
3. Data Settings：simulation oracle、sequential data、behavior policy。
4. Analytic Benchmark for `Reward (1)`：推导 closed-form optimal policy。
5. RL Methods：PPO、TD3、网络结构和更新过程。
6. Experimental Design：参数组、训练设置、评估协议。
7. Results：主结果表、oracle vs sequential、PPO vs TD3。
8. Policy Visualization：Reward (1) 对比 analytic；Reward (2) 展示 smoothing。
9. Ablation Study：learning rate ablation。
10. Conclusion：主要发现、限制和可能扩展。

Report 叙事重点：

- `Reward (1)`：解析策略给出可靠 benchmark，可用于说明 RL 是否学到正确交易结构。
- `Reward (2)`：turnover 和 heatmap 用于说明 transaction cost 让策略更平滑。
- `simulation oracle vs sequential data`：oracle 通常数据覆盖更好；sequential data 更接近历史数据限制。
- `PPO vs TD3`：TD3 通常样本效率更高，但更依赖 replay buffer 覆盖和 exploration noise；PPO 通常更稳定。

## 19. 实现顺序

### Phase 1：环境与解析 benchmark

先实现：

- `MeanRevertingStockEnv`
- `analytic_policy_reward1`
- `evaluate_policy_mc`

验收标准：

- 环境 step 正确。
- price path 符合均值回复 return dynamics。
- `Reward (1)` analytic policy 的 evaluation return 合理。

### Phase 2：PPO on Reward (1), Oracle

训练：

- `Reward (1)`
- `simulation oracle`
- `PPO`
- Baseline parameters

验收标准：

- learned policy curve 接近 analytic optimal policy。
- policy MSE 在训练中下降。

### Phase 3：TD3 on Reward (1), Oracle

训练：

- `Reward (1)`
- `simulation oracle`
- `TD3`
- Baseline parameters

验收标准：

- TD3 return 与 analytic benchmark、PPO baseline 可比较。
- replay buffer 数据分布和 reward 计算通过 sanity check。

### Phase 4：Reward (2), Oracle

训练：

- `Reward (2)`
- `simulation oracle`
- `PPO`
- `TD3`

验收标准：

- policy heatmap 显示 $A_t$ 同时依赖 $L_t$ 和 $A_{t-1}$。
- turnover 低于或不同于 `Reward (1)` 的无交易成本设置。

### Phase 5：Sequential Data

实现：

- sequential path generator。
- behavior policy。
- PPO sequential rollout loop。
- TD3 sequential replay buffer。

训练：

- `Reward (1), sequential, PPO`
- `Reward (1), sequential, TD3`
- `Reward (2), sequential, PPO`
- `Reward (2), sequential, TD3`

验收标准：

- 训练时不调用 fresh independent simulation。
- 评估仍使用 independent simulation paths。

### Phase 6：参数实验和 ablation

运行：

- Minimum viable parameter sets。
- Learning rate ablation。
- Policy visualization。
- Summary tables。

时间允许时扩展到 full parameter sensitivity matrix。

## 20. 风险控制与失败应对

如果训练不稳定，按以下顺序排查：

1. 只运行 `Reward (1), oracle, analytic policy` 和随机策略，确认环境与 evaluation 正确。
2. 运行 `Reward (1), oracle, PPO, baseline`，检查 policy curve 是否接近 analytic。
3. 降低学习率到 `1e-4`，减少 action bound 或增加 entropy regularization。
4. 对 TD3 增加 exploration noise 或扩大 behavior policy noise，检查 replay buffer 覆盖。
5. 暂时减少参数组，只保留 minimum viable matrix，先保证核心要求完整。

如果 `sequential data` 结果弱于 oracle，这是可接受结果；报告中应解释数据相关性、覆盖不足和无法重置带来的影响。

## 21. Implementation Checklist

### Notebook

- [ ] 实现 `MeanRevertingStockEnv`。
- [ ] 支持 `Reward (1)` 和 `Reward (2)`。
- [ ] 实现 action clipping。
- [ ] 实现 `Reward (2)` 的 `A_prev` 状态更新。
- [ ] 实现 `analytic_policy_reward1` 和 clipped analytic benchmark。
- [ ] 实现 simulation oracle sampler。
- [ ] 实现 sequential path generator。
- [ ] 实现 sequential setting 的 behavior policy。
- [ ] 实现 PPO actor、critic、GAE 和 update。
- [ ] 实现 TD3 actor、twin critics、target networks、replay buffer 和 update。
- [ ] 实现 independent Monte Carlo evaluation。
- [ ] 输出 `results_main.csv`。
- [ ] 输出 `results_ablation.csv`。
- [ ] 生成 `fig_reward1_policy_comparison.png`。
- [ ] 生成 `fig_reward2_policy_heatmap.png`。
- [ ] 生成 `fig_training_curves.png`。
- [ ] 生成 `fig_ablation_learning_rate.png`。
- [ ] 设置 random seeds。
- [ ] Notebook 从头到尾可以运行。

### Written Report

- [ ] 解释价格模型。
- [ ] 解释两个 reward。
- [ ] 解释 state/action 设计。
- [ ] 推导 `Reward (1)` 理论最优策略。
- [ ] 描述 simulation oracle 数据采样。
- [ ] 描述 sequential data 数据采样。
- [ ] 描述 off-policy behavior policy。
- [ ] 描述 PPO/TD3 的网络结构和更新步骤。
- [ ] 报告多组参数实验。
- [ ] 使用独立模拟路径做 Monte Carlo 评估。
- [ ] 报告 mean discounted reward 和 standard error。
- [ ] 可视化 learned policy。
- [ ] 对比 `Reward (1)` learned policy 与理论最优策略。
- [ ] 完成至少一个 ablation study。
- [ ] 总结 oracle vs sequential、PPO vs TD3 的差异。
