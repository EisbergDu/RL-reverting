# Reinforcement Learning for Mean-Reverting Stock Trading

Date: 2026-05-06  
Source files: `results_main.csv` and generated figures in this output directory

## Abstract

This project demonstrates how reinforcement learning can be applied to a simplified financial trading problem. The market is a simulated mean-reverting stock model. At each time step, the agent observes the market state, chooses a continuous position \(A_t\), and learns a policy that maps states to trading actions. The project compares three common continuous-control RL algorithms: PPO, TD3, and SAC.

The main finding is that, under the current experiment settings, the off-policy actor-critic methods TD3 and SAC generally achieve higher Monte Carlo mean discounted total reward than PPO. PPO is more conservative and has lower turnover, but its return is weaker. Reward (1) admits an analytic optimal policy, which provides a useful benchmark for checking whether the learned policy captures the correct trading structure. Reward (2) introduces a position-change penalty and is closer to a transaction-cost setup, but the resulting turnover behavior differs strongly across algorithms.

## 1. Background

Financial trading can be formulated naturally as a reinforcement learning problem. The agent observes market information, chooses a position, receives a trading reward, and aims to maximize long-run discounted reward.

This project uses simulated data rather than real stock data. The benefit is that the data-generating process is known, which makes it possible to control market parameters and compare learned policies with a theoretical benchmark.

## 2. Market Model

Let \(S_t\) be the stock price with initial value \(S_0=1\). The log-return process \(L_t\) follows a one-dimensional mean-reverting process:

```text
L_{t+1} = (1-kappa) L_t + sigma Z_t,  Z_t ~ N(0, 1)
S_{t+1} = S_t exp(L_{t+1})
```

The state representation is:

```text
Reward (1): state = (log S_t, L_t)
Reward (2): state = (log S_t, L_t, A_{t-1})
```

The action \(A_t\) is the number of shares held by the agent. Positive values correspond to long positions and negative values correspond to short positions. The implementation clips actions to a bounded interval for numerical stability.

## 3. Reward Design

The project compares two reward functions.

Reward (1) penalizes the absolute position size:

```text
R_t = A_t(S_{t+1} - S_t) - lambda A_t^2
```

This reward is mathematically simple because the action does not affect the future price process. As a result, Reward (1) has a closed-form analytic optimal policy, which is used as a benchmark.

Reward (2) penalizes position changes:

```text
R_t = A_t(S_{t+1} - S_t) - lambda(A_t - A_{t-1})^2 S_t
```

This reward is closer to a stylized transaction-cost model because it penalizes turnover. To preserve the Markov property, the state must include the previous action \(A_{t-1}\).

## 4. Algorithms

PPO is an on-policy method. It collects rollouts using the current policy and updates the actor and value function using a clipped objective. PPO is often stable, but it is typically less sample-efficient than off-policy methods.

TD3 is an off-policy deterministic actor-critic method. It uses a replay buffer, twin critics, target policy smoothing, and delayed policy updates to reduce Q-value overestimation. It is well suited to continuous action spaces.

SAC is an entropy-regularized stochastic actor-critic method. It maximizes both reward and policy entropy, which encourages exploration and makes it suitable for continuous-control trading tasks.

## 5. Experiment Setup

The results are taken from `outputs/results_main.csv`.

| Item | Setting |
| --- | --- |
| Parameter sets | baseline, slow_reversion, high_volatility, strong_regularization |
| Seeds | 0, 1 |
| Rewards | R1, R2 |
| Data settings | simulation oracle, sequential |
| Algorithms | PPO, TD3, SAC |
| Monte Carlo evaluation paths | 200 |
| Evaluation horizon | 80 |

The main performance metric is `mean_return`, defined as the Monte Carlo estimate of the mean discounted total reward. It is not a stock return:

```text
E[sum_t gamma^t R_t]
```

## 6. Figures

### Sanity Check

![Sanity trajectory](fig_sanity_trajectory.png)

This figure verifies that the simulated price path, mean-reverting return process, and random action trajectory are generated correctly.

### Training Curves

![Training curves](fig_training_curves.png)

The training curves show the learning behavior of PPO, TD3, and SAC under the baseline R1 oracle setting. They are useful for checking whether training runs normally, but they are not the final evaluation metric.

### Reward (1) Policy Comparison

![Reward 1 policy comparison](fig_reward1_policy_comparison.png)

Because Reward (1) has an analytic optimal policy, the learned policies can be plotted against the theoretical policy. A closer match suggests that the algorithm has learned the correct trading structure.

### Reward (1) Policy Heatmap

![Reward 1 policy heatmap](fig_reward1_policy_heatmap.png)

The heatmap visualizes the action chosen under different states. This is important because training reward in financial RL can be noisy, and policy visualization helps interpret the learned behavior.

### Reward (2) Policy Heatmap

![Reward 2 policy heatmap](fig_reward2_policy_heatmap.png)

For Reward (2), the policy depends on the previous position \(A_{t-1}\). If transaction costs are effective, the policy should avoid unnecessarily large changes in position.

## 7. Baseline Results

The table below averages the two seeds under the baseline parameter set.

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

Under the baseline R1 oracle setting, TD3 has the highest average reward and reaches about 68.71% of the analytic benchmark. SAC is close to TD3 at about 63.96%. PPO is much more conservative, with lower turnover and smaller positions, but it achieves a much lower return.

Under the baseline R2 oracle setting, SAC obtains the highest mean return, but its turnover is also much higher. This suggests that SAC is more aggressive in this setting. TD3 has slightly lower return but lower turnover. PPO remains conservative and produces a negative return.

## 8. Analytic Benchmark and Parameter Sets

Reward (1) provides an analytic benchmark. The table below reports the analytic policy performance under different parameter sets.

| Parameter set | Analytic mean return | Std. error | Turnover | Avg. abs. position |
| --- | --- | --- | --- | --- |
| baseline | 1895.18 | 1530.11 | 2.49 | 4.86 |
| slow_reversion | 5.224e+16 | 5.224e+16 | 1.43 | 5.32 |
| high_volatility | 8.111e+08 | 8.063e+08 | 2.68 | 5.56 |
| strong_regularization | 1880.77 | 1529.97 | 1.40 | 2.76 |

The slow_reversion and high_volatility settings produce very large reward values. This is mainly caused by the lognormal price update \(S_t=\exp(\log S_t)\), which can generate extreme price paths under slow mean reversion or high volatility. These settings are useful for showing parameter sensitivity, but the magnitudes should not be interpreted as realistic financial returns.

## 9. Algorithm Comparison Under R1 Oracle

The next table aggregates results under Reward (1) and the simulation oracle setting.

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

Overall, TD3 and SAC clearly outperform PPO in most parameter settings. PPO has lower turnover and sometimes lower policy MSE, but it does not exploit the trading signal as strongly as the off-policy methods.

## 10. Discussion

First, reinforcement learning provides a natural framework for learning trading policies. The state contains market information, the action is the position, and the reward combines trading profit with risk or transaction-cost penalties.

Second, reward design is central. R1 penalizes position size and has a clean analytic benchmark. R2 penalizes position changes and is closer to a transaction-cost setting, but its policy behavior is more complex.

Third, off-policy methods are more effective in this experiment. TD3 and SAC reuse data through replay buffers, which generally improves sample efficiency compared with PPO. PPO is more stable and conservative, but its return is lower.

Fourth, the results have substantial statistical uncertainty. Several settings have large standard errors, which means that Monte Carlo reward variance is high. Therefore, the results should be interpreted as an algorithmic demonstration rather than a definitive ranking.

## 11. Limitations

The market is simulated and omits many real trading frictions, including bid-ask spreads, market impact, capital constraints, risk limits, and nonstationarity.

Only two seeds are used, so the conclusions are still sensitive to randomness. A more rigorous study should use more seeds and more evaluation paths.

The slow_reversion and high_volatility parameter sets can produce extreme reward magnitudes. Future work could add price clipping, risk-adjusted metrics, or Sharpe ratio evaluation.

This report does not include a separate learning-rate ablation. The current result file already includes multiple market parameter settings, but hyperparameter ablation remains a useful extension.

## 12. Conclusion

This project demonstrates a complete minimal pipeline for applying reinforcement learning to a financial trading task: defining the market environment, specifying state/action/reward, implementing PPO, TD3, and SAC, and evaluating learned policies through independent Monte Carlo simulation.

The experiments show that TD3 and SAC generally obtain higher reward than PPO in this continuous-action mean-reverting trading task. PPO is more conservative but less profitable. The analytic benchmark under Reward (1) is especially useful for validating learned policies, while Reward (2) shows how transaction-cost-like penalties can significantly change policy behavior.

The main contribution of the project is not a deployable trading strategy, but a clear demonstration of how financial trading can be formalized as a reinforcement learning problem and how different RL algorithms behave under the same simulated environment.
