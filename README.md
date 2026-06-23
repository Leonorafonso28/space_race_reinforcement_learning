# Deep Q-Network & Exploration Strategies (SpaceRace)

## Overview

This notebook implements a Deep Reinforcement Learning agent for the **SpaceRace** environment, where a spaceship must dodge falling debris. The work is divided into three progressive tasks: a basic DQN, an enhanced DQN with Experience Replay and a Target Network, and a comparative analysis of exploration strategies.

## Environment

`SpaceRaceEnv` — RGB observation of 54×39×3 pixels, two actions (up/down), score based on how many times the ship crosses the top row. Supports difficulty levels 0 through 3.

---

## Notebook Structure

### Task 1 — Vanilla DQN (no Replay Buffer)

**1.1 — CNN Architecture (`DQN`):** Two convolutional blocks (16 and 32 filters) with MaxPooling, followed by two fully-connected layers (3744→128→2). Input normalisation (`/255.0`) is applied inside `forward`. Default PyTorch Kaiming Uniform initialisation.

**1.2 — Loss Function:** Bellman equation with `torch.no_grad()` on the target and terminal-state masking via `(1 - done)`. Discount factor γ = 0.9.

**1.3 — Training Loop:** One gradient update per transition. Epsilon-greedy with per-episode decay (1.0→0.05, rate 0.995), gradient clipping (max norm 1.0), Adam (lr=1e-4). Trained separately on difficulty 0 (700 eps) and difficulty 1 (650 eps). A mixed 50/50 curriculum using D0 weights as a warm-start was then applied (500 additional eps, lr=1e-5).

**1.4 — Heuristic Warm-Start:** A rule-based policy using `semantic_obs` checks the three cells directly above the ship; if any contain debris, the agent moves down. Used to pre-train the CNN (50 episodes → 5 imitation epochs), then RL fine-tuning with ε_start=0.3. Roughly halves the number of episodes needed to reach the same final performance.

**Task 1 Results:**

| Strategy | D0 | D1 |
|---|---|---|
| Trained on D0 only | 29 | 24 |
| Trained on D1 only | 7 | 32 |
| Curriculum D0 → D1 | 21 | 31 |
| Mixed curriculum (50/50) | 29 | 29 |

---

### Task 2 — Enhanced DQN with Experience Replay & Target Network

**2.1 — Replay Buffer:** `ReplayBuffer` (FIFO deque, capacity=10,000) and `PrioritizedReplayBuffer` (PER with α=0.6, β annealed to 1.0). States stored as `uint8` for memory efficiency. Vectorised batch loss (`compute_loss_batch`).

**2.2 — Training Integration:** `compute_loss_dqn` uses a separate frozen target network for stable Bellman targets. Buffer is pre-filled with heuristic and random transitions before training begins. Optional PER support.

**2.3 — Target Network Comparison:** Two update modes implemented — **hard update** (full copy every C steps) and **soft update** (Polyak averaging: `τ=0.01` every training step). Mean Q-value tracked per episode. Both modes trained on D0, D1, and D2.

---

### Task 3 — Exploration Strategies

**3.1 — ε-Greedy:** Three decay schedules — exponential (`ε×0.995` per episode), linear (interpolation over 400 episodes), and step-based (factor 0.3 at milestones 100/250/400). Tested with ε_start=1.0 and ε_start=0.5.

**3.2 — Boltzmann (Softmax):** Actions sampled proportionally to `exp(Q/τ)` with a numerically stable softmax (max subtraction). Same three schedules applied to temperature τ. Policy entropy tracked per step.

**3.3 — Comparative Analysis:** 8 runs on D1 (500 episodes each). Metrics reported: episodes to reach score 15, final score (mean of last 50 eps), stability (std), mean Q-value, and training time. Five plots: learning curves, schedule trajectories, exploration behaviour, Q-value evolution, and hyperparameter sensitivity.

**Key findings:**
- ε-greedy converges roughly 2× faster than Boltzmann — with only 2 actions, the softmax distribution stays near-uniform for much longer
- Boltzmann produces smoother, more stable learning curves (lower std)
- Both strategies reach similar final performance (~29)
- ε_start=0.5 is the best single configuration (`eps_exp_low_start`: score 29.08, reaches target at episode 97)

**Difficulty-3 Curriculum:** Best D1 model → fine-tune on D2 (300 eps) → fine-tune on D3 (500 eps), using exponential ε-greedy with progressively lower starting ε. Outcome: ~25 on D3 vs ~11 without curriculum.

## Dependencies

```
torch, numpy, matplotlib, space_race_env, google.colab
```