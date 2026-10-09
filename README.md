# Adventurer

Exploration with BiGAN for Deep Reinforcement Learning

This repository contains a from-scratch PyTorch implementation of the
Adventurer approach: PPO augmented with BiGAN-based novelty estimation and
intrinsic rewards.

The implementation is designed as a modular research foundation rather than a
minimal example. PPO, GAE, rollout storage, BiGAN components, novelty
estimation, reward shaping, reproducibility utilities, logging, and the
training pipeline are implemented directly in Python and PyTorch.

> **Research note:** Reproducing published benchmark numbers requires matching
> the paper's exact environment versions, preprocessing, network details,
> training budget, evaluation protocol, and hardware/software stack. This
> repository records configurations and separates algorithmic components so
> those details can be audited and modified explicitly.

## Master's Thesis Experiment

> This project investigates the effectiveness of action-conditioned transition
> novelty within a BiGAN-based exploration framework. Transition novelty itself
> is an established idea in the reinforcement-learning exploration literature;
> **this thesis does not claim to invent transition novelty**. The contribution
> is the controlled empirical study of transition novelty inside the
> BiGAN/Adventurer representation and reward pipeline described below.

### Research question

> Does transition-based intrinsic reward improve exploration compared with the
> original BiGAN-based state novelty used in Adventurer?

Two experimental conditions are compared. They differ **only** in the intrinsic
exploration signal; the environment, preprocessing, PPO architecture and
hyperparameters, rollout geometry, training budget, and seeds are identical,
and no hyperparameter is tuned per method.

|  | Experiment A -- baseline | Experiment B -- proposed |
|---|---|---|
| command | `python main.py --method state ...` | `python main.py --method transition ...` |
| intrinsic signal | Adventurer BiGAN state novelty `B(s)` | BiGAN action-conditioned transition novelty `N_T` |
| formula | `B(s) = alpha*L_G(s) + (1-alpha)*L_D(s)` | `N_T = \|\| f(E(s_t), a_t) - E(s_{t+1}) \|\|_2` |
| extra networks | none (BiGAN as published) | one latent forward MLP `f` |

The proposed method is, exactly:

```text
Observation -> BiGAN encoder E -> z_t
(z_t, one-hot a_t) -> forward model f -> predicted z_{t+1}
N_T = || predicted z_{t+1} - E(s_{t+1}) ||_2        (L2, not squared, not L1)
intrinsic reward = Equation (5) normalization of N_T  (same mechanism as baseline)
-> two-stream GAE / PPO (unchanged)
```

with `z_t = E(s_t)`, `z_{t+1} = E(s_{t+1})`, `hat_z_{t+1} = f(z_t, a_t)` and

```text
L_f = (1/B) * sum_i || f(E(s_t), a_t) - E(s_{t+1}) ||_2^2
```

Forward-model settings (identical to `NoveltyConfig` defaults): MLP
`(z + one-hot a) -> 256 -> 256 -> z`, Adam `lr = 1e-4`, `B = 64`, gradient-norm
clip `0.5`, one update per PPO update. The BiGAN encoder is **excluded** from
this optimizer -- it keeps being trained by the BiGAN adversarial objective
alone -- and no generator or discriminator term, no EME/ensemble scaling, no
adaptive scaling, and no visit or episodic counts contribute to `N_T`.

Latent targets are cached at collection time as `(z_t, a_t, z_{t+1})` and are
never re-encoded later. This is deliberate: a raw `(s_t, a_t, s_{t+1})` replay
of one rollout costs roughly 10 GB at the 96x128 Montezuma configuration, while
the latent buffer costs a few megabytes. The consequence is documented: the
regression target is the encoder representation *available when the transition
was collected*, i.e. the encoder state before that update's BiGAN step.

### Fairness between the two conditions

`scripts/smoke_master.sh` automatically asserts that both Master's runs record
identical `ppo`, `bigan`, `environment`, `seed`, and training-budget settings in
`config.json`, that the shared Equation (5) normalization settings match, that
the legacy EME/metric bonus is disabled in both, and that the forward-model
settings are the documented ones. Only `novelty.novelty_type` and
`novelty.transition_variant` are allowed to differ.

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-master.txt   # verified versions + notes
```

`requirements-master.txt` documents the verified stack (Python 3.11.2, torch
2.14.0, gymnasium 1.3.0, numpy 2.4.6, ale-py 0.12.1) and explains the
platform-dependent CUDA/CPU wheel situation. Atari environments need `ale-py`
(for `ALE/MontezumaRevenge-v5`); the Noisy-TV maze and the smoke test do not.
Running the original Unity build additionally needs the upstream `unityagents`
client, `Pillow`, and the player executable - see the environment section
below.

### The Noisy-TV maze (default environment)

The default environment is the **Noisy-TV maze** of *Large-Scale Study of
Curiosity-Driven Learning* (ICLR 2019), whose reference implementation is the
Unity project [`luchris429/noisy-tv-env`](https://github.com/luchris429/noisy-tv-env)
("The Noisy TV Environment from Large-Scale Study of Curiosity-Driven
Learning"). In that task an agent navigates a maze of rooms and corridors that
contains a television whose content keeps changing at random; the television is
unpredictable *and* irrelevant to the task, so the setting is the canonical
demonstration that prediction-error novelty can be captured by a stochastic
distractor.

Two interchangeable implementations of that maze are provided, both exposing
the upstream interface (84x84 RGB first-person frames, six discrete actions, a
sparse `+1` reward within 2.5 units of the goal sphere, and the
`startLoc`/`door`/`tv` episode conditions):

| ID | implementation | requirements |
|---|---|---|
| `NoisyTVMaze-v0` | `environments/noisy_tv_maze.py`, pure-python reconstruction | `gymnasium` only |
| `NoisyTVUnity-v0` | `environments/noisy_tv_unity.py`, Gymnasium adapter around the **original Unity build** and the `unityagents` client vendored in the upstream repository | the upstream player executable (download link in that repository's README), `unityagents`, and `Pillow` |

The reconstruction is the default because it needs nothing beyond the pinned
stack. It is not a sketch: the 138 wall cubes, 17 rooms, 18 hallways, 16 start
poses, the television plane, the goal sphere, the sliding door, and the button
semantics (one shared modulo-ten counter, channel changes only within 18 units)
were all read out of the upstream Unity scene and agent scripts, and the module
documents its four deliberate differences (discrete time, 15-degree rotations,
ray-cast rendering, and synthetic channel images replacing the bundled
photographs). Observations are RGB because the upstream camera declares
`blackAndWhite: 0`, so the convolutional PPO/BiGAN path and both the pixel and
the feature novelty terms are exercised exactly as on Atari.

The maze is sealed and fully connected, and the reward sits behind the sliding
door: with the upstream default `door=1` (closed) the goal is unreachable from
most start poses until the agent presses action `4` often enough to open it.
Together with the stochastic television, this is exactly the setting the thesis
instruments:

* `tv="noisy"` (default) redraws the screen every step, so a stationary agent
  in front of the television receives a permanently changing observation that
  no predictor can model; `tv="static"` is the deterministic control;
* watching the television pays nothing - the distractor is purely
  observational, which the test suite asserts.

```python
import gymnasium as gym
from environments.noisy_tv_unity import register_environment, make_noisy_tv_environment

register_environment()                          # registers both IDs
environment = gym.make("NoisyTVMaze-v0", tv="noisy", door=1.0)
observation, info = environment.reset(seed=0)   # info: distance_to_goal, distance_to_television, ...

# "auto" uses the original Unity build when its executable and client are
# installed and falls back to the reconstruction otherwise; "unity" and
# "python" force one of the two.
environment_or_unity = make_noisy_tv_environment("auto", image_size=84)
print(environment_or_unity.backend)             # "unity" or "python"
```

The Unity adapter never falls back silently: `backend="unity"` raises with
instructions rather than quietly switching environment in the middle of a
comparison. The build is found in the locations the installer script uses, or
through `NOISY_TV_UNITY_BINARY=/path/to/tv_maze`; when a training run builds
several environments in one process (`--num-parallel-envs`), each player gets
its own worker id (port), so one process can drive many players at once.

`tests/test_noisy_tv_unity_integration.py` covers the Unity path **without the
Unity binary**, because the binary is distributed out of band and is often
unavailable (no Google Drive access, no GPU, no display). It starts the
genuine `unityagents` client, which binds a socket, launches
`tests/fixtures/fake_tv_maze_player.py` as `tv_maze.x86_64`, and decodes the
frames it sends back; only the Unity engine itself is simulated. That suite
drives the real transport end to end: frames, rewards and termination, the
`startLoc`/`door`/`tv` reset parameters, the noisy-versus-static television
phenomenon, graceful player shutdown, `gymnasium.make("NoisyTVUnity-v0")`, and
two parallel players on distinct ports.

```bash
# default: NoisyTVMaze-v0, 96 envs, 12.288M steps, seeds 0 1 2
bash scripts/run_master_experiments.sh

# a quick CPU-friendly scale
DEVICE=cpu NUM_ENVS=16 TOTAL_STEPS=1024000 SEEDS="0 1 2" \
    bash scripts/run_master_experiments.sh

# the original Unity build (needs the upstream player executable + client);
# NOISY_TV_UNITY_BINARY points at the build when it is not in a default location
NOISY_TV_UNITY_BINARY=/opt/tv_maze/tv_maze \
    ENV_NAME=NoisyTVUnity-v0 bash scripts/run_master_experiments.sh

# the Atari benchmark used elsewhere in the thesis
ENV_NAME=ALE/MontezumaRevenge-v5 DEVICE=cuda bash scripts/run_master_experiments.sh
```

`DEVICE` defaults to `auto`, which selects CUDA when it is available and CPU
otherwise, so the default command runs on a laptop as well as on a GPU node.
`NUM_ENVS`, `ROLLOUT_STEPS`, `TOTAL_STEPS`, and `BIGAN_BATCH_SIZE` keep their
thesis values; lower them (as in the 1.024M-step example above) for a quick run.

### Smoke test (tiny, CPU, a few minutes at most)

```bash
bash scripts/smoke_master.sh
```

It runs both Master's methods and the legacy transition path on a tiny
CartPole configuration (64 environment steps each) and verifies: processes
start, the environment works, PPO updates complete, the intrinsic reward is
finite, the forward model updates (`transition/updated`, finite MSE), the
output files exist (`config.json`, `run_info.json`, `metrics.jsonl`,
`run.log`), the two methods share one configuration, the aggregation and
figure scripts run, and the whole unit test suite passes. Montezuma is
deliberately not used; override with `SMOKE_ENV=ALE/MontezumaRevenge-v5
SMOKE_TOTAL_STEPS=256` to exercise the real environment briefly, or run the
smoke test on the Noisy-TV maze with
`SMOKE_ENV=NoisyTVMaze-v0 SMOKE_TOTAL_STEPS=256`.

### Full experiment

The command lines, the `DEVICE=auto` default, and the scale overrides are given
in the environment section above.

The driver runs `state` seeds 0,1,2 then `transition` seeds 0,1,2, writes
`training.log` per run, prints the git commit before training, verifies
dependencies and schedule consistency, refuses to write into an existing run
directory (`OVERWRITE=1` overrides), stops immediately on the first failed run
(no silent retries), and finally aggregates and plots. Individual runs can
also be launched by hand:

```bash
# the default environment (the Noisy-TV maze) on CPU
python main.py --method state      --env NoisyTVMaze-v0 --seed 0 \
    --num-parallel-envs 8 --rollout-steps 128 --minibatch-size 32 \
    --total-environment-steps 1024000 --device cpu \
    --output-directory results/master/state/seed_0
python main.py --method transition --env NoisyTVMaze-v0 --seed 0 \
    --num-parallel-envs 8 --rollout-steps 128 --minibatch-size 32 \
    --total-environment-steps 1024000 --device cpu \
    --output-directory results/master/transition/seed_0

# the thesis benchmark
python main.py --method state      --env ALE/MontezumaRevenge-v5 --seed 0 \
    --num-parallel-envs 96 --rollout-steps 128 --minibatch-size 32 \
    --total-environment-steps 12288000 --device cuda \
    --output-directory results/master/state/seed_0
python main.py --method transition --env ALE/MontezumaRevenge-v5 --seed 0 \
    --num-parallel-envs 96 --rollout-steps 128 --minibatch-size 32 \
    --total-environment-steps 12288000 --device cuda \
    --output-directory results/master/transition/seed_0
```

### Output structure

```text
results/
└── master/
    ├── state/
    │   ├── seed_0/
    │   │   ├── config.json        # experimental configuration
    │   │   ├── run_info.json      # execution/reproducibility metadata
    │   │   ├── metrics.csv        # update-level metric table (aggregation)
    │   │   ├── metrics.jsonl      # per-update + per-episode diagnostics
    │   │   ├── run.log            # text log
    │   │   ├── training.log       # captured stdout/stderr of the run
    │   │   ├── plt/  tensorboard/
    │   └── seed_1/ ...
    ├── transition/
    │   └── seed_0/ ...            # same layout
    ├── master_summary.csv         # method, seed, final score, returns, ...
    ├── master_summary_stats.csv   # mean / std / median per method
    └── figures/                   # the five thesis figures
```

### Reproducibility

Every run writes two complementary files: `config.json` records the complete
experimental configuration (frozen dataclasses), while `run_info.json` records
execution metadata -- git commit, dirty flag, branch, UTC timestamp, Python /
PyTorch / Gymnasium / ALE / NumPy versions, platform, resolved device,
environment id, seed, environment count, rollout steps, total environment
steps, the selected method, and a `key_settings` block with the PPO/BiGAN/
novelty values a thesis appendix needs. `run_info.json` never replaces
`config.json`. Seeds 0, 1, 2 are the default experimental set (add 3, 4 when
compute allows); individual seed results are always reported, and with three
seeds the summary reports mean/std/median without claiming significance.

### Metrics and figures

Primary metric: true extrinsic game score. Secondary: cumulative extrinsic
reward, final game score, first positive reward step, episode length/count,
intrinsic reward, and learning curves.

```bash
python scripts/summarize_master_results.py --results-dir results/master
python scripts/plot_master_results.py      --results-dir results/master
```

`master_summary.csv` contains one row per (method, seed) with
`method, seed, final_score, total_extrinsic_return, first_reward_step,
episode_count`; `first_reward_step = -1` means the run never obtained a
positive extrinsic reward, which is a valid and informative outcome on a
sparse-reward game. The five figures are: game score vs. environment steps,
mean +/- standard deviation across seeds, individual seed curves, first
positive reward time per run, and intrinsic reward over training.

### Command-line selectors

Three concepts coexist and are deliberately kept distinct:

| selector | meaning |
|---|---|
| `--method state` | Master's baseline: Adventurer BiGAN state novelty |
| `--method transition` | Master's proposed method: `N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2` |
| `--novelty transition` | **legacy** transition novelty (L1 latent error + discriminator feature term), preserved unchanged for backward compatibility |

`--method` is mutually exclusive with `--novelty`, `--novelty-type`, and with
the legacy bonus flags (`--eme`, `--metric-learning`, `--episodic-count-scaling`):
ambiguous or unfair combinations exit with an explicit error instead of
silently launching a wrong comparison. `--transition-batch-size` (default 64)
exposes the forward-model minibatch; only tiny smoke configurations need a
smaller value.

### Not part of the Master's experiment

The EME/metric exploration bonus (`exploration/state_discrepancy.py`,
`exploration/ensemble_scaling.py`, `exploration/episodic_count.py`,
`exploration/eme_metric.py`, `MetricIntrinsicReward`), the resettable episodic
memory, the legacy transition novelty (`exploration/transition_novelty.py`),
and `run_metric_eme_comparison.sh` are previous research work. They remain on
disk, default-disabled, and covered by `tests/test_metric_eme.py`, but the
Master's experiment neither imports their logic at run time nor executes it:
`metric_eme.enabled` is `False` and `resettable` is `False` in every Master's
run, enforced by the CLI validation above. See
*Legacy / Previous EME Experiments* further below.

## Features

- Python 3.9-compatible codebase
- PyTorch 2.x implementation
- PPO implemented from scratch
- Clipped policy objective
- Entropy bonus
- Critic value loss
- Optional value clipping
- GAE and bootstrapped returns
- Gradient-norm clipping
- Minibatch PPO updates
- Linear learning-rate scheduling
- BiGAN encoder, generator, and joint discriminator
- Adversarial BiGAN optimization from a replay buffer
- Pixel reconstruction novelty
- Joint-feature matching novelty
- Running novelty normalization
- Intrinsic reward scaling and clipping
- Explicit extrinsic/intrinsic reward combination
- Gymnasium-compatible environment adapter
- Deterministic seeding utilities
- JSONL and text experiment logging
- Checkpoint-friendly replay and normalization state
- Unit tests based on analytically calculated expected values

## Algorithm Overview

For an observation \(x_t\), the BiGAN encoder produces a latent code:

\[
z_t = E_\psi(x_t).
\]

The generator reconstructs the observation:

\[
\hat{x}_t = G_\theta(E_\psi(x_t)).
\]

Novelty is computed from both pixel and learned feature discrepancies:

\[
e_{pixel}(x_t)
= \frac{1}{d_x}\|x_t - \hat{x}_t\|_2^2,
\]

\[
e_{feature}(x_t)
= \frac{1}{d_h}
\left\|
 h_\omega(x_t,E_\psi(x_t))
 - h_\omega(\hat{x}_t,E_\psi(x_t))
\right\|_2^2,
\]

\[
B(x_t) =
\alpha L_G(x_t) + (1-\alpha)L_D(x_t),
\qquad \alpha\in[0,1].
\]

The paper evaluates a sweep over
\(\alpha\in\{0.5,0.7,0.9,1.0\}\), with \(\alpha=0.9\) as the default
reported setting. The value is configured through `NoveltyConfig.alpha`; the
pixel and feature terms are therefore a convex combination, not an
independent weighted sum.

The normalized novelty score becomes intrinsic reward:

\[
r_t^{int} = \eta\,\widetilde n(x_t).
\]

The reward used by PPO is:

\[
r_t = r_t^{ext} + \rho r_t^{int}.
\]

GAE is then computed from this combined reward:

\[
\delta_t = r_t + \gamma(1-m_t)V_\phi(s_{t+1}) - V_\phi(s_t),
\]

\[
\hat A_t = \delta_t +
\gamma\lambda(1-m_t)\hat A_{t+1},
\]

\[
\hat R_t = V_\phi(s_t) + \hat A_t.
\]

PPO optimizes the clipped policy objective:

\[
r_t(\vartheta) =
\exp\left(
\log\pi_\vartheta(a_t|s_t)
-
\log\pi_{old}(a_t|s_t)
\right),
\]

\[
L^{CLIP}(\vartheta) =
\mathbb E_t\left[
\min\left(
 r_t(\vartheta)\hat A_t,
 \operatorname{clip}(r_t(\vartheta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

The implemented minimization objective is:

\[
L = -L^{CLIP} + c_vL^{VF} - c_eL^{ENT}.
\]

## Data Flow

The environment/PPO/BiGAN integration is explicit:

```text
reset environment
    |
    v
current observation s_t
    |
    +--> BiGAN encoder E(s_t)
    |        |
    |        +--> generator G(E(s_t))
    |        +--> pixel novelty
    |        +--> feature novelty
    |        +--> normalized novelty
    |                 |
    |                 v
    |          intrinsic reward
    |
    +--> actor pi(a_t | s_t)
    +--> critic V(s_t)
             |
             v
       environment.step(a_t)
             |
             +--> extrinsic reward
             +--> terminated/truncated flags
             +--> next observation
                         |
                         v
       combine rewards and store transition
                         |
                         +--> PPO rollout buffer
                         +--> BiGAN replay buffer

rollout complete
    |
    +--> GAE and returns
    +--> PPO minibatch updates
    +--> BiGAN replay updates
    +--> metrics logging
```

## Repository Structure

```text
Adventurer/
├── README.md
├── config.py
├── main.py
├── trainer.py
├── agent/
│   ├── actor.py
│   ├── critic.py
│   ├── ppo.py
│   ├── rollout_buffer.py
│   └── advantage.py
├── bigan/
│   ├── encoder.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── losses.py
│   └── trainer.py
├── exploration/
│   ├── novelty.py              # Adventurer state novelty B(s)  [Master's baseline]
│   ├── master_transition.py    # Master's N_T = ||f(E(s),a)-E(s')||_2  [Master's method]
│   ├── transition_novelty.py   # LEGACY transition novelty (kept, unchanged)
│   ├── intrinsic_reward.py     # Eq. (5) pipeline + LEGACY MetricIntrinsicReward
│   ├── state_discrepancy.py    # LEGACY latent metric
│   ├── ensemble_scaling.py     # LEGACY EME reward ensemble
│   ├── episodic_count.py       # LEGACY visit-count habituation
│   ├── eme_metric.py           # LEGACY learned EME metric
│   └── episodic_memory.py      # LEGACY resettable episodic memory
├── utils/
│   ├── logger.py
│   ├── normalization.py
│   ├── replay.py
│   ├── transition_replay.py
│   ├── run_metadata.py         # run_info.json execution metadata
│   └── seed.py
├── scripts/
│   ├── run_master_experiments.sh   # Master's experiment driver
│   ├── smoke_master.sh             # tiny CPU smoke test
│   ├── smoke_unity_path.sh         # NoisyTVUnity-v0 smoke run, no Unity build needed
│   ├── install_noisy_tv_unity.sh   # clone the client, fetch the Unity player
│   ├── summarize_master_results.py # master_summary.csv + per-seed metrics.csv
│   └── plot_master_results.py      # the five thesis figures
├── environments/
│   ├── adapter.py
│   ├── noisy_tv_maze.py        # Noisy-TV maze of the ICLR-2019 study [default ENV_NAME]
│   ├── noisy_tv_unity.py       # adapter for the original Unity build (NoisyTVUnity-v0)
│   └── vector_adapter.py
├── evaluation/
│   └── comparison.py
├── requirements-master.txt
└── tests/
    ├── fixtures/
    │   └── fake_tv_maze_player.py    # player half of the upstream protocol (tests only)
    ├── test_master_transition.py     # Master's method tests + smoke runs
    ├── test_noisy_tv_maze.py         # Noisy-TV maze + Unity adapter tests
    ├── test_noisy_tv_unity_integration.py  # the real client + protocol, no Unity binary
    └── test_metric_eme.py            # LEGACY EME/metric tests (kept, passing)
```

## Installation

Create a Python 3.9 virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy gymnasium
```

For Atari environments, install the required Gymnasium extras separately. The
exact package set depends on the selected environment and platform. For
example:

```bash
python -m pip install "gymnasium[atari,accept-rom-license]"
```

Verify the Python version:

```bash
python --version
```

The project targets Python 3.9 and a PyTorch 2.x release that supports Python 3.9.
The Master's experiment was additionally verified on Python 3.11.2 with the
stack recorded in `requirements-master.txt`.

## Running an Experiment

> **Master's thesis users:** start with the *Master's Thesis Experiment*
> section above (`--method state` / `--method transition`,
> `scripts/smoke_master.sh`, `scripts/run_master_experiments.sh`). The generic
> commands below document the underlying pipeline and its legacy selectors.

The default command uses `CartPole-v1`, which is useful for validating the
pipeline interface:

```bash
python main.py \
  --environment-id CartPole-v1 \
  --seed 0 \
  --total-environment-steps 10000 \
  --output-directory runs/cartpole
```

A longer image-based run can be launched with an Atari environment:

```bash
python main.py \
  --environment-id ALE/Breakout-v5 \
  --seed 0 \
  --total-environment-steps 10000000 \
  --rollout-steps 128 \
  --minibatch-size 32 \
  --device auto \
  --output-directory runs/breakout
```

Disable intrinsic rewards for an extrinsic-only PPO ablation:

```bash
python main.py \
  --environment-id CartPole-v1 \
  --disable-intrinsic-reward \
  --output-directory runs/cartpole_extrinsic_only
```

Available command-line options can be inspected with:

```bash
python main.py --help
```

## Configuration

All defaults are defined in `config.py` using frozen dataclasses:

- `SeedConfig`: global and deterministic execution settings.
- `EnvironmentConfig`: observation shape, action dimensions, and preprocessing.
- `PPOConfig`: discounting, GAE, clipping, optimization, and scheduling.
- `BiGANConfig`: latent/feature dimensions and adversarial update schedule.
- `NoveltyConfig`: Eq. (4) convex-mixture parameter `alpha` and intrinsic-reward scale. The default is `alpha=0.9`.
- `TrainingConfig`: total steps, device, logging, and reward mixing.
- `ExperimentConfig`: complete cross-component validation.

The command-line entry point discovers the actual environment observation and
action-space dimensions and combines them with the configured algorithmic
hyperparameters.

## PPO Implementation

`agent/actor.py` implements:

- categorical policies for discrete actions,
- diagonal Gaussian policies for continuous actions,
- action sampling,
- deterministic actions,
- log-probabilities,
- entropy.

`agent/critic.py` implements the state-value function.

`agent/rollout_buffer.py` stores the on-policy trajectory:

```text
observation
next observation
action
reward
terminated
truncated
value estimate
old log-probability
extrinsic reward
intrinsic reward
```

`agent/advantage.py` implements the backward GAE recurrence.

`agent/ppo.py` performs:

- old/new log-probability comparison,
- likelihood-ratio clipping,
- policy loss,
- value loss,
- entropy regularization,
- gradient clipping,
- minibatch epochs,
- learning-rate scheduling.

No external PPO implementation is used.

## BiGAN Implementation

The BiGAN modules implement the joint adversarial game:

- `bigan/encoder.py`: \(E_\psi(x)\).
- `bigan/generator.py`: \(G_\theta(z)\).
- `bigan/discriminator.py`: \(D_\omega(x,z)\).
- `bigan/losses.py`: BCE adversarial losses, reconstruction error, feature
  matching, and optional gradient penalty.
- `bigan/trainer.py`: replay sampling and alternating optimization schedule.

The discriminator receives two types of joint pairs:

```text
real pair:      (x, E(x))
generated pair: (G(z), z)
```

BiGAN observations are stored in `utils/replay.py`, independently from PPO's
on-policy rollout buffer.

## Novelty and Intrinsic Rewards

`exploration/novelty.py` provides independently testable components:

- `PixelReconstructionError`
- `FeatureMatchingError`
- `CombinedNoveltyScore`
- `NormalizedNoveltyScore`
- `NoveltyEstimator`

`exploration/intrinsic_reward.py` provides:

- intrinsic reward scaling,
- optional clipping,
- intrinsic reward enable/disable behavior,
- extrinsic/intrinsic reward combination.

Novelty inference runs without gradients. BiGAN parameters are updated only in
the BiGAN trainer.

## Reproducibility

Use a fixed seed:

```bash
python main.py --environment-id CartPole-v1 --seed 0
```

The seed utility configures:

- Python `random`,
- NumPy,
- PyTorch CPU,
- PyTorch CUDA,
- cuDNN deterministic mode,
- Gymnasium reset/action-space seeds,
- worker-local random streams.

Each run writes its configuration to:

```text
runs/<run-name>/config.json
```

Metrics are written to:

```text
runs/<run-name>/metrics.jsonl
runs/<run-name>/run.log
```

Determinism is hardware, driver, CUDA, PyTorch, and environment dependent.
The recorded seed should therefore be treated as necessary but not always
sufficient for bitwise reproducibility across machines.

## Testing

Run the complete suite (Master's transition tests plus the legacy EME/metric
tests) with either runner:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
```

`tests/test_noisy_tv_maze.py` covers the Noisy-TV environment: the upstream
geometry (138 walls, 17 rooms, 18 hallways, 16 start poses), wall collision,
seeded determinism, the sparse reward and terminate-vs-truncate behaviour, the
sliding door (modulo-ten button counter in both deterministic and randomized
modes, the closed door sealing the goal wing, and the agent passing through
once the door is open), the television (channels change only within 18 units,
and under `tv="noisy"` a stationary agent keeps receiving different
observations while the reward stays zero), state snapshots for `--resettable`,
adapter and `build_config` integration for both Master's methods, and the Unity
adapter's protocol translation against a stub client.

Where the build cannot be downloaded (no Google Drive access, no GPU, no
display), the python half of the path can still be exercised end to end with the
protocol stand-in of the test suite:

```bash
# NoisyTVUnity-v0 smoke run without the Unity build: the real unityagents
# client, socket, launch, and frame decoding, with simulated Unity frames
bash scripts/smoke_unity_path.sh

# the real build, once it is installed
NOISY_TV_UNITY_BINARY=/opt/tv_maze/tv_maze \
    ENV_NAME=NoisyTVUnity-v0 bash scripts/run_master_experiments.sh
```

`tests/test_noisy_tv_unity_integration.py` covers the original Unity path
end to end while simulating only the Unity engine: the real `unityagents`
client, the real socket protocol, the real `tv_maze.x86_64` launch, and real
frame decoding, driven against `tests/fixtures/fake_tv_maze_player.py`. That
includes the noisy-versus-static television phenomenon, the upstream reset
parameters, graceful shutdown of the player process, discovery of the build
(including `NOISY_TV_UNITY_BINARY`), construction of
`gymnasium.make("NoisyTVUnity-v0")`, and several parallel players on distinct
worker ids. The suite skips itself when the client or `Pillow` is missing.

`tests/test_master_transition.py` covers the Master's method: the analytic L2
prediction error, the MSE forward-model objective, one-hot action encoding,
encoder exclusion from the forward-model optimizer (bitwise parameter
equality), latent buffer shapes and circular behaviour, finite intrinsic
rewards under sparse/degenerate inputs, the `--method` selection rules, and
tiny end-to-end runs of the baseline, the proposed method, and the legacy
transition path. `tests/test_metric_eme.py` keeps the legacy EME/metric
mathematics covered; its expected values are computed analytically or by an
independent reference computation.

The advantage tests analytically verify:

- nonterminal bootstrap,
- terminal masking,
- truncation bootstrap,
- multistep GAE,
- return targets,
- normalization statistics.

Additional tests should be added before publication-scale experiments for:

- actor distribution shapes,
- critic outputs,
- rollout storage and minibatches,
- PPO clipping and gradient updates,
- BiGAN tensor shapes and adversarial updates,
- novelty components,
- environment API conversion,
- end-to-end short runs.

## Engineering Conventions

- Mathematical quantities are represented explicitly in code.
- Every module has a module-level docstring.
- Public classes and functions use type hints and docstrings.
- Configuration is immutable after construction.
- PPO and BiGAN sample stores are separated.
- Environment termination and truncation are represented separately.
- Generator/discriminator logits are passed to stable BCE-with-logits losses.
- Novelty statistics are detached from neural-network computation graphs.
- Logging is dependency-light and machine-readable.

## Legacy Transition-Level Novelty Variant (`--novelty transition`)

> **Legacy / previous work -- not part of the Master's thesis experiment.**
> The Master's proposed method is `--method transition` (see *Master's Thesis
> Experiment* above). The variant documented here is the historical
> implementation with an L1 latent error, a discriminator feature-matching
> term, an L1 forward-model loss, and a raw `(s, a, s')` replay buffer; it is
> preserved unchanged for backward compatibility and is exercised by the test
> suite and by `scripts/smoke_master.sh`.

The project supports two intrinsic-reward strategies:

```text
novelty_type=state
novelty_type=transition
```

The default `state` strategy preserves the original Adventurer state novelty:

\[
B(s)=\alpha L_G(s)+(1-\alpha)L_D(s).
\]

The transition strategy reuses the BiGAN encoder, generator, and discriminator
and adds a latent forward model:

\[
z_t=E(s_t),\qquad z_{t+1}=E(s_{t+1}),
\]

\[
\hat z_{t+1}=f_\phi(z_t,a_t).
\]

Its transition novelty is:

\[
T(s_t,a_t,s_{t+1})=
\alpha_t\left\|z_{t+1}-\hat z_{t+1}\right\|_1
+(1-\alpha_t)\left\|
 f_D(s_{t+1},z_{t+1})-
 f_D(G(\hat z_{t+1}),\hat z_{t+1})
\right\|_1.
\]

Transition samples are stored independently as:

```text
(s_t, a_t, s_(t+1))
```

The forward model is trained only on latent prediction loss. The feature error
is used as part of the transition novelty score and is logged separately.
The original BiGAN adversarial objective is unchanged.

### Train the state-novelty baseline

```bash
python main.py \\
  --environment-id ALE/Solaris-v5 \\
  --novelty-type state \\
  --seed 0 \\
  --num-parallel-envs 1 \\
  --total-environment-steps 1280 \\
  --output-directory runs/solaris-state
```

### Train the transition-novelty variant

```bash
python main.py \\
  --environment-id ALE/Solaris-v5 \\
  --novelty-type transition \\
  --transition-alpha 0.9 \\
  --seed 0 \\
  --num-parallel-envs 1 \\
  --total-environment-steps 1280 \\
  --output-directory runs/solaris-transition
```

Use identical seeds, environment settings, PPO settings, rollout length, and
training budgets when comparing methods. Only `novelty_type` should differ.

### Transition TensorBoard metrics

Transition runs additionally log:

```text
transition/forward_prediction_loss
transition/feature_matching_loss
transition/total_loss
```

### Generate the comparison figure

After both runs contain `metrics.jsonl` files, generate the comparison artifacts:

```bash
python - <<'PY'
from evaluation.comparison import save_game_score_comparison

csv_path, png_path = save_game_score_comparison(
    "runs/solaris-state/metrics.jsonl",
    "runs/solaris-transition/metrics.jsonl",
    "runs/solaris-comparison",
    smoothing_window=10,
    dpi=300,
)
print(csv_path)
print(png_path)
PY
```

Outputs:

```text
runs/solaris-comparison/game_score_comparison.csv
runs/solaris-comparison/game_score_comparison.png
```

The CSV contains raw and rolling game scores for both variants. The PNG uses
identical axes and a 300 DPI publication-style figure.

## Legacy / Previous EME Experiments -- not part of the Master's thesis experiment

> **Legacy / previous research -- off the Master's main path.** Everything in
> this section (BiGAN latent discrepancy, EME reward ensembles, zeta scaling,
> episodic counts, learned metric `d_phi`, encoder freezing) is disabled by
> default, is not executed by `--method state` or `--method transition`, and is
> not driven by `scripts/run_master_experiments.sh`. It is kept because it is
> previous published-in-progress work with its own test coverage, and because
> `run_metric_eme_comparison.sh` must keep working for that line of research.

### Metric-based exploration bonus (BiGAN latent + EME scaling)

Adventurer scores a single state by how badly the BiGAN reconstructs it. This
previous contribution replaces that reconstruction-error novelty with a *metric*
bonus computed directly in the encoder's latent space and scaled by the
epistemic disagreement of an ensemble of reward models, as in EME:

\[
b_t=\underbrace{\left\|E_\psi(s_t)-E_\psi(s_{t+1})\right\|_p}_{\text{latent state discrepancy}}
\cdot
\underbrace{\min\!\big(\max(\zeta(r),1),\,M\big)}_{\text{diversity-enhanced scaling}},
\qquad
\zeta(r)=\operatorname{Var}\big(\hat r_1(s_{t+1}),\dots,\hat r_K(s_{t+1})\big).
\]

The bonus needs no generator or discriminator pass: one extra encoder forward
per environment step replaces the reconstruction pipeline. The generator and
discriminator remain in the run because the BiGAN adversarial objective still
trains the encoder that defines the metric.

### Modules

| File | Responsibility |
|------|----------------|
| `exploration/state_discrepancy.py` | `LatentStateDiscrepancy`: `d_t = ||E(s_t)-E(s_{t+1})||_p`, `p in {1,2}`, plus encoder freezing. |
| `exploration/ensemble_scaling.py` | `RewardNet`, `FeatureRewardBuffer`, and `EnsembleRewardVariance`: `K` bootstrapped reward regressors and their prediction variance `zeta(r)`. |
| `exploration/intrinsic_reward.py` | `MetricIntrinsicReward`: assembles `b_t`, clamps `zeta` to `[1, M]`, and optionally applies Eq. (5) reward-scale normalization. |
| `config.py` | `MetricEMEConfig`: `enabled`, `ensemble_scaling`, `ensemble_size` (K), `max_reward_scaling` (M), `latent_norm`, and ensemble/optimization settings. |
| `trainer.py` | Stores `(s_t, s_{t+1})` pairs during `collect_rollouts`, computes `b_t` instead of `NoveltyEstimator` novelty when `metric_eme.enabled`, and trains the ensemble once per PPO update. |

`MetricIntrinsicReward` returns a `MetricNoveltyComponents` object rather than a
bare tensor so the trainer, episodic memory, and logger can consume it exactly
like `NoveltyComponents`; its `normalized_score` field is the intrinsic reward
handed to PPO, and `pixel_error` / `feature_error` are compatibility aliases for
the latent distance and the clamped scale.

### Design notes

- **Ensemble inputs.** Members are MLPs. With `ensemble_input=latent` (default)
  they consume `E(s)`, so the cost is independent of the frame resolution;
  `ensemble_input=observation` flattens raw observations instead.
- **Bootstrap diversity.** Each member owns an independent buffer filled with an
  independently masked subset of the collected data
  (`ensemble_bootstrap_probability`, default `0.5`). Sharing one data set makes
  the members converge and collapses `zeta(r)` to zero.
- **Scaling mode.** `--eme-mode clamped` (default) is EME as published,
  `min(max(zeta,1),M)`: the bonus is never smaller than the pure metric
  distance and never inflated by more than `M`. That formulation assumes `zeta`
  is of order one, which holds for shaped rewards but not for sparse Atari
  rewards -- with almost all regression targets equal to zero the members agree,
  `zeta << 1`, the lower clamp binds everywhere, and V3 collapses onto V2.
  `--eme-mode normalised` divides by the running mean instead,
  `min(zeta/E[zeta], M)`, which is scale-free and therefore ranks transitions
  correctly at any reward magnitude. The ratio is exact: adding an epsilon to
  the numerator would pull the factor back towards one in exactly the tiny-`zeta`
  regime the mode exists to rescue. If the running mean itself falls below
  `zeta_epsilon` the ensemble is treated as degenerate and the factor falls back
  to one, so the bonus degrades to V2 rather than to zero reward.
- **Normalization.** The raw `b_t` is passed through Adventurer's Eq. (5)
  running normalization by default so the intrinsic stream stays on the
  extrinsic-reward scale and `beta` keeps the same meaning across variants.
  Latent distances are much more concentrated than reconstruction errors, so the
  normalized bonus is bounded by `--bonus-normalization-clip` (default `5.0`);
  pass `--disable-bonus-normalization` to feed the raw `b_t` to PPO.
- **Frozen metric.** `--freeze-encoder-after-updates N` stops the BiGAN encoder
  optimizer after `N` PPO updates (`0` freezes before the first rollout), so the
  latent metric, and therefore the scale of `d_t`, becomes stationary. The
  generator and discriminator keep training.

### Experiment variants

| Variant | Command |
|---------|---------|
| **V1 (Baseline)** Adventurer original | `python main.py --env ALE/MontezumaRevenge-v5 --novelty bigan` |
| **V2** Latent discrepancy only | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy` |
| **V3** Latent + clamped `zeta` (EME exactly) | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy --eme True --ensemble_K 5` |
| **V4** Latent + normalised `zeta` (previous work's setting) | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy --eme True --eme-mode normalised --ensemble_K 5` |

The four variants isolate one factor each: V2 removes EME's scaling from the
metric bonus, V3 restores it in its published clamped form, and V4 replaces the
clamp with mean normalization. On a sparse-reward game V3 and V2 are expected to
coincide wherever `zeta < 1`, which is what makes V4 the informative comparison
rather than a redundant fourth curve.

A 512-step CartPole run makes the collapse concrete. Over sixteen PPO updates
the raw ensemble variance rises from `4.6e-5` to `7.5e-3` and never approaches
one, so the two modes see identical `zeta` and produce very different factors:

| Update | `zeta` | V3 scale (clamped) | V4 scale (normalised) |
|--------|--------|--------------------|-----------------------|
| 1 | 4.6e-5 | 1.000 | 2.68 |
| 6 | 2.8e-4 | 1.000 | 3.41 |
| 11 | 1.6e-3 | 1.000 | 3.40 |
| 16 | 5.3e-3 | 1.000 | 2.74 |

V3's factor is constant at the clamp floor for every update, so its bonus is
bit-for-bit V2's. V4's factor varies with the ensemble and, because `E[zeta]` is
an EMA rather than a cumulative mean, it tracks the rising variance instead of
saturating at `M`. `--zeta-momentum` (default `0.99`, roughly a hundred-batch
horizon) sets how quickly the reference level adapts.

`--env` is an alias of `--environment-id`, and `--novelty` is an alias layer over
`--novelty-type`: `bigan` and `state` select Adventurer novelty, `transition`
selects the latent forward-model variant, and `latent_discrepancy` enables the
metric bonus. `--eme True` additionally turns on the ensemble scaling factor, so
V2 is the `zeta == 1` ablation of V3. Combining `--novelty transition` with
`--eme True` is rejected by configuration validation because the metric bonus
replaces, rather than augments, the novelty term.

All three variants can be run over several seeds, with comparison figures, via:

```bash
ENVIRONMENT_ID=ALE/MontezumaRevenge-v5 SEEDS="0 1 2" ./run_metric_eme_comparison.sh
```

### Logged metrics

| Tag | Meaning |
|-----|---------|
| `intrinsic/latent_distance` | mean `||E(s_t)-E(s_{t+1})||_p` |
| `intrinsic/ensemble_variance` | mean raw `zeta(r)` before scaling |
| `intrinsic/mean_ensemble_variance` | running `E[zeta]`, the normalised mode's divisor |
| `intrinsic/bonus_scale` | mean `min(max(zeta,1),M)` |
| `intrinsic/bonus` | mean raw bonus `b_t` |
| `intrinsic/encoder_frozen` | `1.0` once the encoder metric is frozen |
| `ensemble/mean_loss`, `ensemble/buffer_size` | reward-ensemble regression diagnostics |
| `novelty/pixel`, `novelty/feature` | Adventurer's `L_G` and `L_D` in V1; the latent distance and bonus scale in V2/V3 |
| `reward/intrinsic`, `reward/total` | reward streams, directly comparable across variants |

Comparing Adventurer's `pixel_loss` and `feature_loss` against the discrepancy
bonus therefore only requires reading `novelty/pixel`, `novelty/feature`,
`intrinsic/latent_distance`, and `reward/intrinsic` from the same JSONL or
TensorBoard stream of each run.

## Extending the Project

### Parallel environments

The rollout buffer and GAE implementation support a time/environment layout, but
the current top-level adapter and trainer use one environment instance. A
parallel collector can be added while reusing the actor, critic, buffer, GAE,
and PPO modules.

### Custom observation preprocessing

The actor, critic, encoder, discriminator, and generator accept vector,
grayscale-image, CHW-image, and HWC-image observations. Environment-specific
frame stacking, cropping, reward clipping, and life-loss handling should be
implemented in an explicit environment wrapper and recorded in the experiment
configuration.

### Research ablations

The ablations marked *(legacy)* belong to the previous EME/metric line of
research and are not part of the Master's thesis comparison. Useful ablations
include:

- extrinsic-only PPO,
- `alpha=1.0` pixel novelty only,
- `alpha=0.0` feature novelty only,
- `alpha` sweep over `{0.5, 0.7, 0.9, 1.0}`,
- intrinsic reward coefficient sweeps,
- BiGAN update frequency sweeps,
- replay-capacity sweeps,
- different latent dimensions,
- different PPO clipping coefficients,
- *(legacy)* `latent_norm` in `{L1, L2}` for the metric bonus,
- *(legacy)* `max_reward_scaling` (M) and `ensemble_size` (K) sweeps,
- *(legacy)* `eme_mode` in `{clamped, normalised}`,
- *(legacy)* frozen vs. continually trained encoder metric.

## License and Citation

Add the project license and the paper citation before public distribution.
Experiments should cite the Adventurer paper and report the exact configuration
file, environment version, seed, hardware, and evaluation protocol used.
