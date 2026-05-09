# CRN_AD — pH-Responsive Chemical Reaction Network with Automatic Differentiation

A differentiable chemical reaction network (CRN) that learns to selectively *fold* (form correct dimers) only when a specific sequence of pH stimuli is applied. Molecular parameters are trained end-to-end via JAX automatic differentiation through an ODE solver, enabling gradient-based optimisation of dynamic molecular recognition.

---

## Overview

The system models a well-mixed solution of charged monomers that dimerise reversibly. Charges are pH-dependent via the Henderson–Hasselbalch equation, so the interaction free energies — and therefore the rates of bond formation and breaking — change as the external pH is stepped through a schedule. The goal is to find molecular parameters (pKa values, coupling strength J, steric mismatch factor φ) such that a *specific* pH sequence causes the monomers to assemble into correct dimers, while any other permutation of the same pH values does not. This is a form of **kinetically-controlled molecular computation**: the system encodes a temporal password in its thermodynamic parameters.

---

## Particle naming convention

Particles are labelled according to their role:

| Label | Type | Role |
|-------|------|------|
| **A1, A2, …, AM** | Acid | Classifier — forms the correct bond A*i* ↔ B*i* |
| **a1, a2, …, a(N−M)** | Acid | Roughness — no correct bond; introduces kinetic disorder |
| **B1, B2, …, BM** | Base | Classifier — correct partner to A*i* |
| **b1, b2, …, b(N−M)** | Base | Roughness — no correct bond |

In all plots: **classifier** species use thick solid lines; **roughness** species use thin dashed lines. Acids are coloured in red/orange shades; bases in blue/cyan shades.

---

## Physical model

### Species and reactions

The system contains `2N` monomer particles: `N` acid-like particles and `N` base-like particles. Every ordered pair can dimerise reversibly. The `M ≤ N` **classifier pairs** are A*i* ↔ B*i* for *i* = 1…M. The remaining `N−M` acid/base particles (a1…and b1…) are **roughness species** — they have no correct bond and participate only in wrong-bond interactions, introducing energetic disorder into the landscape.

Setting `M = N` recovers a pure-classifier model (no roughness). Setting `M < N` creates the disordered rugged landscape: the spread of pKa values across roughness species generates a Gaussian distribution of wrong-bond energies, giving the system kinetic memory of the sequence history.

### Henderson–Hasselbalch charges

```
q_acid = −1 / (1 + 10^(pKa − pH))    (negative at high pH)
q_base = +1 / (1 + 10^(pH − pKa))    (positive at low pH)
```

Opposite-sign charges attract, so correct classifier acid–base pairs bind.

### Interaction free energies

```
ΔG(Ai, Bi) = J · q_Ai · q_Bi                   correct classifier pair
ΔG(i, j)   = φ · J · qᵢ · qⱼ                  all other pairs (wrong bonds)
```

`J > 0` is the electrostatic coupling strength (kT); `φ ∈ [0, 1]` is the steric mismatch factor. Discrimination requires `φ < 1`, and roughness pKa values spread away from the target pH steps to open the kinetic gate for the correct sequence.

### Kinetics (detailed balance)

```
k_fwd = k₀ · exp(−β · max(ΔG, 0))
k_bwd = k₀ · exp(+β · min(ΔG, 0))
```

The ratio k_fwd / k_bwd = exp(−β·ΔG) satisfies detailed balance. `k₀` is the base rate constant (default 1); `β = 1/kT` (default 1, so J is in kT).

### ODE and conservation

The state vector holds all free-monomer concentrations and upper-triangle dimer concentrations. Total monomer content is exactly conserved. The ODE is integrated with the Diffrax `Tsit5` adaptive solver.

---

## Training protocol

1. **Equilibrate at pH 7** for `equil_duration` time units from a fully dissociated state, or start directly from the **Boltzmann equilibrium** at pH 7 (via `--start_equil`).

2. **Record the baseline score**: the correct-bond fraction at the end of pH-7 equilibration.

3. **Score each unique permutation** of the target pH schedule from the equilibrated state. The score is either:
   - **Final-state score** (default, `--post_duration 0`): correct-dimer fraction Σ[Ai·Bi] / (M/2N) at the end of the schedule.
   - **Integral score** (`--post_duration > 0`): time-averaged correct-dimer concentration integrated over the entire schedule *plus* a subsequent pH-7 post-equilibration phase of length `post_duration`. Formally:
     ```
     score = (1 / (T_sched + T_post)) · ∫₀^{T_sched+T_post} Σ[Ai·Bi](t) dt
     ```
     Normalised by the maximum possible dimer concentration M/2N. This rewards schedules that produce a sustained high level of correct dimers, not just a transient spike at the end.

4. **Compute loss** (InfoNCE-style cross-entropy):

```
all_scores  = [sched_0, ..., sched_K, baseline]
loss        = −log_softmax(τ · all_scores)[target_idx]
```

5. **Adam (or AdamW)** updates unconstrained raw parameters. Physical constraints via sigmoid reparametrisation:

| Parameter | Constraint | Reparametrisation |
|-----------|-----------|-------------------|
| `pKa[i]` | [3, 10] | `3 + 7·σ(raw)` |
| `φ` | [0, 1] | `σ(raw)` |
| `J` | [0.5, J_max] | `0.5 + (J_max−0.5)·σ(raw)` |
| `s_i` (entropy) | [0, S_max] | `S_max·σ(raw)` |

---

## `--start_equil`: differentiable Boltzmann initialisation

With `--start_equil`, each training step begins from the **thermodynamic equilibrium** at pH 7, computed analytically from current trained parameters via a Python `for`-loop fixed-point iteration in JAX. Because JAX unrolls Python loops at trace time, the full gradient ∂loss/∂(pKa, φ, J) flows correctly through the initial state into the ODE dynamics. This avoids the variability of kinetic trapping during ODE equilibration and more closely models a situation where the system is pre-stored at neutral pH before each stimulus sequence.

```bash
python main.py --N 4 --M 2 --n_epochs 300 --start_equil
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires: `jax[cpu]`, `diffrax`, `optax`, `numpy`, `matplotlib`. Use `jax[cuda]` for GPU. `pandas` is optional (used for CSV export).

For the CMA-ES hybrid optimiser (`--optimizer hybrid`): `evosax>=0.1.6` (included in `requirements.txt`).

---

## Usage

### Basic training run

```bash
python main.py
# Default: N=4, M=2 → 4 acids + 4 bases, 2 classifier pairs + 2 roughness pairs
# Target schedule [9, 5, 7], 300 epochs, outdir=outputs/
```

### Custom N, M, and schedule

```bash
python main.py --N 6 --M 3 --target_pH 9 5 7 \
  --duration 40 --equil_duration 100 --n_epochs 500 \
  --lr 0.03 --outdir results/N6M3

# Pure classifier (no roughness)
python main.py --N 4 --M 4 --target_pH 9 5 7

# Many roughness species for strong kinetic memory
python main.py --N 8 --M 2 --target_pH 9 5 7
```

### Boltzmann equilibrium initialisation

```bash
python main.py --start_equil --N 4 --M 2 --n_epochs 300
```

### Multiple restarts (Adam)

```bash
# 8 restarts, wide uniform init; best seed is shown in the summary plot
python main.py --n_restarts 8 --wide_init --n_epochs 400 --J_init_max
```

### CMA-ES + L-BFGS-B hybrid optimizer

With 16 Adam restarts, roughly 70 % of runs converge to a poor local minimum (loss ~1.9) rather than finding a good solution (loss <1.5). The hybrid optimizer replaces all restarts with a single two-stage run that covers the loss landscape systematically:

- **Stage 1 — CMA-ES**: maintains a Gaussian distribution over parameter space and evaluates a population of ~20–30 candidates per generation via `vmap`. No learning-rate tuning: step size and covariance are self-adapted.
- **Stage 2 — L-BFGS-B**: starts from the CMA-ES best solution and converges to a high-quality minimum using JAX-computed gradients (compiled once via `jax.jit`, iterated by scipy's L-BFGS-B).

```bash
python main.py \
  --N 10 --M 1 --target_pH 5 9 7 \
  --duration 30 --J_max 20 --J_init_max --start_equil \
  --optimizer hybrid --cmaes_epochs 200 --lbfgs_epochs 100 \
  --outdir my_outputs --seed 50 --post_duration 100
```

**Expected behaviour on first run**: JAX compiles the vmapped CMA-ES fitness function and the gradient function on the first call (~1–3 min one-time cost). Subsequent runs in the same session reuse the cached XLA and are fast. Progress is printed every `cmaes_epochs // 10` generations.

The Adam multi-restart path (`--optimizer adam`, the default) is unchanged.

### Export time traces to CSV

```bash
# After a training run, export all species trajectories for all permutations
python main.py --mode csv --outdir outputs/
```

This writes `outputs/trajectories.csv` with columns:
- `schedule`, `is_target`, `time`, `segment`, `pH`
- `free_A1`, `free_a1`, `free_B1`, `free_b1`, … (free monomer concentrations)
- `dimer_A1-B1`, `dimer_A1-b1`, … (all triu dimer concentrations)
- `eq_A1-B1`, `eq_A1-b1`, … (Boltzmann equilibrium dimer, constant per segment)

Suitable for direct pandas analysis:

```python
import pandas as pd
df = pd.read_csv('outputs/trajectories.csv')
target = df[df['is_target']]
# e.g. plot correct dimer over time
target.groupby('time')['dimer_A1-B1'].mean().plot()
```

### Regenerate plots from a saved parameter file

```bash
python main.py --mode animate --outdir outputs/
```

### Evaluate fixed parameters without training

```bash
python main.py --mode eval --N 2 --M 1 --target_pH 9 5 7 \
  --eval_pKa 7.5 5.0 9.5 7.0 --eval_phi 0.15 --eval_J 3.5
```

### Integral scoring with post-equilibration phase

```bash
# Score each schedule by the time-averaged correct-dimer level over the
# full schedule *plus* a 30-unit pH-7 post-equilibration window.
python main.py --N 4 --M 2 --n_epochs 300 --post_duration 30

# Combine with Boltzmann initialisation
python main.py --N 4 --M 2 --n_epochs 300 --start_equil --post_duration 30
```

The concentration panel in `summary.png` will show the post-equilibration window shaded in purple, and the score bar chart will reflect the integral metric.

### Large J with smooth transitions

```bash
python main.py --J_max 10.0 --smooth_width 2.0 --J_init_max
```

---

## All arguments

### Core training

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` · `animate` · `both` · `eval` · `csv` |
| `--N` | `4` | Number of acid species (= base species). Total particles = 2N. |
| `--M` | `2` | Classifier pairs M ≤ N. Ai ↔ Bi for i = 1…M. Remaining N−M pairs are roughness. |
| `--target_pH` | `9.0 5.0 7.0` | Target pH schedule. |
| `--duration` | `30.0` | Duration per pH segment (units of 1/k₀). |
| `--equil_duration` | `80.0` | pH-7 pre-equilibration duration. Ignored with `--start_equil`. |
| `--start_equil` | off | Initialise from Boltzmann equilibrium at pH 7 (differentiable, JAX-native). |
| `--n_epochs` | `300` | Training epochs. |
| `--lr` | `0.02` | Adam/AdamW learning rate. |
| `--seed` | `42` | Random seed. With `--n_restarts`, the best seed is shown in the summary. |
| `--outdir` | `outputs` | Output directory. |
| `--n_restarts` | `1` | Train N times; keep the lowest-loss result. |

### Physical model

| Argument | Default | Description |
|----------|---------|-------------|
| `--k0` | `1.0` | Base rate k₀ — absorbed into durations. |
| `--beta` | `1.0` | Inverse temperature β (1 = energies in kT). |
| `--J_max` | `3.5` | Upper bound on J (kT). |
| `--smooth_width` | `0.0` | Sigmoid width for pH transitions (0 = step). Recommended ≥1 for J_max > 5. |
| `--no_self_bonds` | off | Identical particles have ΔG = 0. |

### Fixing parameters

| Argument | Description |
|----------|-------------|
| `--fixed_phi VALUE` | Fix φ; do not train it. |
| `--fixed_J VALUE` | Fix J (kT); do not train it. |
| `--pka_default` | Fix all pKa at `--pKa_acid` / `--pKa_base`. |

### Conformational entropy

| Argument | Default | Description |
|----------|---------|-------------|
| `--S_max` | `0.0` | Enable entropy: sᵢ ∈ [0, S_max] added to ΔGᵢⱼ. |
| `--per_monomer_entropy` | off | Train one sᵢ per particle rather than a shared value. |

### Optimiser and loss

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `adam` | `adam`: Adam with `--n_restarts` independent restarts. `hybrid`: single CMA-ES + L-BFGS-B run (requires `evosax`; ignores `--n_restarts`). |
| `--cmaes_epochs` | `200` | CMA-ES generations for `--optimizer hybrid`. Each generation evaluates `popsize ≈ 4+3·ln(n_params)` candidates in parallel. |
| `--lbfgs_epochs` | `100` | Max L-BFGS-B iterations for `--optimizer hybrid` (gradient-based polish after CMA-ES). |
| `--tau` | `6.0` | Softmax temperature in InfoNCE loss. |
| `--weight_decay` | `0.0` | AdamW L2 weight decay on raw parameters (Adam only). |
| `--grad_clip VALUE` | off | Clip gradient norm through each ODE call (Adam only). |
| `--no_baseline` | off | Exclude pH-7 baseline from loss. |
| `--post_duration` | `0.0` | After each pH schedule, return to pH 7 and simulate for this many time units. Scores are computed as the duration-weighted time-averaged correct-dimer concentration over (schedule + post) phases. Setting to 0 (default) uses the original final-state score. |

### Initialisation

| Argument | Description |
|----------|-------------|
| `--wide_init` | Wide uniform sampling for all restarts. |
| `--J_init_max` | Initialise J near J_max. |
| `--phi_init_max` | Initialise φ near 1. |

---

## Outputs

| File | Description |
|------|-------------|
| `trained_params.json` | Trained pKa, φ, J (and entropy if enabled). |
| `summary.png` | Multi-panel figure (see below). |
| `trajectories.csv` | All species time traces for all permutations (`--mode csv`). |
| `animation_target.gif` | Animated dynamics under the target schedule (`--animate`). |

### Reading `summary.png`

- **Loss panel**: InfoNCE training loss vs epoch. Annotated with best seed (when using `--n_restarts`).
- **pKa evolution**: per-species pKa trajectories. Classifier acids (A1…) are thick solid red/dark-red; roughness acids (a1…) thin dashed orange; classifier bases (B1…) thick solid blue; roughness bases (b1…) thin dashed cyan.
- **φ and J evolution**: parameter trajectories during training.
- **Concentration panel**: correct dimer species (individual lines, purple/teal shades) and aggregate Σ-correct (green solid) and Σ-incorrect (red solid) with Boltzmann equilibrium reference lines (dotted same color).
- **ΔG vs pH panel**: free energy curves for (1) each correct bond A*i*–B*i* (solid colored), (2) each classifier acid vs Σ roughness bases (dashed red, represents total competitor load), (3) each classifier base vs Σ roughness acids (dashed blue).
- **Score bar chart**: correct-bond score for all schedule permutations (green = target, red = others, grey = pH-7 baseline). With `--post_duration > 0`, scores are the duration-weighted time integral over (schedule + post) phases; otherwise the final-state correct-dimer fraction.
- **Parameter table**: complete system and training configuration.

---

## Code structure

```
CRN_AD/
├── crn_ad/
│   ├── physics.py      Henderson–Hasselbalch charges, ΔG matrix, rate constants,
│   │                   boltzmann_equilibrium_jax (differentiable), boltzmann_initial_state
│   ├── dynamics.py     ODE system (Diffrax Tsit5), simulate_schedule, lax.scan
│   ├── training.py     InfoNCE loss, constrain_params, train(), train_cmaes_hybrid()
│   └── visualize.py    plot_summary(), animate_crn(), _mn_particle_labels/style
├── main.py             CLI (train / eval / animate / csv modes), export_csv()
├── boltzmann_scan.py   Thermodynamic parameter sweep (no ODE)
└── requirements.txt
```
