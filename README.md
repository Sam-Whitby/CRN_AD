# CRN_AD — pH-Responsive Chemical Reaction Network with Automatic Differentiation

A differentiable chemical reaction network (CRN) that learns to selectively *fold* (form correct dimers) only when a specific sequence of pH stimuli is applied. Molecular parameters are trained end-to-end via JAX automatic differentiation through an ODE solver, enabling gradient-based optimisation of dynamic molecular recognition.

---

## Overview

The system models a well-mixed solution of charged monomers that dimerise reversibly. Charges are pH-dependent via the Henderson–Hasselbalch equation, so the interaction free energies — and therefore the rates of bond formation and breaking — change as the external pH is stepped through a schedule. The goal is to find molecular parameters (pKa values, coupling strength J, steric mismatch factor φ) such that a *specific* pH sequence causes the monomers to assemble into correct dimers, while any other permutation of the same pH values does not. This is a form of **kinetically-controlled molecular computation**: the system encodes a temporal password in its thermodynamic parameters.

---

## Physical model

### Species and reactions

The system contains `2N` monomer particles: `N` acid-like particles (indices 0…N−1) and `N` base-like particles (indices N…2N−1). Every ordered pair can dimerise:

```
Xᵢ + Xⱼ  ⇌  Xᵢ·Xⱼ      for all 0 ≤ i ≤ j ≤ 2N−1
```

The `M ≤ N` **classifier pairs** are acid *i* ↔ base *N+i* for *i* = 0…M−1. The remaining `N−M` acid/base particles (indices M…N−1 and N+M…2N−1) are **roughness species** — they have no correct bond and participate only in wrong-bond interactions, introducing energetic disorder into the landscape. All `2N` pKa values are independent trained parameters.

Setting `M = N` recovers a pure-classifier model (no roughness). Setting `M < N` creates the disordered rugged landscape: the spread of pKa values across roughness species generates a Gaussian distribution of wrong-bond energies, giving the system kinetic memory of the sequence history.

### Henderson–Hasselbalch charges

The fractional charge of particle k at a given pH is:

```
q_k = +1 / (1 + 10^(pH − pKa_k))    (acid-like, charge → −1 at high pH)
q_k = −1 / (1 + 10^(pKa_k − pH))    (base-like, charge → +1 at low pH)
```

(Convention: acids carry negative charge above their pKa; bases carry positive charge below theirs. Opposite charges attract, so correct acid–base pairs bind.)

### Interaction free energies

```
ΔGᵢⱼ = J · qᵢ · qⱼ          correct classifier pair (acid i ↔ base N+i, i < M)
ΔGᵢⱼ = φ · J · qᵢ · qⱼ     all other acid–base pairs (wrong bonds)
ΔGᵢⱼ = 0                    same-sign pairs (acid–acid, base–base)
ΔGᵢⱼ = 0                    identical particles (with --no_self_bonds)
```

`J > 0` is the electrostatic coupling strength (kT); `φ ∈ [0, 1]` is the steric mismatch factor applied uniformly to all wrong bonds. `φ = 0` suppresses all wrong bonds; `φ = 1` makes all bonds energetically equivalent. Discrimination requires `φ < 1`, but the roughness species also need their pKa values spread away from the target pH steps to open the kinetic gate for the correct sequence.

### Kinetics (detailed balance)

```
k_fwd = k₀ · exp(−β · max(ΔG, 0))
k_bwd = k₀ · exp(+β · min(ΔG, 0))
```

The ratio k_fwd / k_bwd = exp(−β·ΔG) satisfies detailed balance for all ΔG. `k₀` is the base rate constant (default 1); `β = 1/kT` is the inverse temperature (default 1, so J is measured directly in kT).

### ODE and conservation

The state vector holds all free-monomer concentrations `[Xᵢ]` and upper-triangle dimer concentrations `[Xᵢ·Xⱼ]`. The total monomer content `Σᵢ[Xᵢ] + 2·Σᵢ≤ⱼ[Xᵢ·Xⱼ]` is exactly conserved. The ODE is integrated with the Diffrax `Tsit5` adaptive solver.

---

## Training protocol

1. **Equilibrate at pH 7** for `equil_duration` time units. The inverse temperature β ramps linearly from 0 to its full value over the first half of equilibration to avoid an ODE stiffness transient at startup. The initial state is all monomers free at equal concentration.

2. **Record the baseline score**: the correct-bond fraction at the end of pH-7 equilibration, before any schedule is applied.

3. **Score each unique permutation** of the target pH schedule from the equilibrated state. The score is the fraction of total monomer content in correct dimers at the *final time step* of the schedule.

4. **Compute loss** (InfoNCE-style cross-entropy):

```
all_scores  = [sched_0, sched_1, ..., sched_K, baseline]
loss        = −log_softmax(τ · all_scores)[target_idx]
```

The baseline appears as an additional negative class. The loss reaches zero only when the target schedule score is strictly highest among all classes, so the optimiser simultaneously rewards folding under the target schedule and penalises folding at pH 7 or under any permutation. With `--no_baseline` the baseline is excluded; with a single-element target schedule the loss becomes `1 − score` (direct maximisation).

5. **Adam (or AdamW)** updates the unconstrained raw parameters. Physical constraints are enforced via sigmoid reparametrisation:

| Parameter | Constraint | Reparametrisation |
|-----------|-----------|-------------------|
| `pKa[i]` | [3, 10] | `3 + 7·σ(raw)` |
| `φ` | [0, 1] | `σ(raw)` |
| `J` | [0.5, J_max] | `0.5 + (J_max−0.5)·σ(raw)` |
| `s_i` (entropy) | [0, S_max] | `S_max·σ(raw)` |
| Fixed params | constant | pass-through (no raw param) |

**Sigmoid boundary warning**: initialising φ or J exactly at their maximum values (φ=1 or J=J_max) places the raw parameter at logit(1−ε) ≈ 9.2, where the sigmoid gradient is ≈10⁻⁴. Furthermore, φ=1 is a physical degeneracy point where all bond types are energetically equivalent and the loss gradient is exactly zero by symmetry. The `--phi_init_max` and `--J_init_max` flags therefore initialise at 90% of the maximum (φ=0.9, J=0.9·J_max) to maintain useful gradients while starting near the boundary.

### Numerical stability

The ODE solver uses [Diffrax](https://docs.kidger.site/diffrax/) `Tsit5` with `RecursiveCheckpointAdjoint`: checkpoints of the forward trajectory are stored and differentiated through directly, avoiding the stiff backwards-ODE NaN instability of the classical adjoint method. All computation runs in float64.

### Dimensional analysis: k₀ and duration

The ODE right-hand side factorises as `dX/dt = k₀·F(X; β·J, φ, pKa, pH(t))`. Substituting `τ = k₀·t` removes k₀ from the equation — only the products `k₀×duration` and `k₀×equil_duration` matter physically. Internally the code always uses `k₀=1` and absorbs the CLI `--k0` into the durations, so `--k0 5 --duration 30` is exactly equivalent to `--k0 1 --duration 150`. The saved `trained_params.json` records the effective (multiplied) durations and `k0=1`.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires: `jax[cpu]`, `diffrax`, `optax`, `numpy`, `matplotlib`. Use `jax[cuda]` for GPU.

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
python main.py \
  --N 6 --M 3 --target_pH 9 5 7 \
  --duration 40 --equil_duration 100 --n_epochs 500 \
  --lr 0.03 --outdir results/N6M3

# Pure classifier (no roughness): M = N
python main.py --N 4 --M 4 --target_pH 9 5 7

# Many roughness species for strong kinetic memory
python main.py --N 8 --M 2 --target_pH 9 5 7
```

### Fix parameters during training

```bash
# Fix φ=0.5 (do not train it); train pKa and J freely
python main.py --fixed_phi 0.5

# Fix J=8 kT (bypasses J_max); train pKa and φ freely
python main.py --fixed_J 8.0

# Fix pKa values at acid=6.0, base=8.0 (Henderson–Hasselbalch defaults)
python main.py --pka_default

# Fix pKa with custom values
python main.py --pka_default --pKa_acid 5.5 --pKa_base 8.5

# Fix all three; only pKa is trained
python main.py --fixed_phi 0.3 --fixed_J 5.0
```

### Large J: stronger binding with smooth pH transitions

```bash
# Smooth pH transitions prevent ODE stiffness at high J
python main.py --J_max 10.0 --smooth_width 2.0

# Start training near J_max to explore strong-coupling regime
python main.py --J_max 15.0 --J_init_max --smooth_width 3.0
```

### Multiple restarts to escape local minima

```bash
# 8 restarts with diverse random initialisations; keeps the best result
python main.py --n_restarts 8 --wide_init --n_epochs 400

# Standard init for restart 0, wide init for the rest
python main.py --n_restarts 5 --n_epochs 300 --lr 0.04
```

### AdamW weight decay (prevents parameters from drifting to boundaries)

```bash
python main.py --weight_decay 1e-3 --n_epochs 500 --lr 0.04
```

### Loss variants

```bash
# Exclude the pH-7 equilibrium baseline from the loss
python main.py --no_baseline

# Single-value target: loss = 1 − score (direct maximisation, no schedule comparison)
python main.py --target_pH 9 --no_baseline
```

### Conformational entropy

```bash
# Shared monomer entropy parameter s ∈ [0, 2] kT
python main.py --S_max 2.0

# Per-species entropy (one sᵢ per species)
python main.py --S_max 2.0 --per_monomer_entropy
```

### Bond topology flags

```bash
# Identical particles have ΔG=0 (A–A, B–B, etc.)
python main.py --no_self_bonds
```

### Evaluate fixed parameters without training

```bash
# N=2, M=1: 4 particles total (2 acids + 2 bases), 1 classifier pair + 1 roughness pair
# --eval_pKa must supply 2*N = 4 values: [acid0, acid1, base0, base1]
python main.py --mode eval \
  --N 2 --M 1 --target_pH 9 5 7 \
  --eval_pKa 7.5 5.0 9.5 7.0 \
  --eval_phi 0.15 \
  --eval_J 3.5
```

### Regenerate plots from a saved parameter file

```bash
python main.py --mode animate --outdir outputs/
```

### Generate animated GIFs

```bash
python main.py --animate --outdir outputs/
# or equivalently
python main.py --mode both --outdir outputs/
```

---

## All arguments

### Core training

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` · `animate` · `both` · `eval` |
| `--N` | `4` | Number of acid species (= number of base species). Total particle count = 2N. |
| `--M` | `2` | Number of classifier pairs (M ≤ N). Acid *i* ↔ base *N+i* for *i* = 0…M−1. Remaining N−M acid/base pairs are roughness-only species. |
| `--target_pH` | `9.0 5.0 7.0` | Target pH schedule (one float per segment). The model is trained to fold only under this exact sequence. |
| `--duration` | `30.0` | Duration of each pH segment (units of 1/k₀). Only the product k₀×duration matters — see note on k₀. |
| `--equil_duration` | `80.0` | Duration of pH-7 pre-equilibration. Should satisfy equil\_duration ≫ exp(J) to reach thermodynamic equilibrium (e.g. J=3.5 → ≫33; J=10 → ≫22,000). |
| `--n_epochs` | `300` | Number of gradient descent epochs. |
| `--lr` | `0.02` | Adam/AdamW learning rate. |
| `--seed` | `42` | Random seed for parameter initialisation. |
| `--outdir` | `outputs` | Output directory (created if absent). |
| `--params_file` | `trained_params.json` | Filename for saved trained parameters within `--outdir`. |
| `--n_restarts` | `1` | Train N times from different random starting points; keep the result with the lowest final loss. Restarts run in parallel via `ProcessPoolExecutor`. |

### Physical model

| Argument | Default | Description |
|----------|---------|-------------|
| `--k0` | `1.0` | Base rate constant k₀. Absorbed into durations internally; only k₀×duration matters. Equivalent to rescaling both durations. |
| `--beta` | `1.0` | Inverse temperature β. When J is a free parameter, β is degenerate with J (only β·J enters the rates). Setting β=1 means J is measured in kT. |
| `--J_max` | `3.5` | Upper bound on the coupling J (kT). Larger values allow stronger binding but increase ODE stiffness; use with `--smooth_width`. |
| `--smooth_width` | `0.0` | Logistic sigmoid width (time units) for smoothing pH transitions. `0` = step function. Recommended 1–3 for `J_max > 5`. |
| `--no_self_bonds` | off | If set, identical particles have ΔG=0 (A–A, a–a, etc.). |

### Fixing parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--fixed_phi VALUE` | off | Fix φ at this value for the entire run; do not train it. |
| `--fixed_J VALUE` | off | Fix J (kT) at this value; do not train it. Not capped by `--J_max`. |
| `--pka_default` | off | Fix all pKa values and do not train them. All N acids get `--pKa_acid`; all N bases get `--pKa_base`. |
| `--pKa_acid` | `6.0` | pKa for all acid species when `--pka_default` is set. |
| `--pKa_base` | `8.0` | pKa for all base species when `--pka_default` is set. |

### Conformational entropy

| Argument | Default | Description |
|----------|---------|-------------|
| `--S_max` | `0.0` | Enable monomer conformational entropy. Each particle contributes `sᵢ ∈ [0, S_max]` kT to ΔGᵢⱼ = ΔGᵢⱼ + sᵢ + sⱼ. `0` = disabled. |
| `--per_monomer_entropy` | off | If set, train a separate sᵢ per particle (2N values). Default: one shared value. |

### Optimiser and loss

| Argument | Default | Description |
|----------|---------|-------------|
| `--tau` | `6.0` | Softmax temperature τ in the InfoNCE loss. Higher values sharpen discrimination between schedules. Does not affect ODE physics. |
| `--weight_decay` | `0.0` | AdamW L2 weight decay on raw (unconstrained) parameters. Provides a restoring force toward the sigmoid midpoint, preventing parameters from drifting to and sticking at boundaries. Try 1e-4 to 1e-2. |
| `--grad_clip VALUE` | off | Clip the L2 norm of gradients flowing back through each ODE call (JAX custom\_vjp). Try 1–10 if adjoint gradients become erratic at high J. |
| `--no_baseline` | off | Exclude the pH-7 equilibrium baseline from the loss. With a single-element target schedule, the loss becomes `1 − score` (direct maximisation). |

### Initialisation

| Argument | Default | Description |
|----------|---------|-------------|
| `--wide_init` | off | All restarts (including restart 0) use wide uniform sampling: pKa ~ U[3.1, 9.9], φ ~ U[0.05, 0.95], J ~ U[0.55, J\_max]. |
| `--J_init_max` | off | Initialise J at 0.9×J\_max rather than the default ~1.5 kT. |
| `--phi_init_max` | off | Initialise φ at 0.9 rather than the default ~0.2. |

### Visualisation

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_points_sim` | auto | ODE output points saved per schedule segment (visualisation only; does not affect ODE accuracy). Default: `max(20, 2×duration)`. |
| `--n_points_equil` | auto | ODE output points saved during equilibration. Default: `max(30, 2×equil_duration)`. |
| `--animate` | off | Also produce animated GIFs of the CRN under the target schedule and two alternative permutations. Requires Pillow. |

### Eval mode

| Argument | Description |
|----------|-------------|
| `--eval_pKa` | pKa values, one per particle — must supply exactly 2×N values (required for `--mode eval`). |
| `--eval_phi` | Steric mismatch factor φ ∈ [0, 1] (required). |
| `--eval_J` | Coupling J (kT). One value → same for all pairs; M values → one per classifier pair. |
| `--eval_monomer_entropy` | Monomer entropy (kT). One value → shared; 2N values → per-particle. |

---

## Outputs

| File | Description |
|------|-------------|
| `trained_params.json` | Trained pKa, φ, J (and entropy if enabled). Also records effective durations and k₀=1. |
| `summary.png` | Six-panel figure: training loss curve, score histories, φ and J evolution, concentration trajectories, schedule bar chart, and parameter table. |
| `animation_target.gif` | Animated CRN dynamics under the target schedule (with `--animate`). |
| `animation_perm{k}.gif` | Animated CRN dynamics under alternative permutations. |

### Reading `summary.png`

- **Loss / scores panel**: InfoNCE training loss and per-schedule scores vs epoch. Lower loss = better target discrimination.
- **φ, J evolution**: parameter trajectories during training; dotted line marks the final value.
- **Concentration trajectories**: free-monomer and dimer concentrations vs time under the target schedule. A successful run shows correct dimers accumulating at the end.
- **Score bar chart**: final scores for all schedule permutations and the pH-7 baseline. The target bar should be tallest.
- **Parameter table**: complete record of all flags and physical parameters used in the run.

---

## Boltzmann equilibrium sweep

`boltzmann_scan.py` scans thermodynamic equilibrium properties over a Cartesian grid of parameters without solving any ODEs. It uses a damped mean-field fixed-point iteration to compute the Boltzmann equilibrium directly.

```bash
# Default sweep: phi × J × N × M × pH
python boltzmann_scan.py

# Fixed acid/base pKa (default 6.0 and 8.0)
python boltzmann_scan.py --pKa_acid 5.5 --pKa_base 8.5 --beta 1.0
```

Note: `boltzmann_scan.py` uses the legacy `n_species`/`n_types` parameterisation and has not yet been updated to the M/N model. The companion notebook `boltzmann_notebook.ipynb` visualises these results.

---

## Code structure

```
CRN_AD/
├── crn_ad/
│   ├── physics.py      Henderson–Hasselbalch charges, ΔG matrix, rate constants
│   ├── dynamics.py     ODE system (Diffrax Tsit5), simulate_schedule, lax.scan loop
│   ├── training.py     Loss function, constrain_params, train(), _print_param_table
│   └── visualize.py    plot_summary() — six-panel figure; animate_crn() — GIF output
├── main.py             CLI entry point (train / eval / animate modes)
├── boltzmann_scan.py   Thermodynamic equilibrium parameter sweep (no ODE)
├── boltzmann_notebook.ipynb   Interactive analysis of boltzmann_scan.csv
├── scan_analysis.ipynb        Analysis notebooks for training runs
└── requirements.txt
```
