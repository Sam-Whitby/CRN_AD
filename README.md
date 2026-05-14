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
| **a1, a2, …, a(N−M)** | Acid | Competitor — no correct bond; introduces kinetic disorder |
| **B1, B2, …, BM** | Base | Classifier — correct partner to A*i* |
| **b1, b2, …, b(N−M)** | Base | Competitor — no correct bond |

In all plots: **classifier** species use thick solid lines; **competitor** species use thin dashed lines. Acids are coloured in red/orange shades; bases in blue/cyan shades.

---

## Physical model

### Species and reactions

The system contains `2N` monomer particles: `N` acid-like particles and `N` base-like particles. Every ordered pair can dimerise reversibly. The `M ≤ N` **classifier pairs** are A*i* ↔ B*i* for *i* = 1…M. The remaining `N−M` acid/base particles (a1…and b1…) are **competitor species** — they have no correct bond and participate only in wrong-bond interactions, introducing energetic disorder into the landscape.

Setting `M = N` recovers a pure-classifier model (no competitors). Setting `M < N` creates a disordered landscape: the spread of pKa values across competitor species generates a distribution of wrong-bond energies, giving the system kinetic memory of the sequence history.

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

Both forward and reverse reactions pay an intrinsic activation barrier ε_‡ ≥ 0 (in kT):

```
k_fwd = k₀ · exp(−β · (ε_‡ + max(ΔG, 0)))
k_bwd = k₀ · exp(−β · (ε_‡ + max(−ΔG, 0)))
```

The ratio k_fwd / k_bwd = exp(−β·ΔG) satisfies detailed balance regardless of ε_‡. `k₀` is the base rate constant (default 1); `β = 1/kT` (default 1, so J is in kT).

**Physical meaning of ε_‡:** Even a thermodynamically favourable contact (ΔG < 0) requires chain stretching and solvation-shell reorganisation before the electrostatic gain is realised. This intrinsic barrier slows all association/dissociation rates uniformly by exp(−β·ε_‡) without altering equilibrium constants. With J = 5 kT and ε_‡ = 10 kT, the trapping timescale τ ~ exp(β(ε_‡ + |ΔG|))/k₀ matches that of Metropolis kinetics (ε_‡ = 0) with J = 15 kT, allowing physically realistic coupling strengths.

Setting ε_‡ = 0 (default) exactly recovers the original Metropolis kinetics. Enable training of ε_‡ via `--eps_barrier_max`.

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

For the hybrid optimiser (`--optimizer hybrid`): `evosax>=0.1.6` is listed in `requirements.txt` (retained for optional use).

---

## Usage

### Basic training run

```bash
python main.py
# Default: N=4, M=2 → 4 acids + 4 bases, 2 classifier pairs + 2 competitor pairs
# Target schedule [9, 5, 7], 300 epochs, outdir=outputs/
```

### Custom N, M, and schedule

```bash
python main.py --N 6 --M 3 --target_pH 9 5 7 \
  --duration 40 --equil_duration 100 --n_epochs 500 \
  --lr 0.03 --outdir results/N6M3

# Pure classifier (no competitors)
python main.py --N 4 --M 4 --target_pH 9 5 7

# Many competitor species for strong kinetic memory
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

### Multi-start L-BFGS-B hybrid optimizer

With 16 Adam restarts, roughly 70 % of individual runs converge to a poor local minimum (loss ~1.9). The hybrid optimizer takes a different approach: it runs multiple independent L-BFGS-B runs from quasi-random starting points in **physical parameter space**, then applies a final high-precision polish:

- **Stage 1 — Multi-start L-BFGS-B**: `--cmaes_epochs` independent L-BFGS-B runs (default 10), each starting from a distinct Sobol quasi-random point in physical parameter space (pKa ∈ [3, 10], φ ∈ [0, 1], J ∈ [0.5, J_max]). Box constraints are enforced by scipy. JAX autodiff provides exact gradients via a single compiled value+grad function (~5–30 s one-time compilation).
- **Stage 2 — Final polish**: starts from the Stage 1 best and runs a longer L-BFGS-B pass with tighter tolerances.

Working in physical (constrained) space avoids the logit-space distortion that can cause gradient-free methods to collapse to degenerate boundary solutions. Gradients near parameter boundaries are amplified via the chain rule through `unconstrain_params`, actively pushing optimisation away from φ ≈ 0 or extreme-pKa solutions.

```bash
python main.py \
  --N 10 --M 1 --target_pH 5 9 7 \
  --duration 30 --J_max 20 --start_equil \
  --optimizer hybrid --cmaes_epochs 16 --lbfgs_epochs 120 \
  --outdir my_outputs --seed 42 --post_duration 100
```

**Expected behaviour**: one-time JAX compilation ~5–30 s, then ~2–5 s per restart for N=10. With 16 restarts, at least one typically reaches loss ≤ 1.37 (competitive with or better than Adam's best across 16 restarts). Use `--cmaes_epochs 32` for more reliable coverage of the loss landscape.

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
| `--M` | `2` | Classifier pairs M ≤ N. Ai ↔ Bi for i = 1…M. Remaining N−M pairs are competitors. |
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
| `--eps_barrier_max` | `0.0` | If > 0, enable a trainable intrinsic activation barrier ε_‡ ∈ [0, eps_barrier_max] kT. Both k_fwd and k_bwd are reduced by exp(−β·ε_‡), preserving equilibrium constants. Typical range: 5–10 kT. Use with `--start_equil` to avoid slow ODE equilibration. |
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
| `--optimizer` | `adam` | `adam`: Adam with `--n_restarts` independent restarts. `hybrid`: multi-start L-BFGS-B in physical parameter space followed by a final polish (ignores `--n_restarts`). |
| `--cmaes_epochs` | `10` | Number of independent L-BFGS-B restarts for `--optimizer hybrid`. Each starts from a distinct Sobol quasi-random point. |
| `--lbfgs_epochs` | `100` | Max L-BFGS-B iterations per restart for `--optimizer hybrid`. Final polish uses `max(lbfgs_epochs, 200)`. |
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
- **pKa evolution**: per-species pKa trajectories. Classifier acids (A1…) are thick solid red/dark-red; competitor acids (a1…) thin dashed orange; classifier bases (B1…) thick solid blue; competitor bases (b1…) thin dashed cyan.
- **φ and J evolution**: parameter trajectories during training.
- **Concentration panel**: correct dimer species (individual lines, purple/teal shades) and aggregate Σ-correct (green solid) and Σ-incorrect (red solid) with Boltzmann equilibrium reference lines (dotted same color).
- **ΔG vs pH panel**: free energy curves for (1) each correct bond A*i*–B*i* (solid colored), (2) each classifier acid vs Σ roughness bases (dashed red, represents total competitor load), (3) each classifier base vs Σ roughness acids (dashed blue).
- **Score bar chart**: correct-bond score for all schedule permutations (green = target, red = others, grey = pH-7 baseline). With `--post_duration > 0`, scores are the duration-weighted time integral over (schedule + post) phases; otherwise the final-state correct-dimer fraction.
- **Parameter table**: complete system and training configuration.

---

---

## Chain position model (`--chain_positions`)

### Motivation

The standard model treats φ (steric mismatch), s_i (monomer entropy), and ε_‡ (Arrhenius barrier) as independent trainable scalars. The chain position model replaces all three with a physically grounded representation: each titratable residue has a **position x_i ∈ [0, 1]** along the polymer backbone, from which φ_ij, ΔS_ij, and k0_ij all emerge as functions of the sequence separation d_ij = |x_i − x_j| · L_chain.

This is justified by three classical polymer physics results:

1. **Jacobson–Stockmayer (1950) loop-closure entropy**: Closing a loop of L residues costs ΔS = (3/2) kB ln(L/l₀) in conformational entropy. Longer loops are harder to close. This generates a per-pair entropy penalty that replaces the scalar s_i.

2. **Wilemski–Fixman (1974) / Szabo–Schulten–Schulten (1980) contact rates**: For an ideal Gaussian chain, the rate at which two residues come within contact distance scales as k ∝ L^(−3/2), where L is the sequence separation. Distant residues diffuse slowly toward each other. This generates a position-dependent rate prefactor k0_ij that replaces the scalar k0.

3. **Proximity-based contact selectivity**: Pairs that are close on the chain encounter each other more frequently even when forming wrong bonds. This generates a position-dependent non-native selectivity φ_ij = φ0 · exp(−d_ij/λ_c) that replaces the scalar φ.

### Derived quantities

From positions x ∈ [0, 1]^{2N} and chain length L_chain (residues):

```
d_ij         = |x_i − x_j| · L_chain + ε_d       (sequence separation, residues)

φ_ij         = 1                                   (native pairs: correct_mask = True)
             = φ₀ · exp(−d_ij / λ_c)              (non-native pairs)

ΔS_ij        = (3/2) · ln(d_ij / l₀)             (loop-closure entropy, kT)

k0_ij        = k₀ · (d₀ / d_ij)^α                (Wilemski–Fixman rate prefactor)
```

The free energy becomes ΔG_ij = J · q_i · q_j · φ_ij + ΔS_ij. All quantities are differentiable with respect to x_i through JAX autodiff — gradients flow from the ODE loss back through the chain geometry.

### What the optimiser learns

Training finds residue positions x that satisfy an "anti-Go" architecture: native classifier pairs are placed **close** on the chain (low entropy penalty, fast association rate, selective: φ_ij = 1 by definition), while competitor pairs are placed at **intermediate** distances that maximise landscape roughness. This recapitulates the sequence design principles of intrinsically disordered proteins (IDPs) that function through charge-pattern-dependent phase separation, as quantified by the κ parameter of Das & Pappu (2013).

The **Zwanzig landscape roughness** σ² = Var[φ_ij · J · q_i · q_j] over non-native pairs provides kinetic memory: ⟨k⟩ ~ k₀ · exp(−β²σ²/2) slows diffusion through the wrong-bond landscape exponentially in J². This roughness **emerges from the trained chain geometry** — no additional free parameters.

### Usage

Chain position mode requires `--start_equil` (the Boltzmann equilibrium initial condition is unaffected by chain kinetics and avoids slow ODE equilibration at large ε_‡-equivalent barriers). It is incompatible with `--optimizer hybrid`.

```bash
# Basic chain position training
python main.py --chain_positions --start_equil \
  --N 4 --M 2 --n_epochs 300 --J_max 5.0 \
  --target_pH 9 5 7

# With custom chain parameters
python main.py --chain_positions --start_equil \
  --N 6 --M 3 --n_epochs 400 --J_max 5.0 \
  --L_chain 150 --lambda_c 25 --l0 4 --d0 4 --phi0 0.8 \
  --target_pH 9 5 7 --outdir results/chain

# Reload and visualise saved chain mode results
python main.py --mode animate --outdir results/chain
```

### Chain position arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--chain_positions` | off | Enable chain position model. Trains 2N residue positions x_i ∈ [0,1] instead of scalar φ, s_i, ε_‡. |
| `--L_chain` | `100.0` | Total chain length (residues). Absolute separation d_ij = \|x_i−x_j\| × L_chain. |
| `--lambda_c` | `20.0` | Contact locality scale λ_c (residues). Non-native φ_ij decays as exp(−d/λ_c). Pairs beyond ~3λ_c are essentially non-interacting. |
| `--l0` | `3.0` | Entropy reference separation l₀ (residues). ΔS_ij = 0 at d_ij = l₀ (≈1 Kuhn length for an IDP). |
| `--d0` | `3.0` | Rate reference separation d₀ (residues). k0_ij = k₀ at d_ij = d₀. |
| `--phi0` | `1.0` | Maximum non-native contact strength at the shortest separation. |
| `--chain_alpha` | `1.5` | Wilemski–Fixman exponent α for k0_ij ∝ d_ij^{−α}. α = 1.5: Gaussian chain; α = 1.76: self-avoiding chain (good solvent). |

### Saved output in chain mode

`trained_params.json` stores the residue positions alongside the usual parameters:

```json
{
  "pKa": [...],
  "phi": 0.12,       
  "J": 4.3,
  "x": [0.08, 0.12, 0.31, 0.44, 0.56, 0.68, 0.82, 0.91],
  "chain_mode": true,
  "L_chain": 100.0,
  "lambda_c": 20.0,
  "l0": 3.0,
  "d0": 3.0,
  "phi0": 1.0,
  "chain_alpha": 1.5
}
```

`phi` in the JSON is the **mean non-native selectivity** (for reporting), not a trainable scalar. The matrices φ_ij, ΔS_ij, k0_ij are recomputed from `x` and chain hyperparameters at load time.

---

## Code structure

```
CRN_AD/
├── crn_ad/
│   ├── physics.py        Henderson–Hasselbalch charges, ΔG matrix, rate constants,
│   │                     boltzmann_equilibrium_jax (differentiable), boltzmann_initial_state
│   ├── dynamics.py       ODE system (Diffrax Tsit5), simulate_schedule, lax.scan
│   ├── chain_physics.py  Chain position model: Jacobson–Stockmayer entropy,
│   │                     Wilemski–Fixman k0_ij, proximity-based φ_ij (differentiable)
│   ├── training.py       InfoNCE loss, constrain_params, train(), train_cmaes_hybrid()
│   └── visualize.py      plot_summary(), animate_crn(), _mn_particle_labels/style
├── main.py               CLI (train / eval / animate / csv modes), export_csv()
├── boltzmann_scan.py     Thermodynamic parameter sweep (no ODE)
└── requirements.txt
```
