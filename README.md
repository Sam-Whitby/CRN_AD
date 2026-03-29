# CRN_AD — pH-Responsive Chemical Reaction Network with Automatic Differentiation

A differentiable chemical reaction network (CRN) that learns to *fold* (form correct dimers) only when a specific sequence of pH stimuli is applied. Parameters are trained end-to-end via JAX automatic differentiation through the ODE solver.

---

## Physical model

### Species and reactions

The system contains `N` monomer particles in a well-mixed solution. Particles are grouped into `n_species` species (A, B, C, …) each with `T` type copies (A1, A2, … with `--n_types T`, default T=1), giving N = n_species × T. Every pair can dimerize:

```
X_i + X_j  ⇌  X_i·X_j    for all 0 ≤ i ≤ j ≤ N−1
```

The **correct** (target) bonds are same-species, same-type pairs: (A1–B1), (A2–B2), (C1–D1), … This mimics a polymer where neighbouring residues form salt bridges when the chain folds.

### Henderson-Hasselbalch charges

Each species carries a pH-dependent fractional charge. Even-indexed species (A, C, …) are base-like (+1 at low pH); odd-indexed (B, D, …) are acid-like (−1 at high pH). Correct pairs carry opposite charges and attract.

### Interaction free energy

```
ΔG_{ij} = J · q_i · q_j                   (correct pairs)
ΔG_{ij} = φ · J · q_i · q_j              (all other pairs)
ΔG_{ij} = 0                               (forbidden pairs, with --specific_bonds)
ΔG_{ij} = 0                               (identical particles, with --no_self_bonds)
```

`J` is the electrostatic coupling (kT); `φ ∈ [0,1]` is the steric mismatch factor.

### Metropolis kinetics (detailed balance)

```
k_f^{ij} = k0 · exp(−β · max(ΔG_{ij}, 0))
k_b^{ij} = k0 · exp(+β · min(ΔG_{ij}, 0))
```

The ratio k_f/k_b = exp(−β·ΔG) satisfies detailed balance for all ΔG.

### ODE and conservation

State vector: free-monomer concentrations `[X_i]` and upper-triangle dimer concentrations `[X_i·X_j]`. Total monomer content `Σ_i [X_i] + 2·Σ_{i≤j} [X_i·X_j]` is conserved.

---

## Training

### Protocol

1. Equilibrate at **pH 7** for `--equil_duration` time units (β ramps linearly from 0 to its full value over the first half of equilibration to avoid a stiff ODE transient).
2. Run each unique permutation of the target pH schedule.
3. Compute the *score* = fraction of total monomer content in correct dimers at the **final state**.
4. Minimise softmax cross-entropy loss: `L = −log_softmax(τ · scores)[target_idx]`.

### Trainable parameters

| Parameter | Physical meaning                     | Constraint          |
|-----------|--------------------------------------|---------------------|
| `pKa[i]`  | pKa of species i (shared across types) | [3, 10]           |
| `φ`       | Steric mismatch factor               | [0, 1] or fixed     |
| `J`       | Electrostatic coupling               | [0.5, `J_max`] kT   |
| `s_i`     | Monomer conformational entropy (opt) | [0, `S_max`] kT     |

All constraints are enforced via sigmoid reparameterisations.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Train (defaults: 4 species, schedule [9, 5, 7])
```bash
python main.py
```

### Custom run
```bash
python main.py \
  --n_species 6 --target_pH 9 5 7 \
  --duration 30 --n_epochs 400 --lr 0.02 \
  --outdir my_outputs
```

### Multiple restarts (escape local minima)
```bash
# Run 8 training restarts from diverse random initialisations; report the best
python main.py --n_restarts 8 --n_epochs 300
```
Restarts use wide uniform parameter sampling across the full valid range. They run in parallel via `ProcessPoolExecutor` when possible, otherwise sequentially.

### No self-bonds (zero identical-particle interaction energy)
```bash
# A-A, B-B, etc. have ΔG=0 regardless of charge
python main.py --no_self_bonds

# With n_types>1: A1-A1 and B2-B2 are zeroed; A1-A2 and B1-B2 are unaffected
python main.py --n_types 2 --no_self_bonds
```

### Restrict interactions to correct species pairs only
```bash
python main.py --specific_bonds
```

### Larger J with smooth pH transitions (recommended for J_max > 3.5 or lr > 0.05)
```bash
python main.py --J_max 5.0 --smooth_width 2.0
```

### Monomer conformational entropy
```bash
python main.py --S_max 2.0                       # shared value
python main.py --S_max 2.0 --per_monomer_entropy # per-species value
```

### Evaluate fixed parameters (no training)
```bash
python main.py --mode eval \
  --eval_pKa 6.5 7.5 --eval_phi 0.1 --eval_J 3.0
```

### Regenerate plots from saved parameters
```bash
python main.py --mode animate --outdir outputs
```

### All options
```
Core
  --n_species          Number of species (even, max 10)              [4]
  --n_types            Types per species T; N = n_species × T        [1]
  --target_pH          pH values for target schedule                 [9.0 5.0 7.0]
  --duration           Duration per segment (time units)             [30.0]
  --equil_duration     Duration of pH-7 pre-equilibration            [80.0]
  --n_epochs           Training epochs                               [300]
  --lr                 Adam learning rate                            [0.02]
  --beta               Inverse temperature β                         [1.0]
  --k0                 Base rate constant k₀                         [1.0]
  --tau                Softmax loss temperature τ                     [6.0]
  --seed               Random seed for initialisation                [42]
  --n_restarts         Restarts from diverse starting points; best   [1]
                       result (lowest loss) is reported.

ODE stability
  --J_max              Hard cap on coupling constant J (kT)          [3.5]
  --smooth_width       Sigmoid ramp width at each pH transition      [0.0]
                       0 = step function; 1–3 recommended for J_max > 3.5.
  --grad_clip          Clip ODE adjoint gradient L2-norm to VALUE    [off]

Physics flags
  --specific_bonds     Only correct-species pairs interact           [off]
  --no_self_bonds      Zero ΔG for identical-particle pairs          [off]
                       (A-A, B-B; or A1-A1, B2-B2 with n_types>1)
  --fixed_phi VALUE    Fix φ at this value; do not train it          [off]

Conformational entropy
  --S_max              Enable monomer entropy; s_i ∈ [0, S_max] kT   [0.0]
  --per_monomer_entropy Train a separate s_i per species              [off]

Simulation accuracy
  --n_points_sim       ODE time points per segment (train)           [40]
  --n_points_equil     ODE time points for equilibration             [60]

Output
  --outdir             Output directory                              [outputs]
  --params_file        Trained parameters filename                   [trained_params.json]
  --animate            Also generate animated GIFs (requires Pillow)
  --mode               train | animate | both | eval                 [train]
```

---

## Parameter sweep (`scan.py`)

Evaluate correct-dimer yield across a Cartesian product of parameter lists:

```bash
python scan.py \
  --n_species 4 --target_pH 9 5 7 \
  --eval_pKa 8.0 7.0 6.0 5.0 \
  --eval_phi 0.1 0.3 \
  --eval_J 2.0 3.5 \
  --outdir scan_outputs
```

Outputs `scan_results.csv` and `scan_plot.png`. Supports `--no_self_bonds` and all physics flags.

---

## Outputs

| File                      | Description                                          |
|---------------------------|------------------------------------------------------|
| `trained_params.json`     | Trained pKa, φ, J (and s_i if entropy enabled)      |
| `summary.png`             | Training curve, scores, concentration trajectories  |
| `animation_target.gif`    | Animated CRN under target schedule (if `--animate`) |
| `animation_perm{k}.gif`   | Animated CRN under alternative permutations          |

---

## Code structure

```
CRN_AD/
├── crn_ad/
│   ├── physics.py    Henderson-Hasselbalch charges, interaction energies, rate matrices
│   ├── dynamics.py   ODE system, simulation (segment + lax.scan schedule)
│   ├── training.py   Loss function, parameter constraints, training loop
│   └── visualize.py  Summary plot (training curve, scores, concentrations, pH trace)
├── main.py           CLI entry point (train, eval, animate)
├── scan.py           Parameter sweep tool
└── requirements.txt
```
