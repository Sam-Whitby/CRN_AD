# CRN_AD — pH-Responsive Chemical Reaction Network with Automatic Differentiation

A differentiable chemical reaction network (CRN) that learns to *fold*
(form correct dimers) only when a specific sequence of pH stimuli is applied.
Parameters are trained end-to-end via JAX automatic differentiation through
the ODE solver.

---

## Physical model

### Species and reactions

The system contains `n` monomer species (A, B, C, …), all in a hypothetical
well-mixed solution at some fixed volume.  Every ordered pair can dimerize:

```
X_i + X_j  ⇌  X_i·X_j    for all 0 ≤ i ≤ j ≤ n−1
```

The **correct** (target) pairs are adjacent in the alphabet: (A,B), (C,D),
(E,F), …  This mimics a protein in which neighbouring residues along the
backbone form salt bridges when the chain folds.

### Henderson-Hasselbalch charges

Each species carries a pH-dependent fractional charge:

```
Base-like residue (e.g. Lys, Arg):
    q_i = +1 / (1 + 10^{pH − pKa_i})
    → +1 at low pH (protonated), → 0 at high pH

Acid-like residue (e.g. Asp, Glu):
    q_i = −1 / (1 + 10^{pKa_i − pH})
    → 0 at low pH (neutral), → −1 at high pH (deprotonated)
```

Species at even index (A, C, E, …) are assigned as *base-like*;
species at odd index (B, D, F, …) are *acid-like*.  Adjacent correct pairs
therefore carry opposite charges and attract electrostatically.

### Electrostatic interaction energy

When two monomers i and j bind, the interaction free energy (in units of
k_B T) is:

```
ΔG_{ij} = J · q_i · q_j
```

where `J` is the dimensionless electrostatic coupling constant.

For **correct** pairs the full interaction is used.  For **incorrect** pairs
a steric mismatch factor `φ ∈ [0,1]` reduces it:

```
ΔG_{ij}^{correct}   = J · q_i · q_j
ΔG_{ij}^{incorrect} = φ · J · q_i · q_j
```

### Monomer conformational entropy (optional)

When `--S_max > 0`, each monomer i carries a conformational-entropy cost
`s_i ∈ [0, S_max]` (in kT) for losing internal degrees of freedom upon
dimerisation.  This adds a symmetric penalty to every pair:

```
ΔG_{ij} += s_i + s_j
```

Because the term is symmetric in i and j, detailed balance is preserved.
By default a single shared value `s` is trained for all monomers; use
`--per_monomer_entropy` to train a separate value per species.

### Detailed balance / rate constants

```
k_f^{ij} = k_0 · exp(−β · ΔG_{ij})
k_b^{ij} = k_0
⟹ k_f / k_b = exp(−β ΔG_{ij})
```

### ODE system

The state vector contains free monomer concentrations `[X_i]` and dimer
concentrations `[X_i·X_j]` (upper triangle, since `[X_i·X_j] = [X_j·X_i]`).

The net flux through reaction (i,j) is:

```
F_{ij} = k_f^{ij} · [X_i][X_j] − k_0 · [X_i·X_j]
```

Rate equations:

```
d[X_i·X_j]/dt = F_{ij}           (i < j, heterodimer)
d[X_i·X_i]/dt = F_{ii}           (homodimer)
d[X_i]/dt = −Σ_j F_{ij} − F_{ii} (extra −F_{ii} for homodimer, 2 monomers consumed)
```

Conservation: `Σ_i [X_i] + 2·Σ_{i≤j} [X_i·X_j] = const` (total monomer content).

---

## Training

### Initial state

Before each pH schedule, the system is equilibrated at **pH 7** for
`--equil_duration` time units.  This gives a reproducible starting point
independent of the subsequent schedule.

### pH schedules

A target schedule specifies a sequence of pH values, one per time segment
(e.g. `[9, 5, 7]`).  All unique permutations of this sequence are generated
automatically; for 3 distinct values that is 6 permutations.

### pH smoothing (`--smooth_width`)

By default pH changes are instantaneous (step function).  With
`--smooth_width W`, each transition is replaced with a logistic sigmoid ramp
over `W` time units at the start of each segment.  This eliminates RHS
discontinuities that can cause NaN gradients in the adjoint ODE, and is
recommended whenever `--J_max > 3.5` or `--lr > 0.05`.  The pH(t) profile
is shown on the concentration panel of the summary plot.

### Score

After integrating the CRN through a schedule, the *score* is the fraction
of total monomer content residing in correct dimers:

```
score(s) = 2 · Σ_{correct pairs} [X_i·X_j] / (Σ_i [X_i] + 2·Σ_{i≤j} [X_i·X_j])
```

### Loss function

The loss is a **softmax cross-entropy** over all permutations:

```
L = −log [ exp(τ · score_target) / Σ_s exp(τ · score_s) ]
  = −log_softmax(τ · scores)[target_idx]
```

Minimising L simultaneously maximises the score under the target schedule
and suppresses it under all other permutations.

### Trainable parameters

| Parameter | Physical meaning                     | Constraint         |
|-----------|--------------------------------------|--------------------|
| `pKa[i]`  | pKa of species i                     | [3, 10]            |
| `φ`       | Steric mismatch factor               | [0, 1]             |
| `J`       | Electrostatic coupling               | [0.5, `J_max`]     |
| `s_i`     | Monomer conformational entropy (opt) | [0, `S_max`] kT    |

All constraints are enforced via sigmoid reparameterisations, keeping
gradient optimisation unconstrained throughout.

### NaN handling

Training uses a Python-level retry loop: before each gradient step the
current parameters and optimiser state are saved.  If the step produces a
NaN loss or NaN parameters, the step is retried from the pre-step state
with the learning rate halved (up to 3 retries).  If all retries fail, the
optimiser is reset to the best parameters seen so far.  The learning-rate
reduction is persistent across subsequent epochs.

---

## Installation

```bash
pip install -r requirements.txt
```

JAX with GPU support (optional but recommended for larger n_species):
```bash
pip install "jax[cuda12]"
```

---

## Usage

### Train with defaults (4 species, schedule [9, 5, 7])
```bash
python main.py
```

### Custom run
```bash
python main.py \
  --n_species 6 \
  --target_pH 9 5 7 \
  --duration 30 \
  --n_epochs 400 \
  --lr 0.02 \
  --outdir my_outputs
```

### Load saved parameters and regenerate plots
```bash
python main.py --mode animate --outdir outputs
```

### Stability — large J or high learning rate
```bash
# Raise J cap with smooth pH transitions for ODE stability
python main.py --J_max 5.0 --smooth_width 2.0

# Smooth transitions only (keep default J_max)
python main.py --smooth_width 1.5
```

### Monomer conformational entropy
```bash
# Single shared entropy value for all monomers (default off)
python main.py --S_max 2.0

# Separate entropy value per monomer species
python main.py --S_max 2.0 --per_monomer_entropy
```

### Generate animated GIFs as well
```bash
python main.py --animate
```

### All options
```
Core
  --n_species          Number of species (even, max 10)              [4]
  --target_pH          pH values for target schedule                 [9.0 5.0 7.0]
  --duration           Duration per segment (time units)             [30.0]
  --equil_duration     Duration of pH-7 pre-equilibration            [80.0]
  --n_epochs           Training epochs                               [300]
  --lr                 Adam learning rate                            [0.02]
  --beta               Inverse temperature β                         [1.0]
  --k0                 Base rate constant k₀                         [1.0]
  --tau                Softmax loss temperature τ                     [6.0]
  --seed               Random seed                                   [42]

Simulation accuracy
  --n_points_sim       ODE time points per segment (train)           [40]
  --n_points_equil     ODE time points for equilibration             [60]

ODE stability
  --J_max              Hard cap on coupling constant J (kT)          [3.5]
                       Larger values allow stronger binding but
                       increase ODE stiffness; use with --smooth_width.
  --smooth_width       Sigmoid ramp width at each pH transition (time units) [0.0]
                       0 = step function; 1–3 recommended for J_max > 3.5.

Conformational entropy
  --S_max              Enable monomer entropy; s_i ∈ [0, S_max] kT   [0.0]
                       ΔG_ij += s_i + s_j for every dimer pair.
  --per_monomer_entropy Train a separate s_i per species (n values)  [off]
                       Default: one shared value for all monomers.

Output
  --outdir             Output directory                              [outputs]
  --params_file        Trained parameters filename                   [trained_params.json]
  --animate            Also generate animated GIFs (requires Pillow)
  --mode               train | animate | both                        [train]
```

---

## Outputs

All outputs are written to `--outdir` (default `outputs/`):

| File                      | Description                                         |
|---------------------------|-----------------------------------------------------|
| `trained_params.json`     | Trained pKa, φ, J (and s_i if entropy enabled)     |
| `summary.png`             | Training curve, scores, concentration trajectories |
| `animation_target.gif`    | Animated CRN under target schedule (if --animate)  |
| `animation_perm{k}.gif`   | Animated CRN under alternative permutations        |

---

## Code structure

```
CRN_AD/
├── crn_ad/
│   ├── physics.py    Henderson-Hasselbalch, interaction energies, rate constants
│   ├── dynamics.py   ODE system, simulation (segment + lax.scan schedule)
│   ├── training.py   Loss function, parameter constraints, training loop
│   └── visualize.py  Summary plot (training curve, scores, concentrations, pH trace)
├── main.py           CLI entry point
└── requirements.txt
```

---

## Critical assessment

### Strengths

1. **Fully differentiable pipeline.**  JAX differentiates through every
   odeint call via the adjoint method, allowing gradient-based optimisation
   of all physical parameters end-to-end.

2. **Thermodynamic consistency.**  All reaction rates satisfy detailed balance,
   so the ODE has a well-defined free energy landscape and the equilibrium
   state is the true Boltzmann distribution for the current Hamiltonian.

3. **Physically motivated parameterisation.**  pKa values, coupling strength,
   steric penalty, and conformational entropy are all grounded in real polymer
   chemistry.

4. **Principled loss function.**  Softmax cross-entropy naturally handles an
   arbitrary number of competing schedules without manual weight tuning.

5. **Fast evaluation.**  Post-training scoring of all permutations uses
   `jax.jit + jax.vmap`, compiled once and evaluated in parallel — no
   per-permutation recompilation.

### Weaknesses and limitations

1. **Mean-field, well-mixed approximation.**  No backbone connectivity,
   excluded volume, or spatial structure.

2. **No conformational degrees of freedom.**  The folded/unfolded transition
   is replaced by a dimerisation equilibrium — a qualitative analogy.

3. **Electrostatics simplified.**  Debye-Hückel screening, distance
   distributions, and geometry are collapsed into two scalars (J and φ).

4. **No noise or fluctuations.**  Deterministic ODE only; stochastic effects
   at low copy numbers are not captured.

5. **Physical relevance of training.**  Trained pKa values and coupling
   constants may not correspond to any real amino acid sequence.  The model
   is a proof-of-concept for in silico design of pH-responsive networks.

### Relevance to protein folding

The model captures the qualitative idea that electrostatic complementarity
between residues drives folding and that pH modulates this via protonation
states.  It is most relevant to:

- **pH-sensitive protein switches** (viral fusion proteins, pH-triggered toxins)
- **Charge patterning in intrinsically disordered proteins**
- **Salt-bridge networks in thermophilic proteins**
- **Chemical computation** (pH-responsive networks as logic elements)
- **Synthetic biology** (designing pH-responsive proteins or RNA aptamers)
