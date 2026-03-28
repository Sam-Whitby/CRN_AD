# CRN_AD — pH-Responsive Chemical Reaction Network with Automatic Differentiation

A differentiable chemical reaction network (CRN) that learns to *fold*
(form correct dimers) only when a specific sequence of pH stimuli is applied.
Parameters are trained end-to-end via JAX automatic differentiation through
the ODE solver.

---

## Physical model

### Species and reactions

The system contains `N` monomer particles in a well-mixed solution.  Every
pair can dimerize:

```
X_i + X_j  ⇌  X_i·X_j    for all 0 ≤ i ≤ j ≤ N−1
```

Particles are grouped into `n_species` species (A, B, C, …) each with `T`
type copies (A1, A2, … with `--n_types T`, default T=1), giving N = n_species × T.

The **correct** (target) pairs are same-species, same-type: (A1–B1), (A2–B2),
(C1–D1), …  This mimics a polymer where neighbouring residues form salt
bridges when the chain folds.

With `--specific_bonds`, only correct-species pairs interact at all (any Ai
with any Bj); φ then controls wrong-type within the correct species (A1–B2
etc.).

### Henderson-Hasselbalch charges

Each species carries a pH-dependent fractional charge:

```
Base-like (e.g. Lys, Arg):  q_i = +1 / (1 + 10^{pH − pKa_i})   → +1 at low pH
Acid-like (e.g. Asp, Glu):  q_i = −1 / (1 + 10^{pKa_i − pH})   → −1 at high pH
```

Even-indexed species (A, C, E, …) are base-like; odd-indexed (B, D, F, …) are
acid-like.  Correct pairs therefore carry opposite charges and attract.

### Interaction free energy

```
ΔG_{ij} = J · q_i · q_j                   (correct pairs)
ΔG_{ij} = φ · J · q_i · q_j              (incorrect pairs, or wrong-type same-species)
ΔG_{ij} = 0                               (forbidden pairs, with --specific_bonds)
```

`J` is the dimensionless electrostatic coupling (kT units); `φ ∈ [0,1]` is
the steric mismatch factor.  Opposite charges give ΔG < 0 (attractive);
like charges give ΔG > 0 (repulsive).

### Monomer conformational entropy (optional)

When `--S_max > 0`, each monomer i carries a conformational-entropy cost
`s_i ∈ [0, S_max]` for losing internal degrees of freedom upon dimerisation:

```
ΔG_{ij} += s_i + s_j
```

### Metropolis kinetics

The rate constants follow **Metropolis-style** kinetics.  The system moves
at full speed whenever a reaction is downhill in free energy, and only pays
a kinetic cost when it must climb an energy barrier:

```
k_f^{ij} = k0 · exp(−β · max(ΔG_{ij}, 0))
k_b^{ij} = k0 · exp(+β · min(ΔG_{ij}, 0))
```

This gives:

| Regime            | k_f       | k_b                 | Physical meaning                          |
|-------------------|-----------|---------------------|-------------------------------------------|
| ΔG ≤ 0 (downhill) | k0        | k0 · exp(β · ΔG) < k0 | Formation is fast; breaking is slowed by the Boltzmann factor |
| ΔG > 0 (uphill)   | k0 · exp(−β · ΔG) < k0 | k0 | Formation is penalised; breaking is fast |

Detailed balance is preserved in both cases:

```
k_f / k_b = exp(−β · ΔG)   ✓   for all ΔG
```

This is the continuous-time analogue of Metropolis–Hastings: the system
never slows the fast direction of a spontaneous process — only the direction
that requires doing work against the free-energy gradient is penalised.  We
are free to make this choice because detailed balance constrains only the
*ratio* k_f/k_b; the overall speed scale (k0) is a free parameter.

### ODE system

The state vector contains free-monomer concentrations `[X_i]` and dimer
concentrations `[X_i·X_j]` (upper triangle only, since [X_i·X_j] = [X_j·X_i]).

Net flux through reaction (i,j):

```
F_{ij} = k_f^{ij} · [X_i][X_j] − k_b^{ij} · [X_i·X_j]
```

Rate equations:

```
d[X_i·X_j]/dt = F_{ij}               (i < j, heterodimer)
d[X_i·X_i]/dt = F_{ii}               (homodimer)
d[X_i]/dt = −Σ_j F_{ij} − F_{ii}    (extra −F_{ii}: 2 monomers consumed per homodimer)
```

Conservation: `Σ_i [X_i] + 2·Σ_{i≤j} [X_i·X_j] = const` (total monomer content M).

---

## Training

### Initial state

Before each pH schedule, the system is equilibrated at **pH 7** for
`--equil_duration` time units.  To avoid a stiff ODE transient at the start
of equilibration, β ramps linearly from 0 to its full value over the first
half of the equilibration period.

### pH schedules

A target schedule specifies a sequence of pH values, one per time segment
(e.g. `[9, 5, 7]`).  All unique permutations are generated automatically;
for 3 distinct values that is 6 permutations.

### pH smoothing (`--smooth_width`)

By default pH changes are instantaneous (step function).  With
`--smooth_width W`, each transition is replaced with a logistic sigmoid ramp
over `W` time units.  This eliminates RHS discontinuities that can cause NaN
gradients in the adjoint ODE, and is recommended whenever `--J_max > 3.5` or
`--lr > 0.05`.

### Score

After integrating the CRN through a schedule, the *score* is the fraction of
total monomer content residing in correct dimers (computed from the **final
state only**, after the full schedule):

```
score = 2 · Σ_{correct pairs} [X_i·X_j] / (Σ_i [X_i] + 2·Σ_{i≤j} [X_i·X_j])
```

### Loss function

Softmax cross-entropy over all permutations:

```
L = −log_softmax(τ · scores)[target_idx]
```

Minimising L simultaneously maximises the score under the target schedule
and suppresses it under all other permutations.

### Trainable parameters

| Parameter | Physical meaning                     | Constraint         |
|-----------|--------------------------------------|--------------------|
| `pKa[i]`  | pKa of species i (shared across types) | [3, 10]          |
| `φ`       | Steric mismatch factor               | [0, 1] or fixed    |
| `J`       | Electrostatic coupling               | [0.5, `J_max`]     |
| `s_i`     | Monomer conformational entropy (opt) | [0, `S_max`] kT    |

All constraints are enforced via sigmoid reparameterisations.
`--fixed_phi VALUE` removes φ from training and holds it at the given value.

### NaN handling

If a NaN loss or NaN parameters are encountered during training, training
stops immediately and the report is generated from the last fully-finite
epoch.

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
python main.py --J_max 5.0 --smooth_width 2.0
```

### Types per species
```bash
# 4 species × 3 types = 12 particles; correct bonds are A1-B1, A2-B2, A3-B3, C1-D1, ...
python main.py --n_species 4 --n_types 3

# Only allow correct-species pairs to interact at all
python main.py --n_species 4 --n_types 3 --specific_bonds
```

### Fixed phi
```bash
# Hold phi at 0.2 for the entire run (not trained)
python main.py --fixed_phi 0.2
```

### Monomer conformational entropy
```bash
python main.py --S_max 2.0                    # single shared value
python main.py --S_max 2.0 --per_monomer_entropy  # per-species value
```

### Generate animated GIFs as well
```bash
python main.py --animate
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
  --seed               Random seed                                   [42]

Simulation accuracy
  --n_points_sim       ODE time points per segment (train)           [40]
  --n_points_equil     ODE time points for equilibration             [60]

ODE stability
  --J_max              Hard cap on coupling constant J (kT)          [3.5]
  --smooth_width       Sigmoid ramp width at each pH transition      [0.0]
                       0 = step function; 1–3 recommended for J_max > 3.5.

Physics flags
  --specific_bonds     Only correct-species pairs interact           [off]
  --fixed_phi VALUE    Fix φ at this value; do not train it          [off]

Conformational entropy
  --S_max              Enable monomer entropy; s_i ∈ [0, S_max] kT   [0.0]
  --per_monomer_entropy Train a separate s_i per species              [off]

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
│   ├── physics.py    Henderson-Hasselbalch charges, interaction energies, rate matrices
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

2. **Thermodynamic consistency.**  All rate constants satisfy detailed balance,
   so the ODE has a well-defined free energy landscape and the equilibrium
   state is the Boltzmann distribution for the current Hamiltonian.

3. **Metropolis kinetics.**  The kinetic rule mirrors Metropolis–Hastings:
   downhill reactions proceed at full speed k0; only uphill steps are
   penalised.  This is physically natural — it ensures that spontaneous
   processes are not artificially slowed, and that kinetic barriers appear
   only where work must be done against the free-energy gradient.

4. **Physically motivated parameterisation.**  pKa values, coupling strength,
   steric penalty, and conformational entropy are all grounded in real polymer
   chemistry.

5. **Principled loss function.**  Softmax cross-entropy naturally handles an
   arbitrary number of competing schedules without manual weight tuning.

6. **Fast evaluation.**  Post-training scoring of all permutations uses
   `jax.jit + jax.vmap`, compiled once and evaluated in parallel.

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
