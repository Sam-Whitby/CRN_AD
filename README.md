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

**Note on the ± sign convention in the problem statement:**
The formula `q^± = ±1/(1+10^{±(pH−pKa)})` uses `+` for the *base* case and
`−` for the *acid* case — the reverse of the verbal description "acid/base
respectively."  The implementation follows the physically correct assignment
described here.

### Electrostatic interaction energy

When two monomers i and j bind, the interaction free energy (in units of
k_B T) is:

```
ΔG_{ij} = J · q_i · q_j
```

where `J = e²/(4π ε₀ r D k_BT)` is the dimensionless coupling constant
(effectively the product of Coulomb energy and inverse temperature,
incorporating the inter-charge distance `r` and dielectric constant `D`).

For **correct** pairs the full interaction is used.  For **incorrect** pairs
a steric mismatch factor `φ ∈ [0,1]` reduces it:

```
ΔG_{ij}^{correct}   = J · q_i · q_j
ΔG_{ij}^{incorrect} = φ · J · q_i · q_j
```

`φ < 1` encodes backbone conformational entropy loss or steric hindrance
that destabilises off-target contacts relative to native contacts.

### Detailed balance / rate constants

The forward and backward rate constants obey detailed balance via a
Boltzmann factor:

```
k_f^{ij} = k_0 · exp(−β · ΔG_{ij})
k_b^{ij} = k_0
⟹ k_f / k_b = exp(−β ΔG_{ij})
```

At equilibrium, mass-action gives `[X_i·X_j]_eq = exp(−β ΔG_{ij}) · [X_i][X_j]`,
which is the law of mass action consistent with the Boltzmann distribution.

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

### Denatured initial state

Before each pH schedule, the system is first equilibrated at **β = 0**
(infinite temperature).  At β = 0, `k_f = k_b = k_0` for every reaction,
so the equilibrium is determined purely by combinatorics (no energy bias).
This gives a reproducible, unbiased starting point.

### pH schedules

A target schedule specifies a sequence of pH values, one per time segment
(e.g. `[4.0, 7.0, 9.0, 11.0]`).  All unique permutations of this sequence
are generated automatically.  For four distinct pH values this yields 4! = 24
permutations.

### Score

After integrating the CRN through a schedule, the *score* is the fraction
of total monomer content residing in correct dimers:

```
score(s) = 2 · Σ_{correct pairs} [X_i·X_j] / (Σ_i [X_i] + 2·Σ_{i≤j} [X_i·X_j])
```

### Loss function

The loss is a **softmax cross-entropy** (InfoNCE / multi-class contrastive
loss) over all permutations:

```
L = −log [ exp(τ · score_target) / Σ_s exp(τ · score_s) ]
  = −log_softmax(τ · scores)[target_idx]
```

**Why this loss?**

- Minimising L simultaneously *maximises* the score under the target
  schedule and *suppresses* it under all other permutations.
- This is equivalent to minimising the KL divergence from the ideal
  one-hot distribution `p*(target) = 1` to the model distribution
  `p_s = softmax(τ · scores)`.
- The temperature τ sharpens the gradient (large τ → winner-take-all),
  avoiding the plateau of a mean squared error formulation.
- With many permutations the loss approximates mutual information
  maximisation between schedule identity and folding outcome.

### Trainable parameters

| Parameter | Physical meaning         | Constraint     |
|-----------|--------------------------|----------------|
| `pKa[i]`  | pKa of species i         | [3, 10]        |
| `φ`       | Steric mismatch factor   | [0, 1]         |
| `J`       | Electrostatic coupling   | > 0.5          |

Parameter constraints are enforced via smooth reparameterisations
(sigmoid, softplus), so gradient optimisation remains unconstrained.

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

### Train with defaults (4 species, 4-segment schedule)
```bash
python main.py
```

### Custom run
```bash
python main.py \
  --n_species 6 \
  --target_pH 4 7 9 11 \
  --duration 100 \
  --n_epochs 400 \
  --lr 0.04 \
  --outdir my_outputs
```

### Animate any saved parameter file
```bash
python main.py --mode animate --params_file outputs/trained_params.json
```

### Skip animation (faster)
```bash
python main.py --no_animation
```

### All options
```
--n_species      Number of species (even, max 10)           [4]
--target_pH      pH values for target schedule              [4.0 7.0 9.0 11.0]
--duration       Duration per segment (time units)          [80.0]
--n_epochs       Training epochs                            [250]
--lr             Adam learning rate                         [0.05]
--beta           Inverse temperature                        [1.0]
--k0             Base rate constant                         [1.0]
--tau            Softmax loss temperature                   [6.0]
--n_points_sim   ODE points per segment (accuracy vs speed) [40]
--outdir         Output directory                           [outputs/]
--no_animation   Skip GIF generation
--anim_fps       Animation frames per second                [15]
```

---

## Outputs

All outputs are written to `--outdir` (default `outputs/`):

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `trained_params.json`     | Trained pKa, φ, J (human-readable)              |
| `training_curve.png`      | Loss and per-schedule score vs epoch             |
| `charge_curves.png`       | q(pH) for each species with trained pKa          |
| `final_conc_target.png`   | Bar chart of final concentrations (target sched) |
| `final_conc_perm{k}.png`  | Same for comparison permutations                 |
| `animation_target.gif`    | Animated CRN under target schedule               |
| `animation_perm{k}.gif`   | Animated CRN under alternative permutations      |

---

## Code structure

```
CRN_AD/
├── crn_ad/
│   ├── physics.py    Henderson-Hasselbalch, interaction energies, rate constants
│   ├── dynamics.py   ODE system, simulation (segment + schedule)
│   ├── training.py   Loss function, parameter constraints, training loop
│   └── visualize.py  Training curves, bar charts, animated network diagrams
├── main.py           CLI entry point
└── requirements.txt
```

---

## Critical assessment

### Strengths

1. **Fully differentiable pipeline.**  JAX differentiates through every
   odeint call via the adjoint method, allowing gradient-based optimisation
   of all physical parameters end-to-end.

2. **Thermodynamic consistency.**  All reaction rates are enforced to satisfy
   detailed balance, so the ODE system has a well-defined free energy landscape
   and cannot spontaneously generate entropy.  The equilibrium state is the
   true Boltzmann distribution for the current Hamiltonian.

3. **Physically motivated parameterisation.**  pKa values, coupling strength,
   and steric penalty are all grounded in real polymer chemistry (Henderson-
   Hasselbalch titration, Coulomb interaction, backbone geometry).

4. **Principled loss function.**  The softmax cross-entropy is equivalent to
   KL divergence minimisation and naturally handles an arbitrary number of
   competing schedules without manual weight tuning.

5. **Clear denatured reference state.**  Starting from β = 0 equilibration
   provides a reproducible, unbiased initial condition that mimics thermal
   denaturation and avoids sensitivity to arbitrary initial concentrations.

### Weaknesses and limitations

1. **Mean-field, well-mixed approximation.**  The model treats all species
   as freely diffusing in a homogeneous volume.  Real proteins have a covalent
   backbone that constrains which residues can interact, introduces excluded
   volume, and generates strong conformational entropy.  The effective
   concentration of intramolecular contacts is vastly different from the
   bulk concentration used here.

2. **No conformational degrees of freedom.**  A protein has ~ 3N−6 internal
   coordinates; this model has none.  The folded/unfolded transition is
   replaced by a dimerization equilibrium, which is a qualitative analogy at
   best.

3. **Electrostatics simplified.**  The Debye-Hückel screening, distance
   distributions, and geometry of contact formation are all collapsed into two
   scalars (J and φ).  Salt concentration (ionic strength) is not modelled,
   which significantly affects electrostatic interactions in biology.

4. **Binary acid/base assignment.**  Real amino acids have a single titratable
   group per residue (mostly), but the net charge at a given pH depends on the
   whole protein's electrostatic environment (pKa shifts).  Here each species
   has one group and pKa is independent of context.

5. **Compilation time.**  Because JAX unrolls the Python loop over schedule
   permutations at trace time, the initial JIT compilation can take several
   minutes for many permutations or long schedules.  The compiled function is
   then fast, but this limits interactivity.

6. **pH as a discrete schedule.**  Real experimental pH ramps are continuous.
   The piecewise-constant approximation is reasonable if equilibration within
   each segment is fast compared to the segment duration, which must be
   verified for a given parameter set.

7. **No noise or fluctuations.**  The ODE is deterministic.  Real biochemical
   systems show stochastic fluctuations that are especially important at low
   copy numbers.  A stochastic simulation (Gillespie) would be more realistic
   but harder to differentiate.

8. **Physical relevance of training.**  The trained pKa values and coupling
   constants may not correspond to any real amino acid sequence.  The model is
   better thought of as a proof-of-concept for *in silico* design of pH-
   responsive networks than a quantitative model of protein folding.

### What has not been accomplished

- **Backbone connectivity and conformational entropy** — not included.
- **Stochastic / Gillespie dynamics** — deterministic ODE only.
- **Ionic strength / Debye screening** — J is a single scalar.
- **Multiple titratable groups per species** — one group per species.
- **Finite-size / copy-number effects** — continuous concentrations only.
- **Experimental validation** — purely computational.

### Relevance to protein folding

The model captures the *qualitative* idea that electrostatic complementarity
between residues drives folding and that pH modulates this by shifting
protonation states.  This is relevant to:

- **pH-sensitive protein switches** (e.g. viral fusion proteins, pH-triggered
  toxins, some IDPs).
- **Charge patterning in intrinsically disordered proteins** — recent work
  shows that the sequence of charges along the backbone (not just the net
  charge) controls compaction.
- **Salt-bridge networks in thermophilic proteins** — electrostatics is a
  genuine contributor to thermostability.

However, as a *quantitative* folding model it is too coarse-grained to make
testable predictions.  The main value of this code is as a toy model for
studying how chemical reaction networks can act as sequence classifiers under
thermodynamic driving, which connects to broader questions in:

- **Chemical computation** (how can chemical networks perform logic?)
- **Origins of life** (primordial sensing of environmental cues)
- **Synthetic biology** (designing pH-responsive proteins or RNA aptamers)
