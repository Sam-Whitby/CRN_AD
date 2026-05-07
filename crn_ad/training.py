"""
Loss function and training loop.

Starting state
--------------
Equilibrate at pH 7 with CURRENT trained parameters (β=1) for
`equil_duration` time units before every schedule run.

Loss design
-----------
Multi-class cross-entropy (InfoNCE / KL divergence):

    L = −log_softmax(τ · scores)[target_idx]

NaN stability
-------------
1. J hard cap via sigmoid prevents ODE stiffness.
2. Smooth pH eliminates RHS discontinuities at segment boundaries.
3. Gradient clipping before Adam.
4. Retry loop: on NaN output, restore pre-step params and Adam state,
   halve lr, retry.  This handles the case where the update is too
   large but the gradient itself is finite.  If all retries fail,
   fall back to the last fully-finite best_params.
   NOTE: gradient NaN-zeroing is intentionally NOT used here — it causes
   a silent plateau because the optimizer takes near-zero steps forever.
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
from itertools import permutations
from functools import partial

from .dynamics import (simulate_schedule, simulate_schedule_scan,
                       make_initial_state, make_triu_indices)


# ---------------------------------------------------------------------------
# Gradient clipping via custom VJP
# ---------------------------------------------------------------------------
# Applied to the ODE output state so that the adjoint sensitivity that flows
# *back through the ODE* starts with a bounded norm.  This is the approach
# described in the JAX docs on custom derivative rules.
#
# Usage:  clipped_state = _clip_grad_norm(max_norm, state)
#   - Forward pass:  identity (no change to values)
#   - Backward pass: scale gradient so its L2 norm ≤ max_norm
#
# nondiff_argnums=(0,) makes max_norm a static Python float — it is captured
# at trace time and never traced through.
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _clip_grad_norm(max_norm, x):
    return x   # identity in the forward pass

def _cgn_fwd(max_norm, x):
    return x, None   # no residuals needed

def _cgn_bwd(max_norm, _residuals, g):
    norm  = jnp.sqrt(jnp.sum(g * g) + 1e-30)
    scale = jnp.minimum(1.0, float(max_norm) / norm)
    return (g * scale,)

_clip_grad_norm.defvjp(_cgn_fwd, _cgn_bwd)


# ---------------------------------------------------------------------------
# Parameter constraints
# ---------------------------------------------------------------------------

def constrain_params(raw, J_max=3.5, S_max=0.0, fixed_phi=None,
                     fixed_J=None, fixed_pKa=None):
    """
    Map unconstrained (ℝ) raw parameters to physical ranges.

    pKa            ∈ [3, 10]       via 3 + 7·σ(raw)  (or fixed_pKa if not None)
    phi            ∈ [0, 1]        via σ(raw)          (or fixed_phi if not None)
    J              ∈ [0.5, J_max]  via 0.5 + (J_max−0.5)·σ(raw)  (or fixed_J if not None)
    monomer_entropy∈ [0, S_max]    via S_max·σ(raw)   (scalar or n-vector)
    """
    out = {
        'pKa': (jnp.array(fixed_pKa, dtype=float) if fixed_pKa is not None
                else 3.0 + 7.0 * jax.nn.sigmoid(raw['pKa'])),
        'phi': (jnp.array(float(fixed_phi)) if fixed_phi is not None
                else jax.nn.sigmoid(raw['phi'])),
        'J':   (jnp.array(float(fixed_J)) if fixed_J is not None
                else 0.5 + (J_max - 0.5) * jax.nn.sigmoid(raw['J'])),
    }
    if S_max > 0.0 and 'monomer_entropy' in raw:
        out['monomer_entropy'] = S_max * jax.nn.sigmoid(raw['monomer_entropy'])
    return out


def unconstrain_params(phys, J_max=3.5, S_max=0.0):
    """Inverse of constrain_params for warm-starting."""
    def _logit(x):
        x = jnp.clip(jnp.array(x), 1e-4, 1 - 1e-4)
        return jnp.log(x / (1 - x))

    pKa_norm = jnp.clip((jnp.array(phys['pKa']) - 3.0) / 7.0, 1e-4, 1 - 1e-4)
    J_norm   = jnp.clip((jnp.array(phys['J']) - 0.5) / (J_max - 0.5), 1e-4, 1 - 1e-4)
    out = {
        'pKa': jnp.log(pKa_norm / (1 - pKa_norm)),
        'phi': _logit(phys['phi']),
        'J':   jnp.log(J_norm   / (1 - J_norm)),
    }
    if S_max > 0.0 and 'monomer_entropy' in phys:
        s_norm = jnp.clip(jnp.array(phys['monomer_entropy']) / S_max, 1e-4, 1 - 1e-4)
        out['monomer_entropy'] = jnp.log(s_norm / (1 - s_norm))
    return out


def _params_finite(params):
    """True iff every leaf of the params pytree is finite."""
    return all(bool(jnp.all(jnp.isfinite(v)))
               for v in jax.tree_util.tree_leaves(params))


def _get_monomer_entropy(p):
    """Return monomer_entropy JAX array or None."""
    s = p.get('monomer_entropy', None)
    return None if s is None else jnp.atleast_1d(jnp.array(s))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def correct_bond_score(state, n, correct_triu_idx):
    """Fraction of total monomer content in correct dimers."""
    free    = state[:n]
    dimers  = state[n:]
    total   = jnp.sum(free) + 2.0 * jnp.sum(dimers)
    correct = jnp.sum(dimers[correct_triu_idx])
    return 2.0 * correct / (total + 1e-10)


def total_monomer_content(state, n):
    """M(t) = Σᵢ[Xᵢ] + 2·Σ_{i≤j}[XᵢXⱼ] — conserved at 1."""
    return jnp.sum(state[:n]) + 2.0 * jnp.sum(state[n:])


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

def all_unique_permutations(seq):
    seen, result = set(), []
    for p in permutations(seq):
        if p not in seen:
            seen.add(p)
            result.append(list(p))
    return result


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(raw_params, all_pH_schedules_array, target_idx,
                 duration_per_seg, static, initial_state):
    """
    Extended softmax cross-entropy loss with pH-7 baseline.

    1. Equilibrate at pH 7 with current parameters.
    2. Record the baseline score (correct-bond fraction at the equilibrated
       pH-7 state — before any schedule is applied).
    3. Score each schedule permutation from equilibrium via vmap.
    4. Build all_scores = [sched_0, ..., sched_K, baseline] and return
       −log_softmax(τ · all_scores)[target_idx].

    The baseline is treated as an additional negative class alongside the
    schedule permutations.  The global minimum (loss = 0) is achieved when
    the target schedule score is strictly highest among all classes including
    the baseline, so the optimiser is simultaneously rewarded for folding
    under the target schedule and penalised for folding at pH 7 or under
    any permutation.
    """
    p = constrain_params(raw_params,
                         J_max=static['J_max'],
                         S_max=static.get('S_max', 0.0),
                         fixed_phi=static.get('fixed_phi'),
                         fixed_J=static.get('fixed_J'),
                         fixed_pKa=static.get('fixed_pKa'))
    mono_s       = _get_monomer_entropy(p)
    sw              = float(static.get('smooth_width', 0.0))
    allowed_mask    = static.get('allowed_mask', None)
    no_self_bonds   = bool(static.get('no_self_bonds', False))
    equil_ramp      = float(static.get('equil_ramp_duration', 0.0))
    grad_clip    = static.get('grad_clip', None)
    # pKa array already has one value per particle (2*N_acids elements)
    pKa_full = p['pKa']

    # Score is computed from the FINAL state after the full schedule —
    # equil gives the starting state, then each permutation is run to completion.
    equil_state = simulate_schedule_scan(
        initial_state, jnp.array([7.0]), static['equil_duration'],
        pKa_full, static['acid_base'], p['phi'], p['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'],
        static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        smooth_width=sw,
        monomer_entropy=mono_s,
        ph_initial=7.0,
        allowed_mask=allowed_mask,
        beta_ramp_duration=equil_ramp,
        no_self_bonds=no_self_bonds,
    )
    # Clip gradient norm flowing back through the equil ODE adjoint.
    if grad_clip is not None:
        equil_state = _clip_grad_norm(float(grad_clip), equil_state)

    # Baseline: correct-bond fraction at the pH-7 equilibrium state, before
    # any schedule is applied.  Gradients flow back through this score so
    # the optimiser is penalised for forming correct dimers at pH 7.
    baseline_score = correct_bond_score(equil_state, static['n'],
                                        static['correct_triu_idx'])

    def score_one(pH_sched):
        final = simulate_schedule_scan(
            equil_state, pH_sched, duration_per_seg,
            pKa_full, static['acid_base'], p['phi'], p['J'],
            static['beta'], static['k0'],
            static['correct_mask'], static['n'],
            static['i_idx'], static['j_idx'],
            n_points=static['n_points_sim'],
            smooth_width=sw,
            monomer_entropy=mono_s,
            ph_initial=7.0,
            allowed_mask=allowed_mask,
            no_self_bonds=no_self_bonds,
        )
        # Clip gradient norm flowing back through each schedule ODE adjoint.
        if grad_clip is not None:
            final = _clip_grad_norm(float(grad_clip), final)
        return correct_bond_score(final, static['n'], static['correct_triu_idx'])

    scores     = jax.vmap(score_one)(all_pH_schedules_array)
    tau        = static.get('tau', 5.0)
    no_baseline = bool(static.get('no_baseline', False))

    if no_baseline:
        if scores.shape[0] == 1:
            # Single schedule, nothing to discriminate: maximise score directly.
            loss = 1.0 - scores[0]
        else:
            log_p = jax.nn.log_softmax(scores * tau)
            loss  = -log_p[target_idx]
    else:
        log_p = jax.nn.log_softmax(jnp.append(scores, baseline_score) * tau)
        loss  = -log_p[target_idx]

    # Always return baseline as the final element so downstream indexing is consistent.
    all_scores = jnp.append(scores, baseline_score)
    return loss, all_scores


# ---------------------------------------------------------------------------
# Fast post-training scoring (JIT + vmap, same as training)
# ---------------------------------------------------------------------------

def compute_scores_fast(p_constrained, all_schedules, duration_per_seg, static):
    """
    Score all schedules in parallel using vmap — same graph as training.

    p_constrained : dict with numpy/jax arrays for pKa, phi, J, monomer_entropy
    Returns       : numpy array (n_schedules + 1,)
                    First n_schedules entries are pH-schedule scores;
                    last entry is the pH-7 baseline score.

    Compiles on first call; subsequent calls with same shapes are cached.
    """
    all_pH_array  = jnp.array(all_schedules, dtype=float)
    initial_state = make_initial_state(static['n'])
    sw            = float(static.get('smooth_width', 0.0))

    allowed_mask  = static.get('allowed_mask', None)
    no_self_bonds = bool(static.get('no_self_bonds', False))
    pKa          = jnp.array(p_constrained['pKa'])  # already 2*N_acids values
    phi          = jnp.array(p_constrained['phi'])
    J            = jnp.array(p_constrained['J'])
    mono_s = (_get_monomer_entropy(p_constrained)
              if p_constrained.get('monomer_entropy') is not None else None)

    equil_ramp = float(static.get('equil_ramp_duration', 0.0))

    # JIT-compiled scoring function (traced once, cached by JAX)
    @jax.jit
    def _score_all(pKa, phi, J, all_pH_array):
        equil = simulate_schedule_scan(
            initial_state, jnp.array([7.0]), static['equil_duration'],
            pKa, static['acid_base'], phi, J,
            static['beta'], static['k0'],
            static['correct_mask'], static['n'],
            static['i_idx'], static['j_idx'],
            n_points=static['n_points_equil'],
            smooth_width=sw,
            monomer_entropy=mono_s,
            ph_initial=7.0,
            allowed_mask=allowed_mask,
            beta_ramp_duration=equil_ramp,
            no_self_bonds=no_self_bonds,
        )

        baseline = correct_bond_score(equil, static['n'],
                                      static['correct_triu_idx'])

        def score_one(pH_sched):
            final = simulate_schedule_scan(
                equil, pH_sched, duration_per_seg,
                pKa, static['acid_base'], phi, J,
                static['beta'], static['k0'],
                static['correct_mask'], static['n'],
                static['i_idx'], static['j_idx'],
                n_points=static['n_points_sim'],
                smooth_width=sw,
                monomer_entropy=mono_s,
                allowed_mask=allowed_mask,
                no_self_bonds=no_self_bonds,
            )
            return correct_bond_score(final, static['n'], static['correct_triu_idx'])

        sched_scores = jax.vmap(score_one)(all_pH_array)
        return jnp.append(sched_scores, baseline)

    return np.array(_score_all(pKa, phi, J, all_pH_array))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    """
    Train CRN parameters to respond selectively to the target pH schedule.

    config keys
    -----------
    N_total            : int  (number of acid species = number of base species)
    M_classifier       : int  (≤ N_total, number of correct acid-base pairs)
    target_pH_schedule : list[float]
    duration_per_seg   : float
    equil_duration     : float   (default 80)
    n_epochs           : int
    learning_rate      : float
    beta, k0           : float
    n_points_sim       : int     (default 40)
    n_points_equil     : int     (default 60)
    tau                : float   (default 5.0)
    J_max              : float   (default 3.5)
    smooth_width       : float   (default 0.0)
    S_max              : float   (default 0.0 = no entropy params)
    per_monomer_entropy: bool    (default False = single shared s)
    seed               : int
    (NaN: training stops at the last finite epoch; report is always generated)
    """
    N_acids = int(config['N_total'])
    M = int(config['M_classifier'])
    assert N_acids >= 1 and M >= 1 and M <= N_acids
    N = 2 * N_acids  # total particle count (N acids + N bases)

    J_max          = float(config.get('J_max', 3.5))
    S_max          = float(config.get('S_max', 0.0))
    smooth         = float(config.get('smooth_width', 0.0))
    per_mono       = bool(config.get('per_monomer_entropy', False))
    no_self_bonds  = bool(config.get('no_self_bonds', False))
    wide_init      = bool(config.get('wide_init', False))
    j_init_max     = bool(config.get('j_init_max', False))
    phi_init_max   = bool(config.get('phi_init_max', False))
    verbose        = bool(config.get('verbose', True))
    fixed_phi_val  = config.get('fixed_phi', None)
    if fixed_phi_val is not None:
        fixed_phi_val = float(np.clip(fixed_phi_val, 0.0, 1.0))
    fixed_J_val    = config.get('fixed_J', None)
    if fixed_J_val is not None:
        fixed_J_val = float(fixed_J_val)   # no upper clip — fixed J bypasses J_max
    pka_default    = bool(config.get('pka_default', False))
    pKa_acid       = float(config.get('pKa_acid', 6.0))
    pKa_base       = float(config.get('pKa_base', 8.0))
    fixed_pKa_val  = (np.array([pKa_acid]*N_acids + [pKa_base]*N_acids, dtype=float)
                      if pka_default else None)

    # ------------------------------------------------------------------
    # Static quantities
    # ------------------------------------------------------------------
    # Particles 0..N_acids-1 are acids; N_acids..2*N_acids-1 are bases.
    # Correct bonds: acid i ↔ base N_acids+i, for i in 0..M-1 (classifier pairs).
    # Particles M..N_acids-1 (acids) and N_acids+M..2*N_acids-1 (bases) are
    # roughness species — they participate in wrong bonds only.
    acid_base_np    = np.array([0]*N_acids + [1]*N_acids, dtype=int)
    correct_mask_np = np.zeros((N, N), dtype=bool)
    for i in range(M):
        correct_mask_np[i, N_acids + i] = True
        correct_mask_np[N_acids + i, i] = True

    i_idx, j_idx = make_triu_indices(N)
    correct_triu_idx = np.array([
        pos for pos, (ii, jj) in enumerate(zip(i_idx, j_idx))
        if correct_mask_np[ii, jj]
    ])

    allowed_mask_jax = None  # specific_bonds not applicable in M/N model

    static = {
        'n'                  : N,
        'n_species'          : N,  # kept for visualize.py compat: 2*N_acids pKa values
        'N_total'            : N_acids,
        'M_classifier'       : M,
        'acid_base'          : jnp.array(acid_base_np),
        'acid_base_np'       : acid_base_np,
        'correct_mask'       : jnp.array(correct_mask_np),
        'correct_mask_np'    : correct_mask_np,
        'i_idx'              : i_idx,
        'j_idx'              : j_idx,
        'correct_triu_idx'   : jnp.array(correct_triu_idx),
        'beta'               : float(config.get('beta', 1.0)),
        'k0'                 : float(config.get('k0',  1.0)),
        # n_points controls how many trajectory points are saved for visualisation.
        # They do not affect ODE accuracy (governed by rtol/atol).
        # Default: ~2 saved points per unit dimensionless time (k0·duration).
        'n_points_sim'       : (int(config['n_points_sim'])
                                if config.get('n_points_sim') is not None
                                else max(20, int(2 * float(config['duration_per_seg'])))),
        'n_points_equil'     : (int(config['n_points_equil'])
                                if config.get('n_points_equil') is not None
                                else max(30, int(2 * float(config.get('equil_duration', 80.0))))),
        'equil_duration'     : float(config.get('equil_duration', 80.0)),
        'tau'                : float(config.get('tau', 5.0)),
        'J_max'              : J_max,
        'S_max'              : S_max,
        'smooth_width'       : smooth,
        'per_monomer_entropy': per_mono,
        'no_self_bonds'      : no_self_bonds,
        'allowed_mask'       : None,
        'fixed_phi'          : fixed_phi_val,
        'fixed_J'            : fixed_J_val,
        'fixed_pKa'          : (fixed_pKa_val.tolist() if fixed_pKa_val is not None
                                else None),
        'no_baseline'        : bool(config.get('no_baseline', False)),
        'equil_ramp_duration': float(config.get('equil_duration', 80.0)) / 2.0,
        'grad_clip'          : (float(config['grad_clip'])
                                if config.get('grad_clip') is not None else None),
    }

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    target_sched  = [float(x) for x in config['target_pH_schedule']]
    all_schedules = all_unique_permutations(target_sched)
    target_idx    = all_schedules.index(target_sched)
    duration      = float(config['duration_per_seg'])
    all_pH_array  = jnp.array(all_schedules, dtype=float)

    # entropy: per-particle if per_mono, else shared
    n_entropy = N if per_mono else 1
    if verbose:
        print(f"N_acids      : {N_acids}  ({M} classifier, {N_acids-M} roughness)")
        print(f"N_bases      : {N_acids}  ({M} classifier, {N_acids-M} roughness)")
        print(f"Particles    : {N}  ({N_acids} acids + {N_acids} bases)")
        print(f"Target sched : {target_sched}")
        print(f"Permutations : {len(all_schedules)}  (target idx = {target_idx})")
        print(f"Equilibration: pH 7,  t = {static['equil_duration']} (β=1)")
        print(f"J_max        : {J_max}  kT")
        if j_init_max:
            print(f"J_init       : {J_max * 0.9:.3g}  kT  (--J_init_max, 90% of J_max)")
        if phi_init_max:
            print(f"phi_init     : 0.9  (--phi_init_max, near but not at boundary)")
        if fixed_J_val is not None:
            print(f"J (fixed)    : {fixed_J_val}  kT  (--fixed_J)")
        if pka_default:
            print(f"pKa (fixed)  : acid={pKa_acid}, base={pKa_base}  (--pka_default)")
        print(f"Smooth width : {smooth}  ({'enabled' if smooth > 0 else 'disabled'})")
        if S_max > 0:
            mode = f"per-particle ({N} values)" if per_mono else "shared (1 value)"
            print(f"Entropy      : S_max = {S_max} kT, {mode}")

    # ------------------------------------------------------------------
    # Initial parameters  (pKa: one per species, shared across types)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(config.get('seed', 42)))
    L = len(target_sched)

    if wide_init:
        # Wide uniform sampling — 2*N_acids values, one per particle.
        pKa_init = list(rng.uniform(3.1, 9.9, N))
        phi_init = float(rng.uniform(0.05, 0.95))
        J_init   = float(rng.uniform(0.55, J_max - 0.01))
    else:
        # Classifier acids (0..M-1): pKa below the corresponding target pH step.
        pKa_clf_acid = [float(np.clip(target_sched[k % L] - 1.5 + rng.normal(0.0, 0.5),
                                      3.1, 9.9)) for k in range(M)]
        # Roughness acids (M..N_acids-1): uniformly spread across full range.
        pKa_rgh_acid = list(rng.uniform(3.1, 9.9, N_acids - M))
        # Classifier bases (N_acids..N_acids+M-1): pKa above the corresponding target pH step.
        pKa_clf_base = [float(np.clip(target_sched[k % L] + 1.5 + rng.normal(0.0, 0.5),
                                      3.1, 9.9)) for k in range(M)]
        # Roughness bases (N_acids+M..2*N_acids-1): uniformly spread.
        pKa_rgh_base = list(rng.uniform(3.1, 9.9, N_acids - M))
        pKa_init = pKa_clf_acid + pKa_rgh_acid + pKa_clf_base + pKa_rgh_base
        phi_init = float(np.clip(0.2 + rng.normal(0.0, 0.05), 0.01, 0.99))
        J_init   = float(np.clip(1.5 + rng.normal(0.0, 0.2),  0.51, J_max - 0.01))

    if j_init_max:
        # Do NOT initialise at exactly J_max: J_norm=1 → raw_J≈9.2 → σ'≈1e-4 (vanishing).
        # Also, phi=1 is a physical degeneracy point (all bond types equivalent → zero gradient).
        # 0.9×J_max gives raw_J≈2.2 → σ'≈0.09, 900× larger gradient while still "near max".
        J_init = J_max * 0.9
    if phi_init_max:
        # Same reasoning: phi=1 → physical degeneracy AND sigmoid vanishing.
        phi_init = 0.9

    init_phys = {
        'pKa': jnp.array(pKa_init),
        'phi': jnp.array(phi_init),
        'J'  : jnp.array(J_init),
    }
    # Snapshot the true starting params (before any gradient step) for reporting.
    init_phys_np = {k: np.array(v) for k, v in init_phys.items()}
    if S_max > 0.0:
        if wide_init:
            s_init = rng.uniform(0.01 * S_max, 0.5 * S_max, n_entropy)
        else:
            s_init = np.clip(
                rng.uniform(0.0, 0.2 * S_max, n_entropy),
                1e-4 * S_max, 0.999 * S_max,
            )
        init_phys['monomer_entropy'] = jnp.array(s_init)
        init_phys_np['monomer_entropy'] = np.array(s_init)

    raw_params = unconstrain_params(init_phys, J_max=J_max, S_max=S_max)

    # All monomers free at equal concentration, no dimers: this is the fully
    # unattached state used as the starting point before every equilibration.
    # Fixed for the entire training run — every gradient step recomputes from
    # here, so there is no state carry-over between epochs.
    initial_state = make_initial_state(N)

    # ------------------------------------------------------------------
    # Optimiser  (lr passed as traced JAX array — no recompile on change)
    # ------------------------------------------------------------------
    clip_norm    = float(config.get('clip_norm', 0.5))
    lr           = float(config.get('learning_rate', 0.02))
    weight_decay = float(config.get('weight_decay', 0.0))

    if weight_decay > 0.0:
        # AdamW: L2 penalty on raw params pulls them toward 0 (sigmoid midpoint),
        # preventing any parameter from drifting to and sticking at a boundary.
        _opt_core = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay),
        )
    else:
        _opt_core = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.scale_by_adam(),
        )
    opt_state = _opt_core.init(raw_params)

    # ------------------------------------------------------------------
    # JIT-compiled step — NO gradient NaN-zeroing inside
    # (zeroing causes a silent zero-gradient plateau; instead the Python
    # loop detects NaN output and retries with a lower lr)
    # ------------------------------------------------------------------
    loss_fn = partial(
        compute_loss,
        all_pH_schedules_array=all_pH_array,
        target_idx=target_idx,
        duration_per_seg=duration,
        static=static,
        initial_state=initial_state,
    )

    @jax.jit
    def step(raw_params, opt_state, lr_val):
        (loss_val, scores), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(raw_params)
        updates, new_opt_state = _opt_core.update(grads, opt_state)
        updates = jax.tree_util.tree_map(lambda u: -lr_val * u, updates)
        new_raw_params = optax.apply_updates(raw_params, updates)
        return new_raw_params, new_opt_state, loss_val, scores

    if verbose:
        print("Compiling JAX graph (first call) ...", flush=True)
    lr_jax = jnp.array(lr)
    raw_params, opt_state, lv, sc = step(raw_params, opt_state, lr_jax)
    if verbose:
        print("Compilation done.\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs      = int(config.get('n_epochs', 300))

    loss_history  = [float(lv)]
    score_history = [np.array(sc)]

    p0 = constrain_params(raw_params, J_max=J_max, S_max=S_max, fixed_phi=fixed_phi_val,
                          fixed_J=fixed_J_val, fixed_pKa=fixed_pKa_val)
    param_history = [_snapshot(p0, S_max)]

    nan_stopped = False
    for epoch in range(1, n_epochs):
        new_params, new_opt, lv, sc = step(raw_params, opt_state, lr_jax)

        if not (np.isfinite(float(lv)) and _params_finite(new_params)):
            if verbose:
                print(f"\nNaN encountered at epoch {epoch} — "
                      f"stopping early and reporting results from epoch {epoch - 1}.",
                      flush=True)
            nan_stopped = True
            break

        raw_params = new_params
        opt_state  = new_opt

        loss_history.append(float(lv))
        score_history.append(np.array(sc))
        p_cur = constrain_params(raw_params, J_max=J_max, S_max=S_max,
                                  fixed_phi=fixed_phi_val, fixed_J=fixed_J_val,
                                  fixed_pKa=fixed_pKa_val)
        param_history.append(_snapshot(p_cur, S_max))

        if verbose and (epoch % max(1, n_epochs // 15) == 0 or epoch == n_epochs - 1):
            pKa_arr    = [float(v) for v in p_cur['pKa']]
            acid_str   = ' '.join(f'{v:.2f}' for v in pKa_arr[:N_acids])
            base_str   = ' '.join(f'{v:.2f}' for v in pKa_arr[N_acids:])
            pKa_str    = (f'acids:[{acid_str}] bases:[{base_str}]'
                          + (' (fixed)' if pka_default else ''))
            s_str      = ''
            if S_max > 0.0 and 'monomer_entropy' in p_cur:
                s = p_cur['monomer_entropy']
                s_str = f' | s̄={float(jnp.mean(s)):.3f} sₘₐₓ={float(jnp.max(s)):.3f}'
            phi_str    = (f'{fixed_phi_val:.3f} (fixed)' if fixed_phi_val is not None
                          else f'{float(p_cur["phi"]):.3f}')
            J_str      = (f'{fixed_J_val:.3f} (fixed)' if fixed_J_val is not None
                          else f'{float(p_cur["J"]):.3f}')
            n_scheds  = len(all_schedules)
            sched_sc  = sc[:n_scheds]
            baseline  = float(sc[n_scheds])
            other_str = (f"mean_other={float(jnp.mean(jnp.delete(sched_sc, target_idx))):.3f} | "
                         if n_scheds > 1 else "")
            bl_tag    = '' if not static.get('no_baseline') else ' (not in loss)'
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"loss={float(lv):.4f} | "
                f"target={float(sc[target_idx]):.3f} | "
                f"{other_str}"
                f"baseline={baseline:.3f}{bl_tag} | "
                f"pKa=[{pKa_str}] | φ={phi_str} | "
                f"J={J_str}{s_str}",
                flush=True,
            )

    if verbose:
        if nan_stopped:
            print("Training terminated early (NaN).")
        else:
            print("\nTraining complete.")

    p_final  = constrain_params(raw_params, J_max=J_max, S_max=S_max,
                               fixed_phi=fixed_phi_val, fixed_J=fixed_J_val,
                               fixed_pKa=fixed_pKa_val)
    mono_s   = _get_monomer_entropy(p_final)
    pKa_full = p_final['pKa']  # already 2*N_acids values, one per particle
    equil_state = simulate_schedule_scan(
        initial_state, jnp.array([7.0]), static['equil_duration'],
        pKa_full, static['acid_base'], p_final['phi'], p_final['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'], static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        smooth_width=smooth, monomer_entropy=mono_s, ph_initial=7.0,
        allowed_mask=allowed_mask_jax,
        beta_ramp_duration=static['equil_ramp_duration'],
        no_self_bonds=no_self_bonds,
    )

    return (raw_params, loss_history, score_history, param_history,
            static, all_schedules, target_idx, equil_state,
            init_phys_np, nan_stopped)


def _snapshot(p, S_max):
    snap = {'pKa': np.array(p['pKa']), 'phi': float(p['phi']), 'J': float(p['J'])}
    if S_max > 0.0 and 'monomer_entropy' in p:
        snap['monomer_entropy'] = np.array(p['monomer_entropy'])
    return snap
