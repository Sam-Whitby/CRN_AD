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
from .physics import (boltzmann_initial_state as _boltzmann_initial_state,
                       boltzmann_equilibrium_jax as _boltzmann_equilibrium_jax)
from .chain_physics import compute_chain_quantities as _compute_chain_quantities


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

def constrain_params(raw, J_max=3.5, S_max=0.0, eps_max=0.0, fixed_phi=None,
                     fixed_J=None, fixed_pKa=None, chain_mode=False,
                     chain_static=None):
    """
    Map unconstrained (ℝ) raw parameters to physical ranges.

    Standard mode:
      pKa            ∈ [3, 10]       via 3 + 7·σ(raw)
      phi            ∈ [0, 1]        via σ(raw)
      J              ∈ [0.5, J_max]  via 0.5 + (J_max−0.5)·σ(raw)
      monomer_entropy∈ [0, S_max]    via S_max·σ(raw)
      eps_barrier    ∈ [0, eps_max]  via eps_max·σ(raw)

    Chain position mode (chain_mode=True):
      x              ∈ [0, 1]^{2N}   via σ(raw['x'])  — residue positions
      J is still trained; phi/s_i/eps_barrier are all replaced by chain-derived
      matrices computed from x via Jacobson–Stockmayer and Wilemski–Fixman theory.
      chain_static must contain the chain hyperparameters (L_chain, lambda_c, …).
    """
    out = {
        'pKa': (jnp.array(fixed_pKa, dtype=float) if fixed_pKa is not None
                else 3.0 + 7.0 * jax.nn.sigmoid(raw['pKa'])),
        'J':   (jnp.array(float(fixed_J)) if fixed_J is not None
                else 0.5 + (J_max - 0.5) * jax.nn.sigmoid(raw['J'])),
    }

    if chain_mode and chain_static is not None:
        # Positions in [0, 1]; derive phi/entropy/k0 matrices from chain geometry.
        x = jax.nn.sigmoid(raw['x'])
        out['x'] = x
        phi_m, ent_m, k0_m = _compute_chain_quantities(
            x, chain_static['correct_mask'],
            L_chain   = chain_static['L_chain'],
            lambda_c  = chain_static['lambda_c'],
            l0        = chain_static['l0'],
            d0        = chain_static['d0'],
            k0_base   = chain_static['k0'],
            phi0      = chain_static.get('phi0', 1.0),
            alpha     = chain_static.get('chain_alpha', 1.5),
            eps_d     = chain_static.get('chain_eps_d', 0.5),
        )
        out['phi_matrix']   = phi_m
        out['pair_entropy'] = ent_m
        out['k0_matrix']    = k0_m
        # Scalar phi for reporting/visualisation (mean non-native selectivity).
        out['phi'] = jnp.mean(phi_m * ~chain_static['correct_mask'])
    else:
        out['phi'] = (jnp.array(float(fixed_phi)) if fixed_phi is not None
                      else jax.nn.sigmoid(raw['phi']))
        if S_max > 0.0 and 'monomer_entropy' in raw:
            out['monomer_entropy'] = S_max * jax.nn.sigmoid(raw['monomer_entropy'])
        if eps_max > 0.0 and 'eps_barrier' in raw:
            out['eps_barrier'] = eps_max * jax.nn.sigmoid(raw['eps_barrier'])

    return out


def unconstrain_params(phys, J_max=3.5, S_max=0.0, eps_max=0.0,
                       chain_mode=False):
    """Inverse of constrain_params for warm-starting."""
    def _logit(x):
        x = jnp.clip(jnp.array(x), 1e-4, 1 - 1e-4)
        return jnp.log(x / (1 - x))

    pKa_norm = jnp.clip((jnp.array(phys['pKa']) - 3.0) / 7.0, 1e-4, 1 - 1e-4)
    J_norm   = jnp.clip((jnp.array(phys['J']) - 0.5) / (J_max - 0.5), 1e-4, 1 - 1e-4)
    out = {
        'pKa': jnp.log(pKa_norm / (1 - pKa_norm)),
        'J':   jnp.log(J_norm   / (1 - J_norm)),
    }
    if chain_mode:
        if 'x' in phys:
            out['x'] = _logit(jnp.array(phys['x']))
        # phi/entropy/eps_barrier not in raw params for chain mode
    else:
        out['phi'] = _logit(phys['phi'])
        if S_max > 0.0 and 'monomer_entropy' in phys:
            s_norm = jnp.clip(jnp.array(phys['monomer_entropy']) / S_max, 1e-4, 1 - 1e-4)
            out['monomer_entropy'] = jnp.log(s_norm / (1 - s_norm))
        if eps_max > 0.0 and 'eps_barrier' in phys:
            eb_norm = jnp.clip(jnp.array(phys['eps_barrier']) / eps_max, 1e-4, 1 - 1e-4)
            out['eps_barrier'] = jnp.log(eb_norm / (1 - eb_norm))
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
# Parameter serialisation for population-based optimisers (evosax)
# ---------------------------------------------------------------------------

def _flatten_params(raw_params):
    """Flatten raw_params pytree dict → 1D JAX array + metadata."""
    leaves, treedef = jax.tree_util.tree_flatten(raw_params)
    shapes = [jnp.shape(l) for l in leaves]
    sizes  = [int(np.prod(s)) if s else 1 for s in shapes]
    flat   = jnp.concatenate([jnp.atleast_1d(jnp.asarray(l)) for l in leaves])
    return flat, treedef, shapes, sizes


def _unflatten_params(flat, treedef, shapes, sizes):
    """Restore 1D JAX array → raw_params pytree dict."""
    leaves = []
    idx = 0
    for shape, size in zip(shapes, sizes):
        leaves.append(flat[idx : idx + size].reshape(shape))
        idx += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def correct_bond_score(state, n, correct_triu_idx):
    """Fraction of classifier pairs that have formed correct dimers.

    Normalized so 1.0 means all M classifier pairs are fully dimerized,
    regardless of how many roughness species are present (M/N ratio).
    Each classifier acid/base particle starts at concentration total/(2N),
    so the max achievable correct dimer sum is M * total/(2N).
    """
    free    = state[:n]
    dimers  = state[n:]
    total   = jnp.sum(free) + 2.0 * jnp.sum(dimers)
    correct = jnp.sum(dimers[correct_triu_idx])
    n_clf   = correct_triu_idx.shape[0]       # M, number of classifier pairs
    max_possible = float(n_clf) / float(n) * total   # = M/(2N) * total
    return correct / (max_possible + 1e-10)


def integral_bond_score(sched_trajs, post_traj, sched_dur_total, post_dur,
                        n, correct_triu_idx):
    """Time-averaged correct-dimer fraction over schedule + post phases.

    sched_trajs     : (n_segs, n_pts_sched, state_size) — schedule trajectories
    post_traj       : (n_pts_post, state_size)           — post-duration trajectory
    sched_dur_total : total duration of all schedule segments (Python float)
    post_dur        : duration of post phase (Python float)

    The score is the duration-weighted mean of Σcorrect_dimers, normalised by
    the same max-possible factor as correct_bond_score (M/2N assuming total=1).
    """
    sched_correct = jnp.sum(
        sched_trajs[:, :, n:][:, :, correct_triu_idx], axis=-1)   # (n_segs, n_pts)
    sched_avg = jnp.mean(sched_correct)

    post_correct = jnp.sum(
        post_traj[:, n:][:, correct_triu_idx], axis=-1)            # (n_pts_post,)
    post_avg = jnp.mean(post_correct)

    total_dur    = sched_dur_total + post_dur
    weighted_avg = (sched_avg * sched_dur_total + post_avg * post_dur) / total_dur

    n_clf        = correct_triu_idx.shape[0]
    max_possible = float(n_clf) / float(n)
    return weighted_avg / (max_possible + 1e-10)


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
    _chain_mode  = bool(static.get('chain_mode', False))
    p = constrain_params(raw_params,
                         J_max=static['J_max'],
                         S_max=static.get('S_max', 0.0),
                         eps_max=static.get('eps_barrier_max', 0.0),
                         fixed_phi=static.get('fixed_phi'),
                         fixed_J=static.get('fixed_J'),
                         fixed_pKa=static.get('fixed_pKa'),
                         chain_mode=_chain_mode,
                         chain_static=static if _chain_mode else None)

    # Resolve phi, k0, entropy, pair_entropy for this mode.
    if _chain_mode:
        _phi     = p['phi_matrix']    # (n, n)
        _k0      = p['k0_matrix']     # (n, n)
        _pe      = p['pair_entropy']  # (n, n)
        mono_s   = None
    else:
        _phi     = p['phi']           # scalar
        _k0      = static['k0']       # scalar
        _pe      = None
        mono_s   = _get_monomer_entropy(p)

    sw              = float(static.get('smooth_width', 0.0))
    allowed_mask    = static.get('allowed_mask', None)
    no_self_bonds   = bool(static.get('no_self_bonds', False))
    equil_ramp      = float(static.get('equil_ramp_duration', 0.0))
    grad_clip    = static.get('grad_clip', None)
    eps_b        = p.get('eps_barrier', jnp.array(0.0))
    pKa_full = p['pKa']

    if static.get('start_equil', False):
        equil_state = _boltzmann_equilibrium_jax(
            pKa_full, _phi, p['J'], 7.0,
            static['acid_base'], static['correct_mask'],
            static['beta'], static['n'],
            static['i_idx'], static['j_idx'],
            monomer_entropy=mono_s,
            allowed_mask=allowed_mask,
            no_self_bonds=no_self_bonds,
            pair_entropy=_pe,
        )
    else:
        equil_state = simulate_schedule_scan(
            initial_state, jnp.array([7.0]), static['equil_duration'],
            pKa_full, static['acid_base'], _phi, p['J'],
            static['beta'], _k0,
            static['correct_mask'], static['n'],
            static['i_idx'], static['j_idx'],
            n_points=static['n_points_equil'],
            smooth_width=sw,
            monomer_entropy=mono_s,
            ph_initial=7.0,
            allowed_mask=allowed_mask,
            beta_ramp_duration=equil_ramp,
            no_self_bonds=no_self_bonds,
            eps_barrier=eps_b,
            pair_entropy=_pe,
        )
        if grad_clip is not None:
            equil_state = _clip_grad_norm(float(grad_clip), equil_state)

    baseline_score = correct_bond_score(equil_state, static['n'],
                                        static['correct_triu_idx'])

    _post_dur    = float(static.get('post_duration', 0.0))
    _n_pts_post  = int(static.get('n_points_post', 20))
    _n_segs      = int(all_pH_schedules_array.shape[1])

    def score_one(pH_sched):
        if _post_dur > 0.0:
            final, all_trajs = simulate_schedule_scan(
                equil_state, pH_sched, duration_per_seg,
                pKa_full, static['acid_base'], _phi, p['J'],
                static['beta'], _k0,
                static['correct_mask'], static['n'],
                static['i_idx'], static['j_idx'],
                n_points=static['n_points_sim'],
                smooth_width=sw,
                monomer_entropy=mono_s,
                ph_initial=7.0,
                allowed_mask=allowed_mask,
                no_self_bonds=no_self_bonds,
                return_traj=True,
                eps_barrier=eps_b,
                pair_entropy=_pe,
            )
            if grad_clip is not None:
                final = _clip_grad_norm(float(grad_clip), final)
            _, post_trajs = simulate_schedule_scan(
                final, jnp.array([7.0]), _post_dur,
                pKa_full, static['acid_base'], _phi, p['J'],
                static['beta'], _k0,
                static['correct_mask'], static['n'],
                static['i_idx'], static['j_idx'],
                n_points=_n_pts_post,
                smooth_width=sw,
                monomer_entropy=mono_s,
                ph_initial=None,
                allowed_mask=allowed_mask,
                no_self_bonds=no_self_bonds,
                return_traj=True,
                eps_barrier=eps_b,
                pair_entropy=_pe,
            )
            post_traj = post_trajs[0]
            return integral_bond_score(
                all_trajs, post_traj,
                float(_n_segs) * duration_per_seg, _post_dur,
                static['n'], static['correct_triu_idx'],
            )
        else:
            final = simulate_schedule_scan(
                equil_state, pH_sched, duration_per_seg,
                pKa_full, static['acid_base'], _phi, p['J'],
                static['beta'], _k0,
                static['correct_mask'], static['n'],
                static['i_idx'], static['j_idx'],
                n_points=static['n_points_sim'],
                smooth_width=sw,
                monomer_entropy=mono_s,
                ph_initial=7.0,
                allowed_mask=allowed_mask,
                no_self_bonds=no_self_bonds,
                eps_barrier=eps_b,
                pair_entropy=_pe,
            )
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
    _chain_mode   = bool(static.get('chain_mode', False))
    all_pH_array  = jnp.array(all_schedules, dtype=float)
    sw            = float(static.get('smooth_width', 0.0))
    allowed_mask  = static.get('allowed_mask', None)
    no_self_bonds = bool(static.get('no_self_bonds', False))
    pKa           = jnp.array(p_constrained['pKa'])
    J             = jnp.array(p_constrained['J'])
    equil_ramp    = float(static.get('equil_ramp_duration', 0.0))
    eps_b_fast    = float(p_constrained.get('eps_barrier', 0.0) or 0.0)

    if _chain_mode:
        _phi_f   = jnp.array(p_constrained['phi_matrix'])
        _k0_f    = jnp.array(p_constrained['k0_matrix'])
        _pe_f    = jnp.array(p_constrained['pair_entropy'])
        mono_s   = None
    else:
        _phi_f   = jnp.array(p_constrained['phi'])
        _k0_f    = static['k0']
        _pe_f    = None
        mono_s   = (_get_monomer_entropy(p_constrained)
                    if p_constrained.get('monomer_entropy') is not None else None)

    _start_equil  = bool(static.get('start_equil', False))
    _dissoc_state = make_initial_state(static['n'])
    _post_dur_f   = float(static.get('post_duration', 0.0))
    _n_pts_post_f = int(static.get('n_points_post', 20))
    _n_segs_f     = int(all_pH_array.shape[1])

    @jax.jit
    def _score_all(pKa, _phi, J, _k0, all_pH_array):
        if _start_equil:
            equil = _boltzmann_equilibrium_jax(
                pKa, _phi, J, 7.0,
                static['acid_base'], static['correct_mask'],
                static['beta'], static['n'],
                static['i_idx'], static['j_idx'],
                monomer_entropy=mono_s,
                allowed_mask=allowed_mask,
                no_self_bonds=no_self_bonds,
                pair_entropy=_pe_f,
            )
        else:
            equil = simulate_schedule_scan(
                _dissoc_state, jnp.array([7.0]), static['equil_duration'],
                pKa, static['acid_base'], _phi, J,
                static['beta'], _k0,
                static['correct_mask'], static['n'],
                static['i_idx'], static['j_idx'],
                n_points=static['n_points_equil'],
                smooth_width=sw,
                monomer_entropy=mono_s,
                ph_initial=7.0,
                allowed_mask=allowed_mask,
                beta_ramp_duration=equil_ramp,
                no_self_bonds=no_self_bonds,
                eps_barrier=eps_b_fast,
                pair_entropy=_pe_f,
            )

        baseline = correct_bond_score(equil, static['n'],
                                      static['correct_triu_idx'])

        def score_one(pH_sched):
            if _post_dur_f > 0.0:
                final, all_trajs = simulate_schedule_scan(
                    equil, pH_sched, duration_per_seg,
                    pKa, static['acid_base'], _phi, J,
                    static['beta'], _k0,
                    static['correct_mask'], static['n'],
                    static['i_idx'], static['j_idx'],
                    n_points=static['n_points_sim'],
                    smooth_width=sw,
                    monomer_entropy=mono_s,
                    allowed_mask=allowed_mask,
                    no_self_bonds=no_self_bonds,
                    return_traj=True,
                    eps_barrier=eps_b_fast,
                    pair_entropy=_pe_f,
                )
                _, post_trajs = simulate_schedule_scan(
                    final, jnp.array([7.0]), _post_dur_f,
                    pKa, static['acid_base'], _phi, J,
                    static['beta'], _k0,
                    static['correct_mask'], static['n'],
                    static['i_idx'], static['j_idx'],
                    n_points=_n_pts_post_f,
                    smooth_width=sw,
                    monomer_entropy=mono_s,
                    ph_initial=None,
                    allowed_mask=allowed_mask,
                    no_self_bonds=no_self_bonds,
                    return_traj=True,
                    eps_barrier=eps_b_fast,
                    pair_entropy=_pe_f,
                )
                post_traj = post_trajs[0]
                return integral_bond_score(
                    all_trajs, post_traj,
                    float(_n_segs_f) * duration_per_seg, _post_dur_f,
                    static['n'], static['correct_triu_idx'],
                )
            else:
                final = simulate_schedule_scan(
                    equil, pH_sched, duration_per_seg,
                    pKa, static['acid_base'], _phi, J,
                    static['beta'], _k0,
                    static['correct_mask'], static['n'],
                    static['i_idx'], static['j_idx'],
                    n_points=static['n_points_sim'],
                    smooth_width=sw,
                    monomer_entropy=mono_s,
                    allowed_mask=allowed_mask,
                    no_self_bonds=no_self_bonds,
                    eps_barrier=eps_b_fast,
                    pair_entropy=_pe_f,
                )
                return correct_bond_score(final, static['n'], static['correct_triu_idx'])

        sched_scores = jax.vmap(score_one)(all_pH_array)
        return jnp.append(sched_scores, baseline)

    return np.array(_score_all(pKa, _phi_f, J, _k0_f, all_pH_array))


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
    eps_max        = float(config.get('eps_barrier_max', 0.0))
    smooth         = float(config.get('smooth_width', 0.0))
    per_mono       = bool(config.get('per_monomer_entropy', False))
    no_self_bonds  = bool(config.get('no_self_bonds', False))
    chain_mode     = bool(config.get('chain_mode', False))
    wide_init      = bool(config.get('wide_init', False))
    j_init_max     = bool(config.get('j_init_max', False))
    phi_init_max   = bool(config.get('phi_init_max', False))
    verbose        = bool(config.get('verbose', True))
    start_equil    = bool(config.get('start_equil', False))
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
        'eps_barrier_max'    : eps_max,
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
        'start_equil'        : start_equil,
        'post_duration'      : float(config.get('post_duration', 0.0)),
        'n_points_post'      : max(20, int(2 * float(config.get('post_duration', 0.0)))),
        # Chain position mode hyperparameters (ignored when chain_mode=False)
        'chain_mode'         : chain_mode,
        'L_chain'            : float(config.get('L_chain', 100.0)),
        'lambda_c'           : float(config.get('lambda_c', 20.0)),
        'l0'                 : float(config.get('l0', 3.0)),
        'd0'                 : float(config.get('d0', 3.0)),
        'phi0'               : float(config.get('phi0', 1.0)),
        'chain_alpha'        : float(config.get('chain_alpha', 1.5)),
        'chain_eps_d'        : float(config.get('chain_eps_d', 0.5)),
    }
    if start_equil:
        static['equil_duration'] = 0.0

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
        print(f"N_acids      : {N_acids}  ({M} classifier, {N_acids-M} competitor)")
        print(f"N_bases      : {N_acids}  ({M} classifier, {N_acids-M} competitor)")
        print(f"Particles    : {N}  ({N_acids} acids + {N_acids} bases)")
        print(f"Target sched : {target_sched}")
        print(f"Permutations : {len(all_schedules)}  (target idx = {target_idx})")
        if start_equil:
            print(f"Equilibration: Boltzmann (pH 7, ODE skipped, recomputed each epoch)")
        else:
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
        if eps_max > 0:
            print(f"ε_‡ max      : {eps_max} kT  (Arrhenius barrier, trainable)")

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
    init_phys_np = {k: np.array(v) for k, v in init_phys.items()}
    if chain_mode:
        # Evenly-spaced initial positions: acids in [0, 0.5], bases in [0.5, 1].
        x_acid_init = np.linspace(0.05, 0.45, N_acids)
        x_base_init = np.linspace(0.55, 0.95, N_acids)
        x_init = np.concatenate([x_acid_init, x_base_init])
        init_phys['x'] = jnp.array(x_init)
        init_phys_np['x'] = x_init
        if verbose:
            print(f"Chain mode   : L_chain={static['L_chain']:.0f} residues, "
                  f"λ_c={static['lambda_c']:.1f}, l0={static['l0']:.1f}, "
                  f"d0={static['d0']:.1f}, α={static['chain_alpha']:.2f}")
            print(f"  (φ, s_i, ε_‡ replaced by {N} trainable positions x_i)")
    elif S_max > 0.0:
        if wide_init:
            s_init = rng.uniform(0.01 * S_max, 0.5 * S_max, n_entropy)
        else:
            s_init = np.clip(
                rng.uniform(0.0, 0.2 * S_max, n_entropy),
                1e-4 * S_max, 0.999 * S_max,
            )
        init_phys['monomer_entropy'] = jnp.array(s_init)
        init_phys_np['monomer_entropy'] = np.array(s_init)
    if eps_max > 0.0 and not chain_mode:
        eps_init = eps_max * 0.05
        init_phys['eps_barrier'] = jnp.array(eps_init)
        init_phys_np['eps_barrier'] = float(eps_init)

    raw_params = unconstrain_params(init_phys, J_max=J_max, S_max=S_max,
                                    eps_max=eps_max, chain_mode=chain_mode)

    # The fully-dissociated state is passed when start_equil=False.
    # When start_equil=True, compute_loss calls boltzmann_equilibrium_jax
    # internally from raw_params, so initial_state is not used.
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

    p0 = constrain_params(raw_params, J_max=J_max, S_max=S_max, eps_max=eps_max,
                          fixed_phi=fixed_phi_val, fixed_J=fixed_J_val,
                          fixed_pKa=fixed_pKa_val, chain_mode=chain_mode,
                          chain_static=static if chain_mode else None)
    param_history = [_snapshot(p0, S_max, eps_max, chain_mode=chain_mode)]

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
                                  eps_max=eps_max, fixed_phi=fixed_phi_val,
                                  fixed_J=fixed_J_val, fixed_pKa=fixed_pKa_val,
                                  chain_mode=chain_mode,
                                  chain_static=static if chain_mode else None)
        param_history.append(_snapshot(p_cur, S_max, eps_max, chain_mode=chain_mode))

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
            eps_str    = ''
            if eps_max > 0.0 and 'eps_barrier' in p_cur:
                eps_str = f' | ε_‡={float(p_cur["eps_barrier"]):.3f}'
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
                f"J={J_str}{s_str}{eps_str}",
                flush=True,
            )

    if verbose:
        if nan_stopped:
            print("Training terminated early (NaN).")
        else:
            print("\nTraining complete.")

    p_final  = constrain_params(raw_params, J_max=J_max, S_max=S_max,
                               eps_max=eps_max, fixed_phi=fixed_phi_val,
                               fixed_J=fixed_J_val, fixed_pKa=fixed_pKa_val,
                               chain_mode=chain_mode,
                               chain_static=static if chain_mode else None)
    pKa_full = p_final['pKa']

    if chain_mode:
        _phi_f  = p_final['phi_matrix']
        _k0_f   = p_final['k0_matrix']
        _pe_f   = p_final['pair_entropy']
        mono_s  = None
    else:
        _phi_f  = p_final['phi']
        _k0_f   = static['k0']
        _pe_f   = None
        mono_s  = _get_monomer_entropy(p_final)

    # Compute final equilibrium state for return value (no gradients needed).
    equil_state = _boltzmann_equilibrium_jax(
        pKa_full, _phi_f, p_final['J'], 7.0,
        static['acid_base'], static['correct_mask'],
        static['beta'], N,
        static['i_idx'], static['j_idx'],
        monomer_entropy=mono_s,
        allowed_mask=allowed_mask_jax,
        no_self_bonds=no_self_bonds,
        pair_entropy=_pe_f,
    )
    if not (start_equil or chain_mode):
        # For visualisation accuracy, re-run the ODE equil from dissociated state.
        eps_b_final = float(p_final.get('eps_barrier', 0.0) or 0.0)
        equil_state = simulate_schedule_scan(
            initial_state, jnp.array([7.0]), static['equil_duration'],
            pKa_full, static['acid_base'], _phi_f, p_final['J'],
            static['beta'], _k0_f,
            static['correct_mask'], static['n'], static['i_idx'], static['j_idx'],
            n_points=static['n_points_equil'],
            smooth_width=smooth, monomer_entropy=mono_s, ph_initial=7.0,
            allowed_mask=allowed_mask_jax,
            beta_ramp_duration=static['equil_ramp_duration'],
            no_self_bonds=no_self_bonds,
            eps_barrier=eps_b_final,
            pair_entropy=_pe_f,
        )

    return (raw_params, loss_history, score_history, param_history,
            static, all_schedules, target_idx, equil_state,
            init_phys_np, nan_stopped)


def _snapshot(p, S_max, eps_max=0.0, chain_mode=False):
    snap = {'pKa': np.array(p['pKa']), 'phi': float(p['phi']), 'J': float(p['J'])}
    if chain_mode and 'x' in p:
        snap['x'] = np.array(p['x'])
    if S_max > 0.0 and 'monomer_entropy' in p:
        snap['monomer_entropy'] = np.array(p['monomer_entropy'])
    if eps_max > 0.0 and 'eps_barrier' in p:
        snap['eps_barrier'] = float(p['eps_barrier'])
    return snap


# ---------------------------------------------------------------------------
# CMA-ES + L-BFGS-B hybrid optimiser
# ---------------------------------------------------------------------------

def train_cmaes_hybrid(config):
    """
    Multi-start L-BFGS-B optimizer in physical parameter space.

    Stage 1 — Multi-start L-BFGS-B (scipy):
        Runs ``cmaes_epochs`` independent L-BFGS-B runs from Sobol
        quasi-random starting points in physical parameter space
        (pKa ∈ [3,10], φ ∈ [0,1], J ∈ [0.5,J_max]).  Box constraints
        are enforced by scipy; JAX autodiff provides exact gradients.
        Gradients near boundaries are amplified via the logit chain rule,
        pushing runs away from degenerate φ≈0 solutions.

    Stage 2 — Final L-BFGS-B polish:
        Starting from the Stage 1 best, runs longer with tighter
        ftol/gtol for high-precision convergence.

    Returns the same 10-tuple as train() for drop-in compatibility.
    Score_history is empty; param_history has one snapshot per restart
    plus a final entry (compatible with plot_summary).
    """
    import scipy.optimize as _scipy_opt
    from scipy.stats.qmc import Sobol as _Sobol

    # ------------------------------------------------------------------
    # Setup (mirrors train() lines 502–692; kept in sync manually)
    # ------------------------------------------------------------------
    N_acids = int(config['N_total'])
    M       = int(config['M_classifier'])
    assert N_acids >= 1 and M >= 1 and M <= N_acids
    N = 2 * N_acids

    J_max         = float(config.get('J_max', 3.5))
    S_max         = float(config.get('S_max', 0.0))
    eps_max       = float(config.get('eps_barrier_max', 0.0))
    smooth        = float(config.get('smooth_width', 0.0))
    per_mono      = bool(config.get('per_monomer_entropy', False))
    no_self_bonds = bool(config.get('no_self_bonds', False))
    wide_init     = bool(config.get('wide_init', False))
    j_init_max    = bool(config.get('j_init_max', False))
    phi_init_max  = bool(config.get('phi_init_max', False))
    verbose       = bool(config.get('verbose', True))
    start_equil   = bool(config.get('start_equil', False))
    fixed_phi_val = config.get('fixed_phi', None)
    if fixed_phi_val is not None:
        fixed_phi_val = float(np.clip(fixed_phi_val, 0.0, 1.0))
    fixed_J_val   = config.get('fixed_J', None)
    if fixed_J_val is not None:
        fixed_J_val = float(fixed_J_val)
    pka_default   = bool(config.get('pka_default', False))
    pKa_acid      = float(config.get('pKa_acid', 6.0))
    pKa_base      = float(config.get('pKa_base', 8.0))
    fixed_pKa_val = (np.array([pKa_acid]*N_acids + [pKa_base]*N_acids, dtype=float)
                     if pka_default else None)

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
    allowed_mask_jax = None

    static = {
        'n'                  : N,
        'n_species'          : N,
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
        'eps_barrier_max'    : eps_max,
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
        'start_equil'        : start_equil,
        'post_duration'      : float(config.get('post_duration', 0.0)),
        'n_points_post'      : max(20, int(2 * float(config.get('post_duration', 0.0)))),
    }
    if start_equil:
        static['equil_duration'] = 0.0

    target_sched  = [float(x) for x in config['target_pH_schedule']]
    all_schedules = all_unique_permutations(target_sched)
    target_idx    = all_schedules.index(target_sched)
    duration      = float(config['duration_per_seg'])
    all_pH_array  = jnp.array(all_schedules, dtype=float)

    n_entropy = N if per_mono else 1
    rng_np    = np.random.default_rng(int(config.get('seed', 42)))
    L         = len(target_sched)

    if wide_init:
        pKa_init = list(rng_np.uniform(3.1, 9.9, N))
        phi_init = float(rng_np.uniform(0.05, 0.95))
        J_init   = float(rng_np.uniform(0.55, J_max - 0.01))
    else:
        pKa_clf_acid = [float(np.clip(target_sched[k % L] - 1.5 + rng_np.normal(0.0, 0.5),
                                      3.1, 9.9)) for k in range(M)]
        pKa_rgh_acid = list(rng_np.uniform(3.1, 9.9, N_acids - M))
        pKa_clf_base = [float(np.clip(target_sched[k % L] + 1.5 + rng_np.normal(0.0, 0.5),
                                      3.1, 9.9)) for k in range(M)]
        pKa_rgh_base = list(rng_np.uniform(3.1, 9.9, N_acids - M))
        pKa_init = pKa_clf_acid + pKa_rgh_acid + pKa_clf_base + pKa_rgh_base
        phi_init = float(np.clip(0.2 + rng_np.normal(0.0, 0.05), 0.01, 0.99))
        J_init   = float(np.clip(1.5 + rng_np.normal(0.0, 0.2),  0.51, J_max - 0.01))

    if j_init_max:
        J_init = J_max * 0.9
    if phi_init_max:
        phi_init = 0.9

    init_phys = {
        'pKa': jnp.array(pKa_init),
        'phi': jnp.array(phi_init),
        'J'  : jnp.array(J_init),
    }
    init_phys_np = {k: np.array(v) for k, v in init_phys.items()}
    if S_max > 0.0:
        if wide_init:
            s_init = rng_np.uniform(0.01 * S_max, 0.5 * S_max, n_entropy)
        else:
            s_init = np.clip(
                rng_np.uniform(0.0, 0.2 * S_max, n_entropy),
                1e-4 * S_max, 0.999 * S_max,
            )
        init_phys['monomer_entropy'] = jnp.array(s_init)
        init_phys_np['monomer_entropy'] = np.array(s_init)
    if eps_max > 0.0:
        eps_init = eps_max * 0.05
        init_phys['eps_barrier'] = jnp.array(eps_init)
        init_phys_np['eps_barrier'] = float(eps_init)

    raw_params_init = unconstrain_params(init_phys, J_max=J_max, S_max=S_max,
                                         eps_max=eps_max)
    initial_state   = make_initial_state(N)

    loss_fn = partial(
        compute_loss,
        all_pH_schedules_array=all_pH_array,
        target_idx=target_idx,
        duration_per_seg=duration,
        static=static,
        initial_state=initial_state,
    )

    def scalar_loss_fn(raw_params):
        loss_val, _ = loss_fn(raw_params)
        return loss_val

    # ------------------------------------------------------------------
    # Physical-space loss and bounds
    # Working in constrained space [pKa∈[3,10], φ∈[0,1], J∈[0.5,J_max]]
    # avoids the logit-space distortion that causes CMA-ES to collapse to
    # degenerate boundary solutions.  Gradients near boundaries are
    # amplified via the logit chain rule (inside unconstrain_params),
    # pushing L-BFGS-B away from φ≈0 / boundary-pKa solutions.
    # ------------------------------------------------------------------
    _n_pka = 2 * N_acids
    _n_ent = n_entropy if S_max > 0.0 else 0
    _n_eps = 1 if eps_max > 0.0 else 0
    n_phys = _n_pka + 2 + _n_ent + _n_eps

    _lo_arr = np.concatenate([
        [3.01] * _n_pka,
        [0.001, 0.51],
        ([0.001 * S_max] * _n_ent if _n_ent > 0 else []),
        ([0.001 * eps_max]         if _n_eps > 0 else []),
    ])
    _hi_arr = np.concatenate([
        [9.99] * _n_pka,
        [0.999, J_max - 0.01],
        ([0.999 * S_max] * _n_ent if _n_ent > 0 else []),
        ([0.999 * eps_max]         if _n_eps > 0 else []),
    ])
    scipy_bounds = list(zip(_lo_arr.tolist(), _hi_arr.tolist()))

    @jax.jit
    def _phys_to_loss(phys_flat):
        phys = {
            'pKa': phys_flat[:_n_pka],
            'phi': phys_flat[_n_pka],
            'J':   phys_flat[_n_pka + 1],
        }
        if S_max > 0.0:
            phys['monomer_entropy'] = phys_flat[_n_pka + 2 : _n_pka + 2 + _n_ent]
        if eps_max > 0.0:
            phys['eps_barrier'] = phys_flat[_n_pka + 2 + _n_ent]
        raw = unconstrain_params(phys, J_max=J_max, S_max=S_max, eps_max=eps_max)
        loss_val, _ = loss_fn(raw)
        return loss_val

    _phys_val_and_grad = jax.jit(jax.value_and_grad(_phys_to_loss))

    # Build initial physical flat vector for warm-up compile
    _init_phys_list = (list(np.array(init_phys_np['pKa']))
                       + [float(init_phys_np['phi']), float(init_phys_np['J'])])
    if S_max > 0.0:
        _init_phys_list += list(np.array(init_phys_np.get('monomer_entropy', [])))
    if eps_max > 0.0:
        _init_phys_list += [float(init_phys_np.get('eps_barrier', eps_max * 0.05))]
    _init_phys_flat = np.array(_init_phys_list, dtype=np.float64)

    if verbose:
        print(f"Multi-start L-BFGS-B  |  n_params={n_phys}  "
              f"target={target_sched}  J_max={J_max}", flush=True)
        print("  Compiling physical-space value+grad ...", flush=True)
    _ = _phys_val_and_grad(jnp.array(_init_phys_flat))
    if verbose:
        print("  Compilation done.\n", flush=True)

    def _scipy_phys_fg(x_np):
        lv, gv = _phys_val_and_grad(jnp.array(x_np, dtype=float))
        return float(lv), np.array(gv, dtype=np.float64)

    # ------------------------------------------------------------------
    # Stage 1: Multi-start L-BFGS-B with Sobol quasi-random initialisation
    # cmaes_epochs is repurposed as the number of independent restarts.
    # ------------------------------------------------------------------
    n_restarts    = int(config.get('cmaes_epochs', 10))
    lbfgs_maxiter = int(config.get('lbfgs_epochs', 100))

    sampler    = _Sobol(d=n_phys, scramble=True,
                        seed=int(config.get('seed', 42)))
    sobol_unit = sampler.random(n=n_restarts)              # (n_restarts, n_phys) ∈ [0,1]
    sobol_phys = _lo_arr + sobol_unit * (_hi_arr - _lo_arr)  # scaled to physical bounds

    if verbose:
        n_scheds = len(all_schedules)
        print(f"Stage 1 — {n_restarts} L-BFGS-B restarts  |  "
              f"maxiter={lbfgs_maxiter}  n_schedules={n_scheds}", flush=True)

    best_loss      = float('inf')
    best_phys_flat = _init_phys_flat.copy()
    loss_history   = []
    param_history  = []

    def _phys_flat_to_snapshot(pf):
        phys_d = {
            'pKa': jnp.array(pf[:_n_pka]),
            'phi': jnp.array(float(pf[_n_pka])),
            'J':   jnp.array(float(pf[_n_pka + 1])),
        }
        if S_max > 0.0:
            phys_d['monomer_entropy'] = jnp.array(pf[_n_pka + 2 : _n_pka + 2 + _n_ent])
        if eps_max > 0.0:
            phys_d['eps_barrier'] = jnp.array(float(pf[_n_pka + 2 + _n_ent]))
        raw_d  = unconstrain_params(phys_d, J_max=J_max, S_max=S_max, eps_max=eps_max)
        phys_c = constrain_params(raw_d, J_max=J_max, S_max=S_max, eps_max=eps_max,
                                  fixed_phi=fixed_phi_val, fixed_J=fixed_J_val,
                                  fixed_pKa=fixed_pKa_val)
        return _snapshot(phys_c, S_max, eps_max)

    for restart_i in range(n_restarts):
        x0  = sobol_phys[restart_i]
        res = _scipy_opt.minimize(
            _scipy_phys_fg, x0=x0,
            method='L-BFGS-B', jac=True, bounds=scipy_bounds,
            options={'maxiter': lbfgs_maxiter, 'disp': False,
                     'ftol': 1e-9, 'gtol': 1e-6},
        )
        r_loss = float(res.fun) if np.isfinite(res.fun) else float('inf')

        if r_loss < best_loss:
            best_loss      = r_loss
            best_phys_flat = np.array(res.x, dtype=np.float64)

        loss_history.append(best_loss)
        param_history.append(_phys_flat_to_snapshot(best_phys_flat))

        if verbose:
            print(f"  Restart {restart_i + 1:3d}/{n_restarts}  "
                  f"this={r_loss:.4f}  best={best_loss:.4f}", flush=True)

    if verbose:
        print(f"\nStage 1 complete.  Best loss: {best_loss:.4f}\n", flush=True)

    # ------------------------------------------------------------------
    # Stage 2: Final polish from Stage 1 best (tighter tolerances)
    # ------------------------------------------------------------------
    polish_maxiter = max(lbfgs_maxiter, 200)
    if verbose:
        print(f"Stage 2 — Final polish  |  maxiter={polish_maxiter}", flush=True)

    res2        = _scipy_opt.minimize(
        _scipy_phys_fg, x0=best_phys_flat,
        method='L-BFGS-B', jac=True, bounds=scipy_bounds,
        options={'maxiter': polish_maxiter, 'disp': False,
                 'ftol': 1e-12, 'gtol': 1e-8},
    )
    polish_loss = float(res2.fun) if np.isfinite(res2.fun) else float('inf')

    if polish_loss < best_loss:
        best_phys_flat = np.array(res2.x, dtype=np.float64)
        best_loss      = polish_loss
    elif verbose:
        print(f"  Polish did not improve ({polish_loss:.4f} vs "
              f"{best_loss:.4f}); keeping Stage 1 best.", flush=True)

    loss_history.append(best_loss)
    param_history.append(_phys_flat_to_snapshot(best_phys_flat))

    # Convert best physical params → raw params for equil + return
    best_phys_d = {
        'pKa': jnp.array(best_phys_flat[:_n_pka]),
        'phi': jnp.array(float(best_phys_flat[_n_pka])),
        'J':   jnp.array(float(best_phys_flat[_n_pka + 1])),
    }
    if S_max > 0.0:
        best_phys_d['monomer_entropy'] = jnp.array(
            best_phys_flat[_n_pka + 2 : _n_pka + 2 + _n_ent])
    if eps_max > 0.0:
        best_phys_d['eps_barrier'] = jnp.array(
            float(best_phys_flat[_n_pka + 2 + _n_ent]))
    raw_params = unconstrain_params(best_phys_d, J_max=J_max, S_max=S_max,
                                    eps_max=eps_max)

    if verbose:
        p_final_v = constrain_params(raw_params, J_max=J_max, S_max=S_max,
                                     eps_max=eps_max, fixed_phi=fixed_phi_val,
                                     fixed_J=fixed_J_val, fixed_pKa=fixed_pKa_val)
        pKa_arr  = [float(v) for v in p_final_v['pKa']]
        acid_str = ' '.join(f'{v:.2f}' for v in pKa_arr[:N_acids])
        base_str = ' '.join(f'{v:.2f}' for v in pKa_arr[N_acids:])
        print(f"Stage 2 complete.  Final loss: {best_loss:.4f}")
        print(f"  pKa acids=[{acid_str}]")
        print(f"  pKa bases=[{base_str}]")
        print(f"  φ={float(p_final_v['phi']):.3f}  J={float(p_final_v['J']):.3f}\n",
              flush=True)

    # ------------------------------------------------------------------
    # Compute equilibrium state for visualisation (same as end of train())
    # ------------------------------------------------------------------
    p_final  = constrain_params(raw_params, J_max=J_max, S_max=S_max,
                                eps_max=eps_max, fixed_phi=fixed_phi_val,
                                fixed_J=fixed_J_val, fixed_pKa=fixed_pKa_val)
    mono_s   = _get_monomer_entropy(p_final)
    pKa_full = p_final['pKa']
    if start_equil:
        equil_state = jnp.array(_boltzmann_initial_state(
            7.0, np.array(pKa_full),
            static['acid_base_np'], static['correct_mask_np'],
            float(p_final['phi']), np.array(p_final['J']),
            static['beta'], N,
            static['i_idx'], static['j_idx'],
            monomer_entropy_np=(np.array(mono_s) if mono_s is not None else None),
            no_self_bonds=no_self_bonds,
        ))
    else:
        eps_b_hybrid = float(p_final.get('eps_barrier', 0.0) or 0.0)
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
            eps_barrier=eps_b_hybrid,
        )

    return (raw_params, loss_history, [], param_history,
            static, all_schedules, target_idx, equil_state,
            init_phys_np, False)
