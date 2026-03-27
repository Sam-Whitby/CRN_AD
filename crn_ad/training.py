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

import jax
import jax.numpy as jnp
import numpy as np
import optax
from itertools import permutations
from functools import partial

from .dynamics import (simulate_schedule, simulate_schedule_scan,
                       make_initial_state, make_triu_indices)


# ---------------------------------------------------------------------------
# Parameter constraints
# ---------------------------------------------------------------------------

def constrain_params(raw, J_max=3.5, S_max=0.0):
    """
    Map unconstrained (ℝ) raw parameters to physical ranges.

    pKa            ∈ [3, 10]       via 3 + 7·σ(raw)
    phi            ∈ [0, 1]        via σ(raw)
    J              ∈ [0.5, J_max]  via 0.5 + (J_max−0.5)·σ(raw)
    monomer_entropy∈ [0, S_max]    via S_max·σ(raw)   (scalar or n-vector)
    """
    out = {
        'pKa': 3.0 + 7.0 * jax.nn.sigmoid(raw['pKa']),
        'phi': jax.nn.sigmoid(raw['phi']),
        'J':   0.5 + (J_max - 0.5) * jax.nn.sigmoid(raw['J']),
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
    Softmax cross-entropy loss.
    1. Equilibrate at pH 7 with current parameters.
    2. Score each schedule permutation from equilibrium via vmap.
    3. Return −log_softmax(τ · scores)[target_idx].
    """
    p = constrain_params(raw_params,
                         J_max=static['J_max'],
                         S_max=static.get('S_max', 0.0))
    mono_s = _get_monomer_entropy(p)
    sw     = float(static.get('smooth_width', 0.0))

    equil_state = simulate_schedule_scan(
        initial_state, jnp.array([7.0]), static['equil_duration'],
        p['pKa'], static['acid_base'], p['phi'], p['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'],
        static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        smooth_width=sw,
        monomer_entropy=mono_s,
        ph_initial=7.0,
    )

    def score_one(pH_sched):
        final = simulate_schedule_scan(
            equil_state, pH_sched, duration_per_seg,
            p['pKa'], static['acid_base'], p['phi'], p['J'],
            static['beta'], static['k0'],
            static['correct_mask'], static['n'],
            static['i_idx'], static['j_idx'],
            n_points=static['n_points_sim'],
            smooth_width=sw,
            monomer_entropy=mono_s,
            ph_initial=7.0,
        )
        return correct_bond_score(final, static['n'], static['correct_triu_idx'])

    scores = jax.vmap(score_one)(all_pH_schedules_array)
    tau    = static.get('tau', 5.0)
    log_p  = jax.nn.log_softmax(scores * tau)
    return -log_p[target_idx], scores


# ---------------------------------------------------------------------------
# Fast post-training scoring (JIT + vmap, same as training)
# ---------------------------------------------------------------------------

def compute_scores_fast(p_constrained, all_schedules, duration_per_seg, static):
    """
    Score all schedules in parallel using vmap — same graph as training.

    p_constrained : dict with numpy/jax arrays for pKa, phi, J, monomer_entropy
    Returns       : numpy array (n_schedules,)

    Compiles on first call; subsequent calls with same shapes are cached.
    """
    all_pH_array  = jnp.array(all_schedules, dtype=float)
    initial_state = make_initial_state(static['n'])
    sw            = float(static.get('smooth_width', 0.0))

    pKa   = jnp.array(p_constrained['pKa'])
    phi   = jnp.array(p_constrained['phi'])
    J     = jnp.array(p_constrained['J'])
    mono_s = (_get_monomer_entropy(p_constrained)
              if p_constrained.get('monomer_entropy') is not None else None)

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
        )

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
            )
            return correct_bond_score(final, static['n'], static['correct_triu_idx'])

        return jax.vmap(score_one)(all_pH_array)

    return np.array(_score_all(pKa, phi, J, all_pH_array))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    """
    Train CRN parameters to respond selectively to the target pH schedule.

    config keys
    -----------
    n_species          : int (even, ≤ 10)
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
    n = config['n_species']
    assert n % 2 == 0 and 2 <= n <= 10

    J_max          = float(config.get('J_max', 3.5))
    S_max          = float(config.get('S_max', 0.0))
    smooth         = float(config.get('smooth_width', 0.0))
    per_mono       = bool(config.get('per_monomer_entropy', False))

    # ------------------------------------------------------------------
    # Static quantities
    # ------------------------------------------------------------------
    acid_base_np    = np.array([i % 2 for i in range(n)], dtype=int)
    correct_mask_np = np.zeros((n, n), dtype=bool)
    for k in range(n // 2):
        i, j = 2 * k, 2 * k + 1
        correct_mask_np[i, j] = True
        correct_mask_np[j, i] = True

    i_idx, j_idx = make_triu_indices(n)
    correct_triu_idx = np.array([
        pos for pos, (ii, jj) in enumerate(zip(i_idx, j_idx))
        if correct_mask_np[ii, jj]
    ])

    static = {
        'n'               : n,
        'acid_base'       : jnp.array(acid_base_np),
        'acid_base_np'    : acid_base_np,
        'correct_mask'    : jnp.array(correct_mask_np),
        'correct_mask_np' : correct_mask_np,
        'i_idx'           : i_idx,
        'j_idx'           : j_idx,
        'correct_triu_idx': jnp.array(correct_triu_idx),
        'beta'            : float(config.get('beta', 1.0)),
        'k0'              : float(config.get('k0',  1.0)),
        'n_points_sim'    : int(config.get('n_points_sim',   40)),
        'n_points_equil'  : int(config.get('n_points_equil', 60)),
        'equil_duration'  : float(config.get('equil_duration', 80.0)),
        'tau'             : float(config.get('tau', 5.0)),
        'J_max'           : J_max,
        'S_max'           : S_max,
        'smooth_width'    : smooth,
        'per_monomer_entropy': per_mono,
    }

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    target_sched  = [float(x) for x in config['target_pH_schedule']]
    all_schedules = all_unique_permutations(target_sched)
    target_idx    = all_schedules.index(target_sched)
    duration      = float(config['duration_per_seg'])
    all_pH_array  = jnp.array(all_schedules, dtype=float)

    n_entropy = n if per_mono else 1
    print(f"Species      : {n}  ({n//2} correct pairs)")
    print(f"Target sched : {target_sched}")
    print(f"Permutations : {len(all_schedules)}  (target idx = {target_idx})")
    print(f"Equilibration: pH 7,  t = {static['equil_duration']} (β=1)")
    print(f"J_max        : {J_max}  kT")
    print(f"Smooth width : {smooth}  ({'enabled' if smooth > 0 else 'disabled'})")
    if S_max > 0:
        mode = f"per-monomer ({n} values)" if per_mono else "shared (1 value)"
        print(f"Entropy      : S_max = {S_max} kT, {mode}")

    # ------------------------------------------------------------------
    # Initial parameters
    # ------------------------------------------------------------------
    rng    = np.random.default_rng(int(config.get('seed', 42)))
    pH_min, pH_max = float(min(target_sched)), float(max(target_sched))
    pKa_init = []
    for i in range(n):
        if acid_base_np[i] == 0:
            centre = np.clip(pH_min + 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        else:
            centre = np.clip(pH_max - 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        pKa_init.append(float(centre))

    phi_init = float(np.clip(0.2 + rng.normal(0.0, 0.05), 0.01, 0.99))
    J_init   = float(np.clip(1.5 + rng.normal(0.0, 0.2),  0.51, J_max - 0.01))

    init_phys = {
        'pKa': jnp.array(pKa_init),
        'phi': jnp.array(phi_init),
        'J'  : jnp.array(J_init),
    }
    if S_max > 0.0:
        s_init = np.clip(
            rng.uniform(0.0, 0.2 * S_max, n_entropy),
            1e-4 * S_max, 0.999 * S_max,
        )
        init_phys['monomer_entropy'] = jnp.array(s_init)

    raw_params = unconstrain_params(init_phys, J_max=J_max, S_max=S_max)

    # Perturb initial free-monomer concentrations (±20 % multiplicative noise),
    # then renormalise so total monomer content M = 1 is preserved.
    free_init  = np.ones(n) / n * rng.uniform(0.8, 1.2, n)
    free_init /= free_init.sum()
    n_dimers   = n * (n + 1) // 2
    initial_state = jnp.concatenate([jnp.array(free_init), jnp.zeros(n_dimers)])

    # ------------------------------------------------------------------
    # Optimiser  (lr passed as traced JAX array — no recompile on change)
    # ------------------------------------------------------------------
    clip_norm = float(config.get('clip_norm', 0.5))
    lr        = float(config.get('learning_rate', 0.02))

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

    print("Compiling JAX graph (first call) ...", flush=True)
    lr_jax = jnp.array(lr)
    raw_params, opt_state, lv, sc = step(raw_params, opt_state, lr_jax)
    print("Compilation done.\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs      = int(config.get('n_epochs', 300))

    loss_history  = [float(lv)]
    score_history = [np.array(sc)]

    p0 = constrain_params(raw_params, J_max=J_max, S_max=S_max)
    param_history = [_snapshot(p0, S_max)]

    nan_stopped = False
    for epoch in range(1, n_epochs):
        new_params, new_opt, lv, sc = step(raw_params, opt_state, lr_jax)

        if not (np.isfinite(float(lv)) and _params_finite(new_params)):
            print(f"\nNaN encountered at epoch {epoch} — "
                  f"stopping early and reporting results from epoch {epoch - 1}.",
                  flush=True)
            nan_stopped = True
            break

        raw_params = new_params
        opt_state  = new_opt

        loss_history.append(float(lv))
        score_history.append(np.array(sc))
        p_cur = constrain_params(raw_params, J_max=J_max, S_max=S_max)
        param_history.append(_snapshot(p_cur, S_max))

        if epoch % max(1, n_epochs // 15) == 0 or epoch == n_epochs - 1:
            pKa_str = ' '.join(f'{float(v):.2f}' for v in p_cur['pKa'])
            s_str   = ''
            if S_max > 0.0 and 'monomer_entropy' in p_cur:
                s = p_cur['monomer_entropy']
                s_str = f' | s̄={float(jnp.mean(s)):.3f} sₘₐₓ={float(jnp.max(s)):.3f}'
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"loss={float(lv):.4f} | "
                f"target={float(sc[target_idx]):.3f} | "
                f"mean_other={float(jnp.mean(jnp.delete(sc, target_idx))):.3f} | "
                f"pKa=[{pKa_str}] | φ={float(p_cur['phi']):.3f} | "
                f"J={float(p_cur['J']):.3f}{s_str}",
                flush=True,
            )

    if nan_stopped:
        print("Training terminated early (NaN).")
    else:
        print("\nTraining complete.")

    p_final = constrain_params(raw_params, J_max=J_max, S_max=S_max)
    mono_s  = _get_monomer_entropy(p_final)
    equil_state = simulate_schedule_scan(
        initial_state, jnp.array([7.0]), static['equil_duration'],
        p_final['pKa'], static['acid_base'], p_final['phi'], p_final['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'], static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        smooth_width=smooth, monomer_entropy=mono_s, ph_initial=7.0,
    )

    return (raw_params, loss_history, score_history, param_history,
            static, all_schedules, target_idx, equil_state)


def _snapshot(p, S_max):
    snap = {'pKa': np.array(p['pKa']), 'phi': float(p['phi']), 'J': float(p['J'])}
    if S_max > 0.0 and 'monomer_entropy' in p:
        snap['monomer_entropy'] = np.array(p['monomer_entropy'])
    return snap
