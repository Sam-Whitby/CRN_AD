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
      = KL(δ_target ‖ softmax(τ · scores))

NaN stability strategy
-----------------------
Several mechanisms work together to prevent and recover from NaN:

1. J hard cap via sigmoid: J ∈ [0.5, J_max] prevents ODE stiffness.
2. Smooth pH transitions: eliminates RHS discontinuities at segment
   boundaries that trigger tiny adjoint steps.
3. Gradient NaN zeroing: any NaN/Inf gradient components are zeroed
   before the Adam update (JAX-native, inside the JIT'd step).
4. Gradient norm clipping: global-norm clip applied before Adam.
5. Adaptive learning rate: lr is passed as a traced argument so it
   can be halved at runtime without recompiling.  On NaN detection the
   Python loop reverts to the last *finite* params, resets Adam
   momentum (stale moments contain bad curvature info), and halves lr.
6. Finite-param guard: best_params is only updated when BOTH the loss
   AND all parameter arrays are finite (fixes the silent NaN capture
   bug where lv was finite but params had already gone NaN).
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

    pKa     ∈ [3, 10]       via  3 + 7·σ(raw_pKa)
    phi     ∈ [0, 1]        via  σ(raw_phi)
    J       ∈ [0.5, J_max]  via  0.5 + (J_max−0.5)·σ(raw_J)
    entropy ∈ [0, S_max]    via  S_max·σ(raw_entropy)   (if S_max > 0)
    """
    out = {
        'pKa': 3.0 + 7.0 * jax.nn.sigmoid(raw['pKa']),
        'phi': jax.nn.sigmoid(raw['phi']),
        'J':   0.5 + (J_max - 0.5) * jax.nn.sigmoid(raw['J']),
    }
    if S_max > 0.0:
        out['entropy'] = S_max * jax.nn.sigmoid(raw['entropy'])
    return out


def unconstrain_params(phys, J_max=3.5, S_max=0.0):
    """Inverse of constrain_params for warm-starting."""
    pKa_norm = jnp.clip((jnp.array(phys['pKa']) - 3.0) / 7.0, 1e-4, 1 - 1e-4)
    phi_val  = jnp.clip(jnp.array(phys['phi']), 1e-4, 1 - 1e-4)
    J_norm   = jnp.clip((jnp.array(phys['J']) - 0.5) / (J_max - 0.5), 1e-4, 1 - 1e-4)
    out = {
        'pKa': jnp.log(pKa_norm / (1.0 - pKa_norm)),
        'phi': jnp.log(phi_val  / (1.0 - phi_val)),
        'J':   jnp.log(J_norm   / (1.0 - J_norm)),
    }
    if S_max > 0.0 and 'entropy' in phys:
        s_norm = jnp.clip(jnp.array(phys['entropy']) / S_max, 1e-4, 1 - 1e-4)
        out['entropy'] = jnp.log(s_norm / (1.0 - s_norm))
    return out


def _params_finite(params):
    """True iff every leaf of the params pytree is finite."""
    leaves = jax.tree_util.tree_leaves(params)
    return all(bool(jnp.all(jnp.isfinite(v))) for v in leaves)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def correct_bond_score(state, n, correct_triu_idx):
    """
    Fraction of total monomer content residing in correct dimers.

        score = 2·Σ_{correct} [XᵢXⱼ] / (Σᵢ [Xᵢ] + 2·Σ_{i≤j} [XᵢXⱼ])
    """
    free       = state[:n]
    dimer_triu = state[n:]
    total      = jnp.sum(free) + 2.0 * jnp.sum(dimer_triu)
    correct    = jnp.sum(dimer_triu[correct_triu_idx])
    return 2.0 * correct / (total + 1e-10)


def total_monomer_content(state, n):
    """M(t) = Σᵢ[Xᵢ] + 2·Σ_{i≤j}[XᵢXⱼ] — should equal 1 throughout."""
    return jnp.sum(state[:n]) + 2.0 * jnp.sum(state[n:])


# ---------------------------------------------------------------------------
# Schedule generation
# ---------------------------------------------------------------------------

def all_unique_permutations(seq):
    """All unique orderings of the elements in seq."""
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
    2. Run each schedule permutation from equilibrium.
    3. Return −log_softmax(τ · scores)[target_idx].
    """
    p = constrain_params(raw_params,
                         J_max=static['J_max'],
                         S_max=static.get('S_max', 0.0))

    entropy_triu = p.get('entropy', None)

    equil_state = simulate_schedule_scan(
        initial_state,
        jnp.array([7.0]),
        static['equil_duration'],
        p['pKa'], static['acid_base'], p['phi'], p['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'],
        static['i_idx'], static['j_idx'],
        n_points    = static['n_points_equil'],
        smooth_width= static.get('smooth_width', 0.0),
        entropy_triu= entropy_triu,
    )

    def score_one_schedule(pH_sched):
        final = simulate_schedule_scan(
            equil_state,
            pH_sched,
            duration_per_seg,
            p['pKa'], static['acid_base'], p['phi'], p['J'],
            static['beta'], static['k0'],
            static['correct_mask'], static['n'],
            static['i_idx'], static['j_idx'],
            n_points    = static['n_points_sim'],
            smooth_width= static.get('smooth_width', 0.0),
            entropy_triu= entropy_triu,
        )
        return correct_bond_score(final, static['n'], static['correct_triu_idx'])

    scores = jax.vmap(score_one_schedule)(all_pH_schedules_array)
    tau    = static.get('tau', 5.0)
    log_p  = jax.nn.log_softmax(scores * tau)
    loss   = -log_p[target_idx]
    return loss, scores


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
    beta               : float   (default 1.0)
    k0                 : float   (default 1.0)
    n_points_sim       : int     (default 40)
    n_points_equil     : int     (default 60)
    tau                : float   (default 5.0)
    J_max              : float   (default 3.5)
    smooth_width       : float   (default 0.0 = sharp transitions)
    S_max              : float   (default 0.0 = no entropy params)
    seed               : int
    """
    n = config['n_species']
    assert n % 2 == 0 and 2 <= n <= 10, "n_species must be even and in [2, 10]"

    J_max  = float(config.get('J_max',  3.5))
    S_max  = float(config.get('S_max',  0.0))
    smooth = float(config.get('smooth_width', 0.0))

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
    }

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    target_sched  = [float(x) for x in config['target_pH_schedule']]
    all_schedules = all_unique_permutations(target_sched)
    target_idx    = all_schedules.index(target_sched)
    duration      = float(config['duration_per_seg'])
    all_pH_array  = jnp.array(all_schedules, dtype=float)

    print(f"Species      : {n}  ({n//2} correct pairs)")
    print(f"Target sched : {target_sched}")
    print(f"Permutations : {len(all_schedules)}  (target idx = {target_idx})")
    print(f"Equilibration: pH 7,  t = {static['equil_duration']} (β=1, current params)")
    print(f"J_max        : {J_max}  kT")
    print(f"Smooth width : {smooth}  time units  ({'enabled' if smooth > 0 else 'disabled'})")
    if S_max > 0:
        n_entropy = n * (n + 1) // 2
        print(f"Entropy params: {n_entropy}  (S_max = {S_max} kT)")

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    initial_state = make_initial_state(n)

    # ------------------------------------------------------------------
    # Initialise trainable parameters
    # ------------------------------------------------------------------
    rng    = np.random.default_rng(int(config.get('seed', 42)))
    pH_min = float(min(target_sched))
    pH_max = float(max(target_sched))
    pKa_init = []
    for i in range(n):
        if acid_base_np[i] == 0:
            centre = np.clip(pH_min + 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        else:
            centre = np.clip(pH_max - 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        pKa_init.append(float(centre))

    init_phys = {
        'pKa': jnp.array(pKa_init),
        'phi': jnp.array(0.2),
        'J'  : jnp.array(1.5),
    }
    if S_max > 0.0:
        n_entropy = n * (n + 1) // 2
        init_phys['entropy'] = jnp.zeros(n_entropy)   # start at ΔS=0

    raw_params = unconstrain_params(init_phys, J_max=J_max, S_max=S_max)

    # ------------------------------------------------------------------
    # Optimiser  (lr is a *traced* argument — no recompile needed to change it)
    # ------------------------------------------------------------------
    clip_norm = float(config.get('clip_norm', 0.5))
    lr        = float(config.get('learning_rate', 0.02))

    # Core optimiser without LR (applied manually so lr can vary at runtime)
    _opt_core = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.scale_by_adam(),
    )
    opt_state = _opt_core.init(raw_params)

    # ------------------------------------------------------------------
    # JIT-compiled step
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
            loss_fn, has_aux=True
        )(raw_params)

        # Zero any NaN/Inf gradient components before they corrupt params.
        # This is the JAX-native approach: replace bad values with zero so
        # the step is skipped for those components rather than exploding.
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            grads
        )

        updates, new_opt_state = _opt_core.update(grads, opt_state)
        # Scale by learning rate (negative sign = gradient descent)
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

    # Only track best params when BOTH loss AND all params are finite
    best_params  = raw_params if _params_finite(raw_params) else None
    nan_count    = 0
    max_nan_halvings = 6    # give up halving after this many consecutive NaNs

    for epoch in range(1, n_epochs):
        new_params, new_opt, lv, sc = step(raw_params, opt_state, lr_jax)

        loss_ok   = np.isfinite(float(lv))
        params_ok = _params_finite(new_params)

        if loss_ok and params_ok:
            raw_params = new_params
            opt_state  = new_opt
            nan_count  = 0
            if best_params is None or float(lv) < min(loss_history):
                best_params = raw_params
        else:
            nan_count += 1
            if nan_count <= max_nan_halvings:
                # Halve learning rate
                lr      = lr * 0.5
                lr_jax  = jnp.array(lr)
                print(f"  NaN at epoch {epoch} — halving lr → {lr:.2e}", flush=True)
            else:
                if nan_count == max_nan_halvings + 1:
                    print(f"  NaN persists after {max_nan_halvings} halvings; "
                          f"holding at lr={lr:.2e}", flush=True)

            if best_params is not None:
                raw_params = best_params
            # Reset Adam momentum — stale moments contain bad curvature info
            opt_state = _opt_core.init(raw_params)
            # Use last known-good values for history
            lv = loss_history[-1]
            sc = score_history[-1]

        loss_history.append(float(lv))
        score_history.append(np.array(sc))
        p_cur = constrain_params(raw_params, J_max=J_max, S_max=S_max)
        param_history.append(_snapshot(p_cur, S_max))

        if epoch % max(1, n_epochs // 15) == 0 or epoch == n_epochs - 1:
            pKa_str = ' '.join(f'{float(v):.2f}' for v in p_cur['pKa'])
            entropy_str = ''
            if S_max > 0.0:
                ent_mean = float(jnp.mean(p_cur['entropy']))
                ent_max  = float(jnp.max(p_cur['entropy']))
                entropy_str = f' | S̄={ent_mean:.3f} S_max={ent_max:.3f}'
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"loss={float(lv):.4f} | "
                f"target={float(sc[target_idx]):.3f} | "
                f"mean_other={float(jnp.mean(jnp.delete(sc, target_idx))):.3f} | "
                f"pKa=[{pKa_str}] | "
                f"φ={float(p_cur['phi']):.3f} | J={float(p_cur['J']):.3f}"
                f"{entropy_str}",
                flush=True,
            )

    print("\nTraining complete.")

    # Final equilibrium state for reporting / visualisation
    p_final      = constrain_params(raw_params, J_max=J_max, S_max=S_max)
    entropy_triu = p_final.get('entropy', None)
    equil_state  = simulate_schedule_scan(
        initial_state,
        jnp.array([7.0]),
        static['equil_duration'],
        p_final['pKa'], static['acid_base'], p_final['phi'], p_final['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'], static['i_idx'], static['j_idx'],
        n_points    = static['n_points_equil'],
        smooth_width= smooth,
        entropy_triu= entropy_triu,
    )

    return (
        raw_params,
        loss_history,
        score_history,
        param_history,
        static,
        all_schedules,
        target_idx,
        equil_state,
    )


def _snapshot(p, S_max):
    """Dict snapshot of constrained params for history recording."""
    snap = {
        'pKa': np.array(p['pKa']),
        'phi': float(p['phi']),
        'J'  : float(p['J']),
    }
    if S_max > 0.0 and 'entropy' in p:
        snap['entropy'] = np.array(p['entropy'])
    return snap
