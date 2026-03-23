"""
Loss function and training loop.

Starting state
--------------
Rather than a β=0 'denatured' state, we now equilibrate at pH 7 with the
CURRENT trained parameters (β=1) for a long time before every schedule run.
This gives a physically motivated resting state (the polymer at physiological
pH) and means the gradient flows through the equilibration step.

Loss design
-----------
Multi-class cross-entropy (InfoNCE / KL divergence):

    L = −log_softmax(τ · scores)[target_idx]
      = KL(δ_target ‖ softmax(τ · scores))

Maximises correct-bond fraction under the target schedule while simultaneously
suppressing it under all permutation schedules.
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

def constrain_params(raw):
    """Map unconstrained (ℝ) raw parameters to physical ranges."""
    return {
        'pKa': 3.0 + 7.0 * jax.nn.sigmoid(raw['pKa']),   # [3, 10]
        'phi': jax.nn.sigmoid(raw['phi']),                  # [0, 1]
        'J':   0.5 + 3.0 * jax.nn.sigmoid(raw['J']),      # [0.5, 3.5] — hard cap prevents stiff ODE
    }


def unconstrain_params(phys):
    """Inverse of constrain_params for warm-starting."""
    pKa_norm = (jnp.array(phys['pKa']) - 3.0) / 7.0
    pKa_norm = jnp.clip(pKa_norm, 1e-4, 1 - 1e-4)
    phi_val   = jnp.clip(jnp.array(phys['phi']), 1e-4, 1 - 1e-4)
    return {
        'pKa': jnp.log(pKa_norm / (1.0 - pKa_norm)),
        'phi': jnp.log(phi_val / (1.0 - phi_val)),
        'J':   jnp.log(jnp.exp(jnp.array(phys['J']) - 0.5) - 1.0 + 1e-6),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def correct_bond_score(state, n, correct_triu_idx):
    """
    Fraction of total monomer content residing in correct dimers.

        score = 2·Σ_{correct} [XᵢXⱼ] / (Σᵢ [Xᵢ] + 2·Σ_{i≤j} [XᵢXⱼ])

    The denominator is the conserved total monomer content (= 1 initially).
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
# Loss  (vmap over schedules, lax.scan over segments, includes pre-equilibration)
# ---------------------------------------------------------------------------

def compute_loss(raw_params, all_pH_schedules_array, target_idx,
                 duration_per_seg, static, initial_state):
    """
    Softmax cross-entropy loss.

    Procedure
    ---------
    1. Equilibrate the system at pH 7 with the CURRENT trained parameters
       for `static['equil_duration']` time units.  This gives the resting
       state before the experiment (β=1, pH=7, current pKa/φ/J).
    2. Run each schedule permutation from that equilibrium state.
    3. Compute the correct-bond fraction for each schedule.
    4. Return −log_softmax(τ · scores)[target_idx].

    The gradient flows back through both the schedule simulation and the
    pH-7 equilibration step.

    Args
    ----
    raw_params              : dict of unconstrained JAX arrays
    all_pH_schedules_array  : (n_schedules, n_segments) JAX float array
    target_idx              : int
    duration_per_seg        : float
    static                  : dict of non-differentiated quantities
    initial_state           : JAX array (n + n*(n+1)//2,), all monomers free

    Returns
    -------
    (loss, scores)
    """
    p = constrain_params(raw_params)

    # ------------------------------------------------------------------
    # Step 1: equilibrate at pH 7 with current parameters
    # ------------------------------------------------------------------
    equil_state = simulate_schedule_scan(
        initial_state,
        jnp.array([7.0]),          # single pH-7 segment
        static['equil_duration'],
        p['pKa'], static['acid_base'], p['phi'], p['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'],
        static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
    )

    # ------------------------------------------------------------------
    # Step 2: score each schedule from the pH-7 equilibrium
    # ------------------------------------------------------------------
    def score_one_schedule(pH_sched):
        final = simulate_schedule_scan(
            equil_state,
            pH_sched,
            duration_per_seg,
            p['pKa'], static['acid_base'], p['phi'], p['J'],
            static['beta'], static['k0'],
            static['correct_mask'], static['n'],
            static['i_idx'], static['j_idx'],
            n_points=static['n_points_sim'],
        )
        return correct_bond_score(final, static['n'], static['correct_triu_idx'])

    scores = jax.vmap(score_one_schedule)(all_pH_schedules_array)

    tau   = static.get('tau', 5.0)
    log_p = jax.nn.log_softmax(scores * tau)
    loss  = -log_p[target_idx]
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
    equil_duration     : float   (pH-7 pre-equilibration time, default 200)
    n_epochs           : int
    learning_rate      : float
    beta               : float   (default 1.0)
    k0                 : float   (default 1.0)
    n_points_sim       : int     (ODE points per schedule segment, default 40)
    n_points_equil     : int     (ODE points for equilibration, default 60)
    tau                : float   (softmax temperature, default 5.0)
    seed               : int

    Returns
    -------
    raw_params, loss_history, score_history, param_history,
    static, all_schedules, target_idx, equil_state
    """
    n = config['n_species']
    assert n % 2 == 0 and 2 <= n <= 10, "n_species must be even and in [2, 10]"

    # ------------------------------------------------------------------
    # Static (non-differentiated) quantities
    # ------------------------------------------------------------------
    # Even index → base-like (positive at low pH)
    # Odd index  → acid-like (negative at high pH)
    # Adjacent pairs (0,1),(2,3),... carry opposite charges → attract.
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
        'k0'              : float(config.get('k0', 1.0)),
        'n_points_sim'    : int(config.get('n_points_sim', 40)),
        'n_points_equil'  : int(config.get('n_points_equil', 60)),
        'equil_duration'  : float(config.get('equil_duration', 80.0)),
        'tau'             : float(config.get('tau', 5.0)),
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

    # ------------------------------------------------------------------
    # Initial state (all monomers free, no dimers)
    # ------------------------------------------------------------------
    initial_state = make_initial_state(n)

    # ------------------------------------------------------------------
    # Initialise trainable parameters
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(config.get('seed', 42)))

    # Acid-like species: pKa near pH_min+1 → charged (negative) across schedule
    # Base-like species: pKa near pH_max-1 → charged (positive) across schedule
    # This maximises salt-bridge driving force from the start.
    pH_min = float(min(target_sched))
    pH_max = float(max(target_sched))
    pKa_init = []
    for i in range(n):
        if acid_base_np[i] == 0:   # acid-like
            centre = np.clip(pH_min + 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        else:                       # base-like
            centre = np.clip(pH_max - 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        pKa_init.append(float(centre))

    # Cap J raw so that constrained J = softplus(raw)+0.5 ≤ 3.5 initially
    raw_params = unconstrain_params({
        'pKa': jnp.array(pKa_init),
        'phi': jnp.array(0.2),
        'J'  : jnp.array(1.5),      # softplus(1.5)+0.5 ≈ 2.3 kT
    })

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    lr        = float(config.get('learning_rate', 0.02))
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),   # strict clipping to prevent NaN
        optax.adam(lr),
    )
    opt_state = optimizer.init(raw_params)

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
    def step(raw_params, opt_state):
        (loss_val, scores), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(raw_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_raw_params = optax.apply_updates(raw_params, updates)
        return new_raw_params, new_opt_state, loss_val, scores

    print("Compiling JAX graph (first call) ...", flush=True)
    raw_params, opt_state, lv, sc = step(raw_params, opt_state)
    print("Compilation done.\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs      = int(config.get('n_epochs', 300))
    loss_history  = [float(lv)]
    score_history = [np.array(sc)]

    p0 = constrain_params(raw_params)
    param_history = [{
        'pKa': np.array(p0['pKa']),
        'phi': float(p0['phi']),
        'J'  : float(p0['J']),
    }]

    best_params  = raw_params   # track best finite params for NaN recovery
    nan_count    = 0

    for epoch in range(1, n_epochs):
        new_params, new_opt, lv, sc = step(raw_params, opt_state)

        # NaN guard: revert to best params with a reduced learning rate
        if not np.isfinite(float(lv)):
            nan_count += 1
            if nan_count == 1:
                print(f"  Warning: NaN at epoch {epoch} — reverting to best params")
            raw_params = best_params
            # reinitialise optimizer with smaller lr
            lr_red    = lr * 0.3
            optimizer = optax.chain(
                optax.clip_by_global_norm(0.3),
                optax.adam(lr_red),
            )
            opt_state = optimizer.init(raw_params)
            lv = loss_history[-1]
            sc = score_history[-1]
        else:
            raw_params = new_params
            opt_state  = new_opt
            if float(lv) < min(loss_history):
                best_params = raw_params
            nan_count = 0

        loss_history.append(float(lv))
        score_history.append(np.array(sc))
        p_cur = constrain_params(raw_params)
        param_history.append({
            'pKa': np.array(p_cur['pKa']),
            'phi': float(p_cur['phi']),
            'J'  : float(p_cur['J']),
        })

        if epoch % max(1, n_epochs // 15) == 0 or epoch == n_epochs - 1:
            pKa_str = ' '.join(f'{float(v):.2f}' for v in p_cur['pKa'])
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"loss={float(lv):.4f} | "
                f"target={float(sc[target_idx]):.3f} | "
                f"mean_other={float(jnp.mean(jnp.delete(sc, target_idx))):.3f} | "
                f"pKa=[{pKa_str}] | "
                f"φ={float(p_cur['phi']):.3f} | J={float(p_cur['J']):.3f}",
                flush=True,
            )

    print("\nTraining complete.")

    # Compute final equilibrium state for reporting / visualisation
    p_final  = constrain_params(raw_params)
    equil_state = simulate_schedule_scan(
        initial_state,
        jnp.array([7.0]),
        static['equil_duration'],
        p_final['pKa'], static['acid_base'], p_final['phi'], p_final['J'],
        static['beta'], static['k0'],
        static['correct_mask'], static['n'], static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
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
