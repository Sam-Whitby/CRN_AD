"""
Loss function and training loop.

Loss design
-----------
We treat the problem as a *multi-class classification* task:
given a pH schedule (one of all unique permutations of the target
sequence), the system should only 'fold' correctly under the one target
permutation.

Score for schedule s:
    score(s) = fraction of total monomer content residing in *correct* dimers
               after running from the denatured state through schedule s.

Loss (softmax cross-entropy / InfoNCE):
    L = −log [ exp(τ · score_target) / Σ_s exp(τ · score_s) ]
      = −log_softmax(τ · scores)[target_idx]

  •  τ (temperature) sharpens the gradient: large τ → winner-take-all.
  •  Minimising L simultaneously maximises the target score and suppresses
     all other permutation scores.
  •  In the limit of many permutations this approximates a contrastive loss
     (InfoNCE), whose minimiser is a mutual-information maximiser between
     the target schedule identity and the folding outcome.

Relationship to KL divergence
------------------------------
Let p_s = softmax(τ · scores) be the 'predicted' distribution over schedules
and let p* be the one-hot distribution at the target index.  Then:
    L = KL(p* ‖ p_s) + H(p*)  = KL(p* ‖ p_s)   (since H(p*)=0)
So we are minimising the KL divergence from the ideal one-hot distribution,
which is exactly what we want.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from itertools import permutations
from functools import partial

from .dynamics import (simulate_schedule, simulate_schedule_scan,
                       equilibrate_denatured, make_triu_indices)


# ---------------------------------------------------------------------------
# Parameter constraints
# ---------------------------------------------------------------------------

def constrain_params(raw):
    """
    Map unconstrained (ℝ-valued) raw parameters to physical ranges.

        pKa ∈ [3, 10]  via  3 + 7·σ(raw_pKa)
        φ   ∈ [0,  1]  via  σ(raw_phi)
        J   > 0        via  softplus(raw_J) + 0.5   (lower-bounded at 0.5)
    """
    return {
        'pKa': 3.0 + 7.0 * jax.nn.sigmoid(raw['pKa']),
        'phi': jax.nn.sigmoid(raw['phi']),
        'J':   jax.nn.softplus(raw['J']) + 0.5,
    }


def unconstrain_params(phys):
    """
    Inverse of constrain_params – useful for warm-starting from a known
    physical configuration.
    """
    pKa_norm = (jnp.array(phys['pKa']) - 3.0) / 7.0
    return {
        'pKa': jnp.log(pKa_norm / (1.0 - pKa_norm + 1e-8) + 1e-8),
        'phi': jnp.log(jnp.array(phys['phi']) / (1.0 - jnp.array(phys['phi']) + 1e-8) + 1e-8),
        'J':   jnp.log(jnp.exp(jnp.array(phys['J']) - 0.5) - 1.0 + 1e-8),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def correct_bond_score(state, n, correct_triu_idx):
    """
    Fraction of total monomer content residing in correct dimers.

        score = 2 · Σ_{correct pairs} [X_i X_j] / (total monomer content)

    The factor 2 accounts for the two monomers per dimer.
    Total monomer content = Σ_i [X_i] + 2 · Σ_{i≤j} [X_i X_j] (conserved).

    Returns a scalar ∈ [0, 1].
    """
    free       = state[:n]
    dimer_triu = state[n:]
    total      = jnp.sum(free) + 2.0 * jnp.sum(dimer_triu)
    correct    = jnp.sum(dimer_triu[correct_triu_idx])
    return 2.0 * correct / (total + 1e-10)


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
                 duration_per_seg, static, denatured_state):
    """
    Softmax cross-entropy loss – compiled efficiently with vmap + lax.scan.

    Instead of unrolling a Python loop over 24 schedules × 4 segments,
    we use:
      • jax.lax.scan   – iterates over the N segments of one schedule
      • jax.vmap       – vectorises over the M different schedules

    This keeps the JAX computation graph O(1) in both M and N, giving
    a large speedup in JIT compilation time.

    Args
    ----
    raw_params              : dict of unconstrained JAX arrays
    all_pH_schedules_array  : JAX array (n_schedules, n_segments) of pH values
    target_idx              : int, row index of the target schedule
    duration_per_seg        : float
    static                  : dict of non-differentiated quantities
    denatured_state         : JAX array (state_dim,)

    Returns
    -------
    (loss, scores)  where scores has shape (n_schedules,)
    """
    p = constrain_params(raw_params)

    def score_one_schedule(pH_sched):
        """Score the CRN after running through one pH schedule (1D array)."""
        final = simulate_schedule_scan(
            denatured_state,
            pH_sched,
            duration_per_seg,
            p['pKa'],
            static['acid_base'],
            p['phi'],
            p['J'],
            static['beta'],
            static['k0'],
            static['correct_mask'],
            static['n'],
            static['i_idx'],
            static['j_idx'],
            n_points=static['n_points_sim'],
        )
        return correct_bond_score(final, static['n'], static['correct_triu_idx'])

    # Vectorise over the batch of schedules (axis 0 of all_pH_schedules_array)
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
    target_pH_schedule : list[float], pH values for each segment
    duration_per_seg   : float
    n_epochs           : int
    learning_rate      : float
    beta               : float  (inverse temperature, default 1.0)
    k0                 : float  (base rate constant, default 1.0)
    n_points_sim       : int    (ODE time points per segment, default 40)
    tau                : float  (softmax temperature, default 5.0)
    seed               : int

    Returns
    -------
    raw_params, loss_history, score_history, static, all_schedules,
    target_idx, denatured_state
    """
    n = config['n_species']
    assert n % 2 == 0 and 2 <= n <= 10, "n_species must be even and in [2, 10]"

    # ------------------------------------------------------------------
    # Static quantities (numpy arrays – never differentiated)
    # ------------------------------------------------------------------
    # Alternating acid/base: species 0=base, 1=acid, 2=base, 3=acid, ...
    # Correct pairs (0,1), (2,3), ... then have opposite charges → attract.
    acid_base_np = np.array([i % 2 for i in range(n)], dtype=int)

    correct_mask_np = np.zeros((n, n), dtype=bool)
    for k in range(n // 2):
        i, j = 2 * k, 2 * k + 1
        correct_mask_np[i, j] = True
        correct_mask_np[j, i] = True

    i_idx, j_idx = make_triu_indices(n)

    # Which positions in the triu vector correspond to correct pairs?
    correct_triu_idx = np.array([
        pos for pos, (ii, jj) in enumerate(zip(i_idx, j_idx))
        if correct_mask_np[ii, jj]
    ])

    static = {
        'n'               : n,
        'acid_base'       : jnp.array(acid_base_np),
        'correct_mask'    : jnp.array(correct_mask_np),
        'correct_mask_np' : correct_mask_np,
        'i_idx'           : i_idx,
        'j_idx'           : j_idx,
        'correct_triu_idx': jnp.array(correct_triu_idx),
        'beta'            : float(config.get('beta', 1.0)),
        'k0'              : float(config.get('k0', 1.0)),
        'n_points_sim'    : int(config.get('n_points_sim', 40)),
        'tau'             : float(config.get('tau', 5.0)),
    }

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    target_sched  = [float(x) for x in config['target_pH_schedule']]
    all_schedules = all_unique_permutations(target_sched)
    target_idx    = all_schedules.index(target_sched)
    duration      = float(config['duration_per_seg'])
    # Pre-convert to JAX array once: (n_schedules, n_segments)
    all_pH_array  = jnp.array(all_schedules, dtype=float)

    print(f"Species      : {n}  ({n//2} correct pairs)")
    print(f"Target sched : {target_sched}")
    print(f"Permutations : {len(all_schedules)}  (target idx = {target_idx})")

    # ------------------------------------------------------------------
    # Denatured starting state (β = 0 equilibration)
    # ------------------------------------------------------------------
    print("Equilibrating denatured state (β=0) ...", flush=True)
    denatured_state = equilibrate_denatured(
        n,
        static['acid_base'],
        static['correct_mask'],
        i_idx, j_idx,
        J=2.0,
        k0=static['k0'],
        ref_pH=7.0,
        duration=300.0,
        n_points=150,
    )
    print(f"  Free monomer total : {float(jnp.sum(denatured_state[:n])):.4f}")
    print(f"  Dimer total (×2)   : {2*float(jnp.sum(denatured_state[n:])):.4f}")

    # ------------------------------------------------------------------
    # Initialise trainable parameters
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(config.get('seed', 42)))

    # Initialise pKa so that adjacent correct pairs (acid + base) both carry
    # significant charges across the schedule pH range, maximising the salt-
    # bridge driving force from the start.
    #
    # Acid-like species (i=0,2,...): low pKa → charged (negative) throughout
    #   the schedule range.  We target pKa ≈ pH_min + 1.
    # Base-like species (i=1,3,...): high pKa → charged (positive) throughout.
    #   We target pKa ≈ pH_max − 1.
    #
    # This ensures strong A–B interactions early in training so the gradient
    # is meaningful.  The optimiser then differentiates pKa values between
    # species pairs to create schedule selectivity.
    pH_min = float(min(target_sched))
    pH_max = float(max(target_sched))
    pKa_init = []
    for i in range(n):
        if acid_base_np[i] == 0:   # acid-like
            centre = np.clip(pH_min + 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        else:                       # base-like
            centre = np.clip(pH_max - 1.5 + rng.normal(0.0, 0.5), 3.1, 9.9)
        pKa_init.append(centre)
    pKa_phys = jnp.array(pKa_init)

    raw_params = unconstrain_params({
        'pKa': pKa_phys,
        'phi': jnp.array(0.2),   # start with mostly-blocked incorrect pairs
        'J'  : jnp.array(2.0),   # moderate coupling ≈ 2 kT
    })

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    lr        = float(config.get('learning_rate', 0.05))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = optimizer.init(raw_params)

    # ------------------------------------------------------------------
    # JIT-compiled gradient function
    # ------------------------------------------------------------------
    loss_fn = partial(
        compute_loss,
        all_pH_schedules_array=all_pH_array,
        target_idx=target_idx,
        duration_per_seg=duration,
        static=static,
        denatured_state=denatured_state,
    )

    @jax.jit
    def step(raw_params, opt_state):
        # has_aux=True: loss_fn returns (scalar_loss, aux_scores)
        (loss_val, scores), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(raw_params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_raw_params = optax.apply_updates(raw_params, updates)
        return new_raw_params, new_opt_state, loss_val, scores

    # Warm-up (first call compiles, may take a minute)
    print("Compiling JAX graph (first call) ...", flush=True)
    raw_params, opt_state, lv, sc = step(raw_params, opt_state)
    print("Compilation done.\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs     = int(config.get('n_epochs', 300))
    loss_history  = [float(lv)]
    score_history = [np.array(sc)]

    for epoch in range(1, n_epochs):
        raw_params, opt_state, lv, sc = step(raw_params, opt_state)
        loss_history.append(float(lv))
        score_history.append(np.array(sc))

        if epoch % max(1, n_epochs // 15) == 0 or epoch == n_epochs - 1:
            p = constrain_params(raw_params)
            pKa_str = ' '.join(f'{float(v):.2f}' for v in p['pKa'])
            print(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"loss={float(lv):.4f} | "
                f"target={float(sc[target_idx]):.3f} | "
                f"mean_other={float(jnp.mean(jnp.delete(sc, target_idx))):.3f} | "
                f"pKa=[{pKa_str}] | "
                f"φ={float(p['phi']):.3f} | J={float(p['J']):.3f}",
                flush=True,
            )

    print("\nTraining complete.")
    return (
        raw_params,
        loss_history,
        score_history,
        static,
        all_schedules,
        target_idx,
        denatured_state,
    )
