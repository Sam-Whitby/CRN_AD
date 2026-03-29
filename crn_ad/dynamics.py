"""
ODE dynamics for the CRN and simulation utilities.

State vector layout
-------------------
state[:n]   free monomer concentrations  [X_0], [X_1], ..., [X_{n-1}]
state[n:]   dimer concentrations in upper-triangle order
            Length = n*(n+1)//2

Reactions
---------
  X_i + X_j  ⇌  X_i X_j   for all 0 ≤ i ≤ j ≤ n−1
  Metropolis kinetics (detailed balance):
    k_f^{ij} = k0 · exp(−β · max(ΔG_{ij}, 0))
    k_b^{ij} = k0 · exp(+β · min(ΔG_{ij}, 0))

Non-negativity
--------------
  The RHS clips state to ≥ 0 before computing fluxes, and every odeint
  output is projected via jnp.maximum(state, 0) before being carried
  forward.  Both operations are differentiable through JAX autodiff.

Conservation
------------
  Σ_i [X_i] + 2 · Σ_{i≤j} [X_i X_j] = const
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from .physics import henderson_hasselbalch, interaction_energy_matrix, rate_matrices


def make_triu_indices(n):
    """Upper-triangle indices for an n×n matrix, including diagonal."""
    return np.triu_indices(n)


def triu_to_full(triu_vec, n, i_idx, j_idx):
    mat = jnp.zeros((n, n))
    mat = mat.at[i_idx, j_idx].set(triu_vec)
    mat = mat + mat.T - jnp.diag(jnp.diag(mat))
    return mat


def make_initial_state(n):
    """All monomers free at equal concentration, no dimers. M(0) = 1."""
    n_dimers = n * (n + 1) // 2
    return jnp.concatenate([jnp.ones(n) / n, jnp.zeros(n_dimers)])


def crn_ode(state, t,
            pKa, acid_base, phi, J, beta, k0, pH,
            correct_mask, n, i_idx, j_idx,
            monomer_entropy=None, allowed_mask=None, no_self_bonds=False):
    """CRN ODE right-hand side."""
    free       = jnp.maximum(state[:n], 0.0)
    dimer_triu = jnp.maximum(state[n:], 0.0)

    charges    = henderson_hasselbalch(pKa, pH, acid_base)
    dG         = interaction_energy_matrix(charges, correct_mask, phi, J,
                                           monomer_entropy=monomer_entropy,
                                           allowed_mask=allowed_mask)
    kf, kb     = rate_matrices(dG, beta, k0)
    dimer_full = triu_to_full(dimer_triu, n, i_idx, j_idx)
    flux       = kf * jnp.outer(free, free) - kb * dimer_full

    if no_self_bonds:
        # Zero diagonal flux entries: d[X_i·X_i]/dt = 0, so self-dimer
        # concentrations stay at 0 forever (true absence of species, not
        # merely zero energy).  Baked in at JAX trace time as a static branch.
        flux = flux * (1.0 - jnp.eye(n))

    d_free       = -(jnp.sum(flux, axis=1) + jnp.diag(flux))
    d_dimer_triu = flux[i_idx, j_idx]
    return jnp.concatenate([d_free, d_dimer_triu])


def simulate_segment(state, pH, duration,
                     pKa, acid_base, phi, J, beta, k0,
                     correct_mask, n, i_idx, j_idx,
                     n_points=60, monomer_entropy=None, allowed_mask=None,
                     beta_ramp_duration=0.0, no_self_bonds=False):
    """Simulate one pH segment.

    beta_ramp_duration: if > 0, beta ramps linearly from 0 → beta over the
    first beta_ramp_duration time units, then stays at beta.  Used for the
    equilibration segment to avoid a sharp-switch ODE transient.
    """
    t_span = jnp.linspace(0.0, float(duration), n_points)
    if beta_ramp_duration > 0.0:
        _ramp = float(beta_ramp_duration)
        def ode_fn(s, t, _pKa, _phi, _J):
            beta_t = jnp.where(t < _ramp, float(beta) * t / _ramp, float(beta))
            return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                           beta_t, k0, pH, correct_mask, n, i_idx, j_idx,
                           monomer_entropy, allowed_mask, no_self_bonds)
    else:
        def ode_fn(s, t, _pKa, _phi, _J):
            return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                           beta, k0, pH, correct_mask, n, i_idx, j_idx,
                           monomer_entropy, allowed_mask, no_self_bonds)
    traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                  rtol=1e-4, atol=1e-6, mxstep=1000)
    # Clip to non-negative: odeint can drift slightly below zero due to
    # numerical error even though the RHS already clips when computing flux.
    final = jnp.maximum(traj[-1], 0.0)
    return final, traj


def simulate_schedule(initial_state, pH_schedule, duration_per_seg,
                      pKa, acid_base, phi, J, beta, k0,
                      correct_mask, n, i_idx, j_idx,
                      n_points=60, monomer_entropy=None, allowed_mask=None,
                      beta_ramp_duration=0.0, no_self_bonds=False):
    """Python-loop simulation — use for visualisation only (not inside JIT)."""
    state, traj_list = initial_state, []
    for pH in pH_schedule:
        state, traj = simulate_segment(
            state, float(pH), duration_per_seg,
            pKa, acid_base, phi, J, beta, k0,
            correct_mask, n, i_idx, j_idx, n_points,
            monomer_entropy, allowed_mask, beta_ramp_duration, no_self_bonds)
        traj_list.append(traj)
    return state, traj_list


def simulate_schedule_scan(initial_state, pH_schedule_array,
                           duration_per_seg,
                           pKa, acid_base, phi, J, beta, k0,
                           correct_mask, n, i_idx, j_idx,
                           n_points=40,
                           smooth_width=0.0,
                           monomer_entropy=None,
                           ph_initial=None,
                           allowed_mask=None,
                           beta_ramp_duration=0.0,
                           no_self_bonds=False):
    """
    Scan-based simulation — O(1) JAX graph via lax.scan + vmap.

    smooth_width       : if > 0, pH ramps smoothly at each segment start.
    ph_initial         : pH before this schedule (for smooth ramp on first seg).
    allowed_mask       : if not None, pairs outside this mask have ΔG=0.
    beta_ramp_duration : if > 0, beta ramps linearly 0→beta over this many
                         time units at the start of the segment (used for the
                         equilibration segment to avoid stiff transients).
    """
    t_span = jnp.linspace(0.0, float(duration_per_seg), n_points)
    _ramp  = float(beta_ramp_duration)   # Python float — used in Python if below

    if smooth_width > 0.0:
        w = float(smooth_width)
        ph0 = pH_schedule_array[0] if ph_initial is None else jnp.array(float(ph_initial))

        # Build the ODE function once, with the beta-ramp decision baked in at
        # Python (trace) time so no dead-branch gradient blowup occurs.
        if _ramp > 0.0:
            def ode_fn_smooth(s, t, _pKa, _phi, _J, _pH_prev, _pH_target):
                blend  = jax.nn.sigmoid((t - w * 0.5) / (w * 0.2 + 1e-8))
                pH     = _pH_prev + (_pH_target - _pH_prev) * blend
                beta_t = jnp.where(t < _ramp, beta * t / _ramp, beta)
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta_t, k0, pH, correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)
        else:
            def ode_fn_smooth(s, t, _pKa, _phi, _J, _pH_prev, _pH_target):
                blend  = jax.nn.sigmoid((t - w * 0.5) / (w * 0.2 + 1e-8))
                pH     = _pH_prev + (_pH_target - _pH_prev) * blend
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta, k0, pH, correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)

        def segment_fn(carry, pH_target):
            state, pH_prev = carry
            traj = odeint(ode_fn_smooth, state, t_span,
                          pKa, phi, J, pH_prev, pH_target,
                          rtol=1e-4, atol=1e-6, mxstep=1000)
            return (jnp.maximum(traj[-1], 0.0), pH_target), None

        (final_state, _), _ = jax.lax.scan(segment_fn,
                                            (initial_state, ph0),
                                            pH_schedule_array)
    else:
        # Build with beta-ramp decision baked in at Python (trace) time.
        if _ramp > 0.0:
            def ode_fn_flat(s, t, _pKa, _phi, _J, _pH):
                beta_t = jnp.where(t < _ramp, beta * t / _ramp, beta)
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta_t, k0, _pH, correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)
        else:
            def ode_fn_flat(s, t, _pKa, _phi, _J, _pH):
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta, k0, _pH, correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)

        def segment_fn(state, pH):
            traj = odeint(ode_fn_flat, state, t_span, pKa, phi, J, pH,
                          rtol=1e-4, atol=1e-6, mxstep=1000)
            return jnp.maximum(traj[-1], 0.0), None

        final_state, _ = jax.lax.scan(segment_fn, initial_state, pH_schedule_array)

    return final_state
