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
  k_f^{ij} = k0 · exp(−β ΔG_{ij}),   k_b = k0   (detailed balance)

Conservation
------------
  Σ_i [X_i] + 2 · Σ_{i≤j} [X_i X_j] = const
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from .physics import henderson_hasselbalch, interaction_energy_matrix, forward_rate_matrix


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
            monomer_entropy=None, allowed_mask=None):
    """CRN ODE right-hand side."""
    free       = jnp.maximum(state[:n], 0.0)
    dimer_triu = jnp.maximum(state[n:], 0.0)

    charges    = henderson_hasselbalch(pKa, pH, acid_base)
    dG         = interaction_energy_matrix(charges, correct_mask, phi, J,
                                           monomer_entropy=monomer_entropy,
                                           allowed_mask=allowed_mask)
    kf         = forward_rate_matrix(dG, beta, k0)
    dimer_full = triu_to_full(dimer_triu, n, i_idx, j_idx)
    flux       = kf * jnp.outer(free, free) - k0 * dimer_full

    d_free       = -(jnp.sum(flux, axis=1) + jnp.diag(flux))
    d_dimer_triu = flux[i_idx, j_idx]
    return jnp.concatenate([d_free, d_dimer_triu])


def simulate_segment(state, pH, duration,
                     pKa, acid_base, phi, J, beta, k0,
                     correct_mask, n, i_idx, j_idx,
                     n_points=60, monomer_entropy=None, allowed_mask=None):
    t_span = jnp.linspace(0.0, float(duration), n_points)
    ode_fn = lambda s, t, _pKa, _phi, _J: crn_ode(
        s, t, _pKa, acid_base, _phi, _J,
        beta, k0, pH, correct_mask, n, i_idx, j_idx,
        monomer_entropy, allowed_mask)
    traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                  rtol=1e-4, atol=1e-6, mxstep=1000)
    return traj[-1], traj


def simulate_schedule(initial_state, pH_schedule, duration_per_seg,
                      pKa, acid_base, phi, J, beta, k0,
                      correct_mask, n, i_idx, j_idx,
                      n_points=60, monomer_entropy=None, allowed_mask=None):
    """Python-loop simulation — use for visualisation only (not inside JIT)."""
    state, traj_list = initial_state, []
    for pH in pH_schedule:
        state, traj = simulate_segment(
            state, float(pH), duration_per_seg,
            pKa, acid_base, phi, J, beta, k0,
            correct_mask, n, i_idx, j_idx, n_points,
            monomer_entropy, allowed_mask)
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
                           allowed_mask=None):
    """
    Scan-based simulation — O(1) JAX graph via lax.scan + vmap.

    smooth_width : if > 0, pH ramps smoothly at each segment start using
                  a logistic sigmoid over this many time units.
    ph_initial   : pH before this schedule (used for the first segment's
                   ramp when smooth_width > 0).  Defaults to pH_schedule[0]
                   (no ramp on first segment).
    allowed_mask : if not None, pairs outside this mask have ΔG=0
                   (used with --specific_bonds).
    """
    t_span = jnp.linspace(0.0, float(duration_per_seg), n_points)

    if smooth_width > 0.0:
        w = float(smooth_width)
        ph0 = pH_schedule_array[0] if ph_initial is None else jnp.array(float(ph_initial))

        def segment_fn(carry, pH_target):
            state, pH_prev = carry

            def ode_fn(s, t, _pKa, _phi, _J, _pH_prev, _pH_target):
                blend = jax.nn.sigmoid((t - w * 0.5) / (w * 0.2 + 1e-8))
                pH    = _pH_prev + (_pH_target - _pH_prev) * blend
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta, k0, pH, correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask)

            traj = odeint(ode_fn, state, t_span,
                          pKa, phi, J, pH_prev, pH_target,
                          rtol=1e-4, atol=1e-6, mxstep=1000)
            return (traj[-1], pH_target), None

        (final_state, _), _ = jax.lax.scan(segment_fn,
                                            (initial_state, ph0),
                                            pH_schedule_array)
    else:
        def segment_fn(state, pH):
            def ode_fn(s, t, _pKa, _phi, _J):
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta, k0, pH, correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask)
            traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                          rtol=1e-4, atol=1e-6, mxstep=1000)
            return traj[-1], None

        final_state, _ = jax.lax.scan(segment_fn, initial_state, pH_schedule_array)

    return final_state
