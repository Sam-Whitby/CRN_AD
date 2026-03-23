"""
ODE dynamics for the CRN and simulation utilities.

State vector layout
-------------------
state[:n]   free monomer concentrations  [X_0], [X_1], ..., [X_{n-1}]
state[n:]   dimer concentrations in upper-triangle order
            [X_0 X_0], [X_0 X_1], ..., [X_{n-1} X_{n-1}]
            Length = n*(n+1)//2

Reactions
---------
  X_i + X_j  ⇌  X_i X_j   for all 0 ≤ i ≤ j ≤ n−1
  k_f^{ij} = k0 · exp(−β ΔG_{ij}),   k_b = k0   (detailed balance)

Conservation
------------
  Σ_i [X_i] + 2 · Σ_{i≤j} [X_i X_j] = const  (total monomer content)

Smooth-pH option
----------------
  When smooth_width > 0, the pH ramps smoothly from the previous segment's
  value to the current target using a logistic sigmoid over `smooth_width`
  time units at the start of each segment.  This eliminates discontinuities
  in the ODE right-hand side at segment boundaries, which prevents the
  adjoint solver from taking excessively small steps and producing NaN
  gradients.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

from .physics import henderson_hasselbalch, interaction_energy_matrix, forward_rate_matrix


# ---------------------------------------------------------------------------
# Index helpers (numpy, static – not differentiated)
# ---------------------------------------------------------------------------

def make_triu_indices(n):
    """Upper-triangle indices for an n×n matrix, including diagonal."""
    return np.triu_indices(n)


def triu_to_full(triu_vec, n, i_idx, j_idx):
    """
    Expand upper-triangle vector to a full symmetric n×n matrix.

    Diagonal entries appear once (homodimers [X_i X_i]).
    Off-diagonal entries are mirrored.
    """
    mat = jnp.zeros((n, n))
    mat = mat.at[i_idx, j_idx].set(triu_vec)
    mat = mat + mat.T - jnp.diag(jnp.diag(mat))
    return mat


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def crn_ode(state, t,
            pKa, acid_base, phi, J, beta, k0, pH,
            correct_mask, n, i_idx, j_idx,
            entropy_triu=None):
    """
    CRN ODE right-hand side.

    Flux for reaction (i, j):
        F_{ij} = k_f^{ij} · [X_i][X_j] − k_b · [X_i X_j]

    d[X_i]/dt   = −row_sum_i(F) − F_{ii}
    d[XiXj]/dt  = F_{ij}
    """
    free       = jnp.maximum(state[:n], 0.0)
    dimer_triu = jnp.maximum(state[n:], 0.0)

    charges = henderson_hasselbalch(pKa, pH, acid_base)
    dG      = interaction_energy_matrix(charges, correct_mask, phi, J,
                                        entropy_triu=entropy_triu,
                                        i_idx=i_idx, j_idx=j_idx, n=n)
    kf      = forward_rate_matrix(dG, beta, k0)

    dimer_full   = triu_to_full(dimer_triu, n, i_idx, j_idx)
    flux         = kf * jnp.outer(free, free) - k0 * dimer_full

    d_free       = -(jnp.sum(flux, axis=1) + jnp.diag(flux))
    d_dimer_triu = flux[i_idx, j_idx]

    return jnp.concatenate([d_free, d_dimer_triu])


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def make_initial_state(n):
    """All monomers free at equal concentration, no dimers.  M(0) = 1."""
    n_dimers = n * (n + 1) // 2
    return jnp.concatenate([jnp.ones(n) / n, jnp.zeros(n_dimers)])


def simulate_segment(state, pH, duration,
                     pKa, acid_base, phi, J, beta, k0,
                     correct_mask, n, i_idx, j_idx,
                     n_points=60, entropy_triu=None):
    """
    Integrate the CRN ODE over one constant-pH segment.

    Returns
    -------
    final_state : state at t = duration
    trajectory  : array (n_points, state_dim) including t=0
    """
    t_span = jnp.linspace(0.0, float(duration), n_points)

    ode_fn = lambda s, t, _pKa, _phi, _J: crn_ode(
        s, t, _pKa, acid_base, _phi, _J,
        beta, k0, pH,
        correct_mask, n, i_idx, j_idx, entropy_triu
    )

    traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                  rtol=1e-4, atol=1e-6, mxstep=1000)
    return traj[-1], traj


def simulate_schedule(initial_state, pH_schedule, duration_per_seg,
                      pKa, acid_base, phi, J, beta, k0,
                      correct_mask, n, i_idx, j_idx,
                      n_points=60, entropy_triu=None):
    """
    Simulate through a sequence of constant-pH segments.
    Python for-loop version — convenient for visualisation (returns trajectories).
    Use simulate_schedule_scan for training inside JIT.
    """
    state     = initial_state
    traj_list = []
    for pH in pH_schedule:
        state, traj = simulate_segment(
            state, float(pH), duration_per_seg,
            pKa, acid_base, phi, J, beta, k0,
            correct_mask, n, i_idx, j_idx, n_points, entropy_triu
        )
        traj_list.append(traj)
    return state, traj_list


def simulate_schedule_scan(initial_state, pH_schedule_array,
                           duration_per_seg,
                           pKa, acid_base, phi, J, beta, k0,
                           correct_mask, n, i_idx, j_idx,
                           n_points=40,
                           smooth_width=0.0,
                           entropy_triu=None):
    """
    Scan-based simulation: O(1) JAX graph size via lax.scan + vmap.

    pH_schedule_array : JAX array (n_segments,)
    smooth_width      : if > 0, pH ramps smoothly at segment starts using
                        a logistic sigmoid over this many time units.
                        Eliminates discontinuities that cause adjoint NaNs.

    Returns final state only (no trajectory stored — use simulate_schedule
    for visualisation).
    """
    t_span = jnp.linspace(0.0, float(duration_per_seg), n_points)

    if smooth_width > 0.0:
        # Carry: (state, pH_prev).  pH ramps smoothly at each segment start.
        def segment_fn(carry, pH_target):
            state, pH_prev = carry

            def ode_fn(s, t, _pKa, _phi, _J, _pH_prev, _pH_target):
                # Logistic ramp: reaches ~88% of target at t = smooth_width
                blend = jax.nn.sigmoid((t - smooth_width * 0.5) / (smooth_width * 0.2 + 1e-8))
                pH    = _pH_prev + (_pH_target - _pH_prev) * blend
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta, k0, pH, correct_mask, n, i_idx, j_idx,
                               entropy_triu)

            traj = odeint(ode_fn, state, t_span,
                          pKa, phi, J, pH_prev, pH_target,
                          rtol=1e-4, atol=1e-6, mxstep=1000)
            return (traj[-1], pH_target), None

        # Initial pH_prev = pH of first segment (no ramp needed for very first)
        init_carry = (initial_state, pH_schedule_array[0])
        (final_state, _), _ = jax.lax.scan(segment_fn, init_carry, pH_schedule_array)

    else:
        # Original: constant pH per segment
        def segment_fn(state, pH):
            def ode_fn(s, t, _pKa, _phi, _J):
                return crn_ode(s, t, _pKa, acid_base, _phi, _J,
                               beta, k0, pH, correct_mask, n, i_idx, j_idx,
                               entropy_triu)
            traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                          rtol=1e-4, atol=1e-6, mxstep=1000)
            return traj[-1], None

        final_state, _ = jax.lax.scan(segment_fn, initial_state, pH_schedule_array)

    return final_state
