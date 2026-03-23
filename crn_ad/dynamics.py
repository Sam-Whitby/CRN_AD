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
    Off-diagonal entries are mirrored: both [i,j] and [j,i] equal the
    stored value (we only count each heterodimer once in the state).
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
            correct_mask, n, i_idx, j_idx):
    """
    CRN ODE right-hand side.

    Flux for reaction (i, j):
        F_{ij} = k_f^{ij} · [X_i][X_j] − k_b · [X_i X_j]

    For heterodimers (i ≠ j):
        d[X_i X_j]/dt = F_{ij}
        d[X_i]/dt    += −F_{ij}   (one monomer of each type consumed)
        d[X_j]/dt    += −F_{ij}

    For homodimers (i = j):
        d[X_i X_i]/dt = F_{ii}
        d[X_i]/dt    += −2 F_{ii}  (two monomers of type i consumed)

    In matrix form (using the full symmetric flux matrix):
        d[X_i]/dt = −Σ_j F_{ij} − F_{ii}
                  = −row_sum_i(F) − F_{ii}

    This correctly accounts for the extra homodimer contribution.
    """
    # Clamp to avoid spurious negatives from ODE integrator
    free       = jnp.maximum(state[:n], 0.0)
    dimer_triu = jnp.maximum(state[n:], 0.0)

    # Charges at current pH
    charges = henderson_hasselbalch(pKa, pH, acid_base)

    # Free energy and rate matrices
    dG = interaction_energy_matrix(charges, correct_mask, phi, J)
    kf = forward_rate_matrix(dG, beta, k0)

    # Full symmetric dimer matrix
    dimer_full = triu_to_full(dimer_triu, n, i_idx, j_idx)

    # Net flux  F[i,j] = kf[i,j]*[Xi][Xj] − k0*[Xi Xj]
    flux = kf * jnp.outer(free, free) - k0 * dimer_full

    # Rate equations
    d_free       = -(jnp.sum(flux, axis=1) + jnp.diag(flux))
    d_dimer_triu = flux[i_idx, j_idx]

    return jnp.concatenate([d_free, d_dimer_triu])


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def make_initial_state(n):
    """
    Fully denatured initial state: all monomers free at equal concentration,
    no dimers.  Total monomer content = 1.
    """
    n_dimers = n * (n + 1) // 2
    return jnp.concatenate([jnp.ones(n) / n, jnp.zeros(n_dimers)])


def simulate_segment(state, pH, duration,
                     pKa, acid_base, phi, J, beta, k0,
                     correct_mask, n, i_idx, j_idx,
                     n_points=60):
    """
    Integrate the CRN ODE over one constant-pH segment.

    pH may be a Python float or a JAX scalar (compatible with lax.scan).

    Returns
    -------
    final_state : state at t = duration
    trajectory  : array (n_points, state_dim) including t=0
    """
    t_span = jnp.linspace(0.0, float(duration), n_points)

    # Close over static (non-differentiated) quantities; odeint only
    # receives the trainable JAX arrays as *args so autodiff flows through them.
    ode_fn = lambda s, t, _pKa, _phi, _J: crn_ode(
        s, t, _pKa, acid_base, _phi, _J,
        beta, k0, pH,             # pH passed directly – no float() conversion
        correct_mask, n, i_idx, j_idx
    )

    traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                  rtol=1e-4, atol=1e-6, mxstep=1000)
    return traj[-1], traj


def simulate_schedule(initial_state, pH_schedule, duration_per_seg,
                      pKa, acid_base, phi, J, beta, k0,
                      correct_mask, n, i_idx, j_idx,
                      n_points=60):
    """
    Simulate the CRN through a sequence of constant-pH segments.
    Python for-loop version – convenient for visualisation (returns trajectories).
    Not used inside JIT; call simulate_schedule_scan for training.
    """
    state = initial_state
    traj_list = []

    for pH in pH_schedule:
        state, traj = simulate_segment(
            state, float(pH), duration_per_seg,
            pKa, acid_base, phi, J, beta, k0,
            correct_mask, n, i_idx, j_idx, n_points
        )
        traj_list.append(traj)

    return state, traj_list


def simulate_schedule_scan(initial_state, pH_schedule_array,
                           duration_per_seg,
                           pKa, acid_base, phi, J, beta, k0,
                           correct_mask, n, i_idx, j_idx,
                           n_points=40):
    """
    Scan-based simulation: uses jax.lax.scan over segments so that the
    JAX computation graph is O(1) in the number of segments and can be
    composed with jax.vmap over batches of schedules.

    pH_schedule_array : JAX array (n_segments,) of pH values.

    Returns
    -------
    final_state : state at end of last segment (no trajectory stored)
    """
    t_span = jnp.linspace(0.0, float(duration_per_seg), n_points)

    def segment_fn(state, pH):
        ode_fn = lambda s, t, _pKa, _phi, _J: crn_ode(
            s, t, _pKa, acid_base, _phi, _J,
            beta, k0, pH, correct_mask, n, i_idx, j_idx
        )
        traj = odeint(ode_fn, state, t_span, pKa, phi, J,
                  rtol=1e-4, atol=1e-6, mxstep=1000)
        return traj[-1], None

    final_state, _ = jax.lax.scan(segment_fn, initial_state, pH_schedule_array)
    return final_state


def equilibrate_denatured(n, acid_base, correct_mask, i_idx, j_idx,
                          J=2.0, k0=1.0, ref_pH=7.0,
                          duration=300.0, n_points=150):
    """
    Equilibrate the system at β = 0 (infinite temperature) to obtain
    a chemically randomised, 'denatured' starting state.

    At β = 0:  k_f = k_b = k_0 for every reaction regardless of ΔG.
    The equilibrium satisfies  [X_i X_j]_eq = [X_i][X_j],  which gives
    a mixture of all monomers and dimers weighted by combinatorics alone.

    This is used as the initial condition before every pH-schedule run,
    ensuring the system always starts from the same unbiased reference.
    """
    pKa_dummy = jnp.ones(n) * 7.0   # charges irrelevant at β=0
    phi_dummy  = jnp.array(1.0)
    J_dummy    = jnp.array(float(J))
    beta_zero  = jnp.array(0.0)
    initial    = make_initial_state(n)

    final, _ = simulate_segment(
        initial, ref_pH, duration,
        pKa_dummy, acid_base, phi_dummy, J_dummy, beta_zero, k0,
        correct_mask, n, i_idx, j_idx, n_points
    )
    return final
