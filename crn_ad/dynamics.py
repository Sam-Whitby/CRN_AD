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
  The RHS clips state to ≥ 0 before computing fluxes, and every ODE
  output is projected via jnp.maximum(state, 0) before being carried
  forward.  Both operations are differentiable through JAX autodiff.

Conservation
------------
  Σ_i [X_i] + 2 · Σ_{i≤j} [X_i X_j] = const

ODE solver
----------
  Uses Diffrax (Tsit5 adaptive solver) with RecursiveCheckpointAdjoint.
  RecursiveCheckpointAdjoint stores checkpoints of the forward trajectory
  and differentiates through them directly, avoiding the classical adjoint
  ODE solve backwards in time that causes NaN for stiff systems.
"""

import jax
import jax.numpy as jnp
import numpy as np
import diffrax

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
    t0    = 0.0
    t1    = float(duration)
    dt0   = t1 / max(n_points - 1, 1)
    ts    = jnp.linspace(t0, t1, n_points)
    _ramp = float(beta_ramp_duration)

    if _ramp > 0.0:
        def vf(t, s, _args):
            beta_t = jnp.where(t < _ramp, float(beta) * t / _ramp, float(beta))
            return crn_ode(s, t, pKa, acid_base, phi, J, beta_t, k0, float(pH),
                           correct_mask, n, i_idx, j_idx,
                           monomer_entropy, allowed_mask, no_self_bonds)
    else:
        def vf(t, s, _args):
            return crn_ode(s, t, pKa, acid_base, phi, J, float(beta), k0, float(pH),
                           correct_mask, n, i_idx, j_idx,
                           monomer_entropy, allowed_mask, no_self_bonds)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        diffrax.Tsit5(),
        t0=t0, t1=t1, dt0=dt0,
        y0=state,
        args=None,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        max_steps=4096,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    traj  = sol.ys                       # (n_points, state_size)
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
                           no_self_bonds=False,
                           return_traj=False):
    """
    Scan-based simulation — O(1) JAX graph via lax.scan + vmap.

    smooth_width       : if > 0, pH ramps smoothly at each segment start.
    ph_initial         : pH before this schedule (for smooth ramp on first seg).
    allowed_mask       : if not None, pairs outside this mask have ΔG=0.
    beta_ramp_duration : if > 0, beta ramps linearly 0→beta over this many
                         time units at the start of the segment.
    return_traj        : if True, also return stacked segment trajectories as a
                         (n_segs, n_points, state_size) array alongside the
                         final state.  When False (default), returns only the
                         final state (cheaper: uses SaveAt(t1=True)).
    """
    t0    = 0.0
    t1    = float(duration_per_seg)
    dt0   = t1 / max(n_points - 1, 1)
    _ramp = float(beta_ramp_duration)

    if smooth_width > 0.0:
        w   = float(smooth_width)
        ph0 = pH_schedule_array[0] if ph_initial is None else jnp.array(float(ph_initial))

        if _ramp > 0.0:
            def vf(t, s, ph_args):
                _pH_prev, _pH_target = ph_args
                blend  = jax.nn.sigmoid((t - w * 0.5) / (w * 0.2 + 1e-8))
                pH     = _pH_prev + (_pH_target - _pH_prev) * blend
                beta_t = jnp.where(t < _ramp, float(beta) * t / _ramp, float(beta))
                return crn_ode(s, t, pKa, acid_base, phi, J, beta_t, k0, pH,
                               correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)
        else:
            def vf(t, s, ph_args):
                _pH_prev, _pH_target = ph_args
                blend = jax.nn.sigmoid((t - w * 0.5) / (w * 0.2 + 1e-8))
                pH    = _pH_prev + (_pH_target - _pH_prev) * blend
                return crn_ode(s, t, pKa, acid_base, phi, J, float(beta), k0, pH,
                               correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)

        if return_traj:
            def segment_fn(carry, pH_target):
                state, pH_prev = carry
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(vf), diffrax.Tsit5(),
                    t0=t0, t1=t1, dt0=dt0, y0=state,
                    args=(pH_prev, pH_target),
                    saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points)),
                    stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                    max_steps=4096, adjoint=diffrax.RecursiveCheckpointAdjoint(),
                )
                traj  = sol.ys                        # (n_points, state_size)
                final = jnp.maximum(traj[-1], 0.0)
                return (final, pH_target), traj
        else:
            def segment_fn(carry, pH_target):
                state, pH_prev = carry
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(vf), diffrax.Tsit5(),
                    t0=t0, t1=t1, dt0=dt0, y0=state,
                    args=(pH_prev, pH_target),
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                    max_steps=4096, adjoint=diffrax.RecursiveCheckpointAdjoint(),
                )
                return (jnp.maximum(sol.ys[0], 0.0), pH_target), None

        (final_state, _), trajs = jax.lax.scan(
            segment_fn, (initial_state, ph0), pH_schedule_array)

    else:
        if _ramp > 0.0:
            def vf(t, s, pH):
                beta_t = jnp.where(t < _ramp, float(beta) * t / _ramp, float(beta))
                return crn_ode(s, t, pKa, acid_base, phi, J, beta_t, k0, pH,
                               correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)
        else:
            def vf(t, s, pH):
                return crn_ode(s, t, pKa, acid_base, phi, J, float(beta), k0, pH,
                               correct_mask, n, i_idx, j_idx,
                               monomer_entropy, allowed_mask, no_self_bonds)

        if return_traj:
            def segment_fn(state, pH):
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(vf), diffrax.Tsit5(),
                    t0=t0, t1=t1, dt0=dt0, y0=state,
                    args=pH,
                    saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_points)),
                    stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                    max_steps=4096, adjoint=diffrax.RecursiveCheckpointAdjoint(),
                )
                traj  = sol.ys                       # (n_points, state_size)
                final = jnp.maximum(traj[-1], 0.0)
                return final, traj
        else:
            def segment_fn(state, pH):
                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(vf), diffrax.Tsit5(),
                    t0=t0, t1=t1, dt0=dt0, y0=state,
                    args=pH,
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                    max_steps=4096, adjoint=diffrax.RecursiveCheckpointAdjoint(),
                )
                return jnp.maximum(sol.ys[0], 0.0), None

        final_state, trajs = jax.lax.scan(
            segment_fn, initial_state, pH_schedule_array)

    if return_traj:
        return final_state, trajs   # trajs: (n_segs, n_points, state_size)
    return final_state
