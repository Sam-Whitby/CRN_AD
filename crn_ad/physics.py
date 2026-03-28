"""
Core physics: Henderson-Hasselbalch charges, electrostatic energies,
and detailed-balance rate constants.
"""

import jax.numpy as jnp
import numpy as np

LN10 = float(np.log(10.0))


def henderson_hasselbalch(pKa, pH, acid_base):
    """
    Fractional charges from the Henderson-Hasselbalch equation.

        acid_base = 1  →  base  →  q ∈ (0, +1)   (positive at low pH)
        acid_base = 0  →  acid  →  q ∈ (−1, 0)   (negative at high pH)
    """
    base_q = 1.0 / (1.0 + jnp.exp(LN10 * (pH - pKa)))
    acid_q = -1.0 / (1.0 + jnp.exp(LN10 * (pKa - pH)))
    return jnp.where(acid_base == 1, base_q, acid_q)


def interaction_energy_matrix(charges, correct_mask, phi, J,
                               monomer_entropy=None, allowed_mask=None):
    """
    Free-energy matrix ΔG_{ij} for all monomer pairs.

    Default (allowed_mask=None):
        ΔG_{ij} = J · q_i · q_j                (correct pairs)
        ΔG_{ij} = φ · J · q_i · q_j            (all other pairs)

    With allowed_mask (--specific_bonds mode):
        ΔG_{ij} = J · q_i · q_j                (correct species + correct type)
        ΔG_{ij} = φ · J · q_i · q_j            (correct species, wrong type)
        ΔG_{ij} = 0                             (wrong species — no interaction)

    allowed_mask is True wherever an interaction is permitted (correct species
    pair, any type).  It is a superset of correct_mask.

    If monomer_entropy is provided, an additive conformational-entropy
    penalty is included:

        ΔG_{ij} += s_i + s_j

    monomer_entropy : jax array, shape () (scalar, shared) or (n,) (per-monomer)
                      Constrained to [0, S_max].  Ignored when None.
    """
    qi = charges[:, None]
    qj = charges[None, :]
    V  = J * qi * qj
    if allowed_mask is None:
        dG = jnp.where(correct_mask, V, phi * V)
    else:
        # Correct pair: full V; allowed-but-wrong-type: phi*V; forbidden: 0
        dG = jnp.where(correct_mask, V, jnp.where(allowed_mask, phi * V, 0.0))

    if monomer_entropy is not None:
        n  = charges.shape[0]
        # Broadcast scalar or per-monomer vector to length-n
        s  = jnp.broadcast_to(jnp.atleast_1d(monomer_entropy), (n,))
        dG = dG + s[:, None] + s[None, :]

    return dG


def rate_matrices(dG, beta, k0):
    """
    Metropolis kinetics — guaranteed detailed balance for all ΔG.

    k_f^{ij} = k0 · exp(−β · max(ΔG_{ij}, 0))
    k_b^{ij} = k0 · exp(+β · min(ΔG_{ij}, 0))

    When ΔG ≤ 0 (favourable formation):
        k_f = k0            — formation is unimpeded
        k_b = k0·exp(βΔG) < k0  — breaking is slowed by the Boltzmann factor
    When ΔG > 0 (unfavourable formation):
        k_f = k0·exp(−βΔG) < k0 — formation is penalised
        k_b = k0            — breaking is unimpeded

    Ratio:  k_f / k_b = exp(−β·ΔG)  in both cases  ✓  (detailed balance).

    This is the continuous-time analogue of Metropolis–Hastings: the
    system moves at full speed whenever a reaction is downhill, and only
    pays a kinetic cost when climbing an energy barrier.
    """
    kf = k0 * jnp.exp(-beta * jnp.maximum(dG, 0.0))
    kb = k0 * jnp.exp( beta * jnp.minimum(dG, 0.0))
    return kf, kb
