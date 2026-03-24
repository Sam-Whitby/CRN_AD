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
                               monomer_entropy=None):
    """
    Free-energy matrix ΔG_{ij} for all monomer pairs.

        ΔG_{ij} = J · q_i · q_j                (correct pairs)
        ΔG_{ij} = φ · J · q_i · q_j            (incorrect pairs)

    If monomer_entropy is provided, an additive conformational-entropy
    penalty is included:

        ΔG_{ij} += s_i + s_j

    where s_i is the entropy cost (kT) of monomer i losing conformational
    freedom upon dimerisation.  This term is symmetric in i,j, ensuring
    detailed balance is preserved (the Boltzmann factor exp(-β·ΔG) is
    still the unique equilibrium ratio [X_i X_j]/([X_i][X_j])).

    monomer_entropy : jax array, shape () (scalar, shared) or (n,) (per-monomer)
                      Constrained to [0, S_max].  Ignored when None.
    """
    qi = charges[:, None]
    qj = charges[None, :]
    V  = J * qi * qj
    dG = jnp.where(correct_mask, V, phi * V)

    if monomer_entropy is not None:
        n  = charges.shape[0]
        # Broadcast scalar or per-monomer vector to length-n
        s  = jnp.broadcast_to(jnp.atleast_1d(monomer_entropy), (n,))
        dG = dG + s[:, None] + s[None, :]

    return dG


def forward_rate_matrix(dG, beta, k0):
    """
    k_f^{ij} = k_0 · exp(−β · ΔG_{ij}),   k_b^{ij} = k_0
    """
    return k0 * jnp.exp(-beta * dG)
