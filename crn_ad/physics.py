"""
Core physics: Henderson-Hasselbalch charges, electrostatic energies,
and detailed-balance rate constants.
"""

import jax.numpy as jnp
import numpy as np

# ln(10) used for base-10 exponent conversion
LN10 = float(np.log(10.0))


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------

def henderson_hasselbalch(pKa, pH, acid_base):
    """
    Fractional charges from the Henderson-Hasselbalch equation.

        acid_base = 1  →  base  →  q ∈ (0, +1)   (positive at low pH)
        acid_base = 0  →  acid  →  q ∈ (−1, 0)   (negative at high pH)
    """
    base_q = 1.0 / (1.0 + jnp.exp(LN10 * (pH - pKa)))
    acid_q = -1.0 / (1.0 + jnp.exp(LN10 * (pKa - pH)))
    return jnp.where(acid_base == 1, base_q, acid_q)


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------

def interaction_energy_matrix(charges, correct_mask, phi, J,
                               entropy_triu=None, i_idx=None, j_idx=None, n=None):
    """
    Free-energy matrix ΔG_{ij} for all monomer pairs.

        ΔG_{ij} = J · q_i · q_j                      (correct pairs)
        ΔG_{ij} = φ · J · q_i · q_j                  (incorrect pairs)

    If entropy_triu is provided, a conformational-entropy penalty is added:
        ΔG_{ij} += ΔS_{ij}   (ΔS ≥ 0 makes dimerisation less favourable)

    entropy_triu : jax array (n*(n+1)//2,), upper-triangle entropy costs (kT)
    i_idx, j_idx : upper-triangle row/col indices (required if entropy_triu given)
    n            : int (required if entropy_triu given)
    """
    qi = charges[:, None]
    qj = charges[None, :]
    V  = J * qi * qj
    dG = jnp.where(correct_mask, V, phi * V)

    if entropy_triu is not None:
        # Expand triu entropy vector to symmetric n×n matrix
        entropy_mat = jnp.zeros((n, n))
        entropy_mat = entropy_mat.at[i_idx, j_idx].set(entropy_triu)
        entropy_mat = entropy_mat + entropy_mat.T - jnp.diag(jnp.diag(entropy_mat))
        dG = dG + entropy_mat

    return dG


# ---------------------------------------------------------------------------
# Rate constants (detailed balance)
# ---------------------------------------------------------------------------

def forward_rate_matrix(dG, beta, k0):
    """
    Forward rate constants consistent with detailed balance.

        k_f^{ij} = k_0 · exp(−β · ΔG_{ij})
        k_b^{ij} = k_0
    """
    return k0 * jnp.exp(-beta * dG)
