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

    The ± formula from the problem statement:

        q^{+}_i = +1 / (1 + 10^{+(pH - pKa_i)})   [base-like residue]
        q^{-}_i = -1 / (1 + 10^{-(pH - pKa_i)})   [acid-like residue]

    Note: the user's formula states ± denotes "acid/base respectively",
    but physically the + case describes a *base* (positive at low pH,
    neutral at high pH, e.g. Lys/Arg/His) and the − case describes an
    *acid* (neutral at low pH, negative at high pH, e.g. Asp/Glu).
    We adopt the physically correct convention here:
        acid_base = 1  →  base  →  q ∈ (0, +1)
        acid_base = 0  →  acid  →  q ∈ (−1, 0)

    Args:
        pKa      : jax array (n,), pKa values constrained to [3, 10]
        pH       : float scalar
        acid_base: int array (n,), 1 = base-like, 0 = acid-like

    Returns:
        charges  : jax array (n,) in (−1, +1)
    """
    # Base: q = +1 / (1 + 10^{pH − pKa})
    #         = +1 / (1 + exp((pH − pKa) · ln10))
    # Approaches +1 at low pH (protonated), 0 at high pH.
    base_q = 1.0 / (1.0 + jnp.exp(LN10 * (pH - pKa)))

    # Acid: q = −1 / (1 + 10^{pKa − pH})
    #         = −1 / (1 + exp((pKa − pH) · ln10))
    # Approaches 0 at low pH (protonated/neutral), −1 at high pH.
    acid_q = -1.0 / (1.0 + jnp.exp(LN10 * (pKa - pH)))

    return jnp.where(acid_base == 1, base_q, acid_q)


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------

def interaction_energy_matrix(charges, correct_mask, phi, J):
    """
    Free-energy matrix ΔG_{ij} for all monomer pairs.

    The electrostatic potential when monomers i and j are bonded is:

        V_{ij} = q_i · q_j / (r · D)

    which, in dimensionless kT units, becomes:

        ΔG_{ij} = J · q_i · q_j

    where J = e² / (4π ε₀ r D k_B T) is the dimensionless coupling.

    Correct pairs (adjacent alphabet, e.g. A–B, C–D) receive the full
    interaction.  Incorrect pairs receive a fraction φ ∈ [0, 1], modelling
    steric mismatch or backbone conformational entropy loss.

    If the correct partners carry opposite charges (one base + one acid),
    ΔG < 0 for correct pairs → thermodynamically favourable.
    φ < 1 makes incorrect pairs less stable.

    Args:
        charges     : (n,) fractional charges from Henderson-Hasselbalch
        correct_mask: bool (n, n), True for pairs (0,1),(2,3),(4,5),...
        phi         : scalar ∈ [0, 1], steric mismatch factor
        J           : scalar > 0, coupling constant (kT units)

    Returns:
        dG : (n, n) symmetric matrix
    """
    qi = charges[:, None]   # (n, 1)
    qj = charges[None, :]   # (1, n)
    V = J * qi * qj          # (n, n)
    return jnp.where(correct_mask, V, phi * V)


# ---------------------------------------------------------------------------
# Rate constants (detailed balance)
# ---------------------------------------------------------------------------

def forward_rate_matrix(dG, beta, k0):
    """
    Forward rate constants consistent with detailed balance.

        k_f^{ij} = k_0 · exp(−β · ΔG_{ij})
        k_b^{ij} = k_0   (uniform backward rate)

    Ratio: k_f / k_b = exp(−β ΔG) = Boltzmann factor ✓

    At equilibrium:
        [X_i X_j]_eq = exp(−β ΔG_{ij}) · [X_i] · [X_j]

    Args:
        dG  : (n, n) free-energy matrix
        beta: scalar, inverse temperature β = 1/(k_B T)
        k0  : scalar, reference rate constant

    Returns:
        kf  : (n, n) forward rates
    """
    return k0 * jnp.exp(-beta * dG)
