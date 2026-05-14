"""
Chain-position model: derives phi_matrix, pair_entropy, and k0_matrix from
trainable residue positions x ∈ [0, 1].

Replaces three ad-hoc scalar parameters (φ, s_i, ε_‡) with 2N position
parameters x_i that encode where each titratable residue sits along the
polymer backbone.  All three phenomenological quantities emerge from the
sequence separation d_ij = |x_i − x_j| · L_chain:

  phi_ij   = phi0 · exp(−d_ij / λ_c)          (contact selectivity, non-native)
  ΔS_ij    = (3/2) kT · ln(d_ij / l0)          (Jacobson–Stockmayer loop entropy)
  k0_ij    = k0 · (d0 / d_ij)^α               (Wilemski–Fixman contact rate)

Physical references
-------------------
phi_ij  : proximity-dependent encounter probability, analogous to the
          contact selectivity used in Go-type models [Go 1983].  Pairs
          close on the chain have higher effective phi because they are
          more likely to encounter each other in 3D space.

ΔS_ij   : Jacobson–Stockmayer (1950) loop-closure entropy for a Gaussian
          chain.  The J-factor J(L) ~ L^(−3/2) gives an entropy cost
          ΔS = (3/2) k_B ln L for closing a loop of L residues.

k0_ij   : Wilemski–Fixman (1974) and Szabo–Schulten–Schulten (1980)
          first-passage time for intramolecular contact formation.
          For a Gaussian chain, the contact rate scales as k ∝ L^(−3/2),
          so k0_ij = k0 · (d0/d_ij)^(3/2).  With hydrodynamics (Zimm):
          k ∝ L^(−3ν), giving α = 3ν ≈ 1.76 for good-solvent chains.

All operations are differentiable with respect to x via JAX autodiff.
"""

import jax.numpy as jnp


def compute_pair_separation(x, L_chain, eps_d=0.5):
    """Sequence separation d_ij = |x_i − x_j| · L_chain + eps_d  (residues).

    eps_d regularises d=0 (self-pairs) so log/power operations are finite.
    Default eps_d=0.5 residues ≪ lambda_c (typically 20 residues).
    """
    return jnp.abs(x[:, None] - x[None, :]) * float(L_chain) + float(eps_d)


def compute_phi_matrix(d_ij, correct_mask, phi0, lambda_c):
    """Contact selectivity matrix  phi_ij ∈ (0, phi0].

    Native  pairs (correct_mask=True) : phi_ij = 1.0  (full interaction).
    Non-native pairs                  : phi_ij = phi0 · exp(−d_ij / λ_c).

    phi0   — maximum non-native phi (typically ≤ 1.0; sets the contact
             strength at the shortest non-native separation d ≈ eps_d).
    lambda_c — contact locality length scale (residues).  Pairs separated
             by d ≫ λ_c have phi ≈ 0 and effectively do not interact.
    """
    phi_nonnative = float(phi0) * jnp.exp(-d_ij / float(lambda_c))
    return jnp.where(correct_mask, 1.0, phi_nonnative)


def compute_pair_entropy(d_ij, l0):
    """Jacobson–Stockmayer loop-closure entropy in kT.

    ΔS_ij = (3/2) ln(d_ij / l0).

    Positive values (d_ij > l0) are unfavourable — longer loops cost more
    entropy.  l0 is the reference separation (≈ 1 Kuhn length ≈ 3–4 residues
    for an IDP) at which the entropy penalty is zero.
    """
    return 1.5 * jnp.log(d_ij / float(l0))


def compute_k0_matrix(d_ij, k0_base, d0, alpha=1.5):
    """Wilemski–Fixman contact rate prefactor  k0_ij = k0 · (d0 / d_ij)^α.

    Distant residues (d_ij ≫ d0) associate much slower than nearby ones.
    alpha = 1.5  : Gaussian chain (Rouse/Smoluchowski limit)
    alpha = 1.76 : self-avoiding chain (Zimm with ν ≈ 0.588)
    alpha = 2.0  : 1D Rouse scaling (no hydrodynamics, theta solvent)

    d0 is the reference separation at which k0_ij = k0_base.
    """
    return float(k0_base) * (float(d0) / d_ij) ** float(alpha)


def compute_chain_quantities(x, correct_mask, L_chain, lambda_c, l0, d0,
                              k0_base, phi0=1.0, alpha=1.5, eps_d=0.5):
    """
    Derive all chain-physics quantities from residue positions x.

    Mean-field 3D Gaussian chain model: φ_ij = 1 for all pairs (native and
    non-native alike).  Discrimination between native and non-native contacts
    comes entirely from ΔS_ij (Jacobson–Stockmayer loop-closure entropy) and
    k0_ij (Wilemski–Fixman contact rate).  The (3/2) exponents in both
    formulas encode averaging over all 3D Gaussian-chain conformations.

    Parameters
    ----------
    x            : JAX array (2N,)   normalised positions ∈ [0, 1]
    correct_mask : JAX bool (2N, 2N) True for native classifier pairs
    L_chain      : float             chain length in residues
    lambda_c     : float             unused (kept for backward compatibility)
    l0           : float             entropy reference separation (residues)
    d0           : float             rate-prefactor reference separation (residues)
    k0_base      : float             base rate constant (units of main k0)
    phi0         : float             unused (kept for backward compatibility)
    alpha        : float             Rouse/WF exponent (default 1.5)
    eps_d        : float             d=0 regularisation offset (residues)

    Returns
    -------
    phi_matrix   : (2N, 2N)  all-ones — φ = 1 everywhere (mean-field chain)
    pair_entropy : (2N, 2N)  loop-closure entropy cost ΔS_ij  (kT, additive to ΔG)
    k0_matrix    : (2N, 2N)  pair-specific rate prefactors
    """
    d_ij         = compute_pair_separation(x, L_chain, eps_d)
    phi_matrix   = jnp.ones_like(d_ij)   # φ=1: mean-field 3D Gaussian chain
    pair_entropy = compute_pair_entropy(d_ij, l0)
    k0_matrix    = compute_k0_matrix(d_ij, k0_base, d0, alpha)
    return phi_matrix, pair_entropy, k0_matrix


def chain_roughness_diagnostic(phi_matrix, J, charges, correct_mask):
    """Zwanzig landscape roughness σ² around native contacts.

    σ² = mean over non-native pairs of (phi_ij · J · qi · qj)²

    This is the variance of non-native contact energies, which controls the
    Zwanzig exp(β²σ²/2) kinetic enhancement without needing ε_‡ as a free
    parameter.  Larger σ² → slower effective diffusion toward the correct
    native state → longer kinetic memory.

    Returns (sigma2_mean, roughness_matrix).
    """
    V                = phi_matrix * J * charges[:, None] * charges[None, :]
    roughness_matrix = jnp.where(correct_mask, 0.0, V ** 2)
    n_nonnative      = jnp.maximum(jnp.sum(~correct_mask), 1.0)
    sigma2_mean      = jnp.sum(roughness_matrix) / n_nonnative
    return sigma2_mean, roughness_matrix
