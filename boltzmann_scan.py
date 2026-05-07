#!/usr/bin/env python3
"""
boltzmann_scan.py
-----------------
Sweep (phi, J, n_species, n_types, pH) across a Cartesian grid and write
the thermodynamic Boltzmann equilibrium correct-dimer fraction to a CSV.

Each row records the state of the CRN at thermodynamic equilibrium for a
specific parameter combination.  No ODE integration is performed — the
equilibrium is solved directly from the Boltzmann distribution via a
damped fixed-point iteration (same method used for the dotted reference
lines in summary.png).

Columns written
---------------
  phi, J, n_species, n_types, N, n_correct_pairs
  pH, pKa_acid, pKa_base, beta
  sum_correct      : Σ correct-dimer concentrations at equilibrium
  sum_incorrect    : Σ incorrect-dimer concentrations at equilibrium
  correct_fraction : 2·sum_correct  (= fraction of monomer content in correct dimers)
  selectivity      : sum_correct / (sum_correct + sum_incorrect)
                     (fraction of ALL dimers that are correct)

Usage
-----
    python boltzmann_scan.py                          # defaults
    python boltzmann_scan.py --outfile custom.csv
    python boltzmann_scan.py --phi 0.0 0.1 0.5 1.0  # custom phi grid
    python boltzmann_scan.py --J 1.0 2.0 3.0
    python boltzmann_scan.py --pKa_acid 5.5 --pKa_base 8.5 --beta 1.0
"""

import argparse
import csv
import itertools
import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from crn_ad.physics import henderson_hasselbalch, interaction_energy_matrix
from crn_ad.dynamics import make_triu_indices


# ---------------------------------------------------------------------------
# Boltzmann fixed-point solver
# (self-contained copy — avoids importing matplotlib via crn_ad.visualize)
# ---------------------------------------------------------------------------

def boltzmann_equilibrium(pH, pKa_full, acid_base, correct_mask, phi, J,
                           beta=1.0, n_iter=3000, tol=1e-12):
    """
    Mean-field Boltzmann equilibrium at fixed pH.

    Solves  x_i = C_i / (1 + (K·x)_i + K_ii·x_i)  via damped fixed-point
    iteration (x ← ½(x + F(x))), which converges robustly even when
    equilibrium constants K are large.

    C_i = 1/N for all species i (uniform initial monomer content; total M = 1).

    Returns
    -------
    (sum_correct_dimers, sum_incorrect_dimers)
        Raw dimer concentrations summed over correct / incorrect upper-triangle
        entries.  correct_fraction = 2·sum_correct (since M = 1).
    """
    n = len(pKa_full)
    i_idx, j_idx = make_triu_indices(n)
    charges = np.array(henderson_hasselbalch(
        jnp.array(pKa_full, dtype=float), float(pH),
        jnp.array(acid_base, dtype=float)))
    dG = np.array(interaction_energy_matrix(
        jnp.array(charges, dtype=float),
        jnp.array(correct_mask, dtype=bool),
        float(phi), float(J)))
    K = np.exp(-float(beta) * dG)
    C = np.ones(n) / n
    x = C.copy()
    K_diag = np.diag(K).copy()
    for _ in range(n_iter):
        Fx = C / (1.0 + K.dot(x) + K_diag * x)
        x_new = 0.5 * (x + Fx)
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    d_correct = d_incorrect = 0.0
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        d_k = float(K[ii, jj]) * float(x[ii]) * float(x[jj])
        if correct_mask[ii, jj]:
            d_correct += d_k
        else:
            d_incorrect += d_k
    return d_correct, d_incorrect


# ---------------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------------

def build_system(n_species, n_types, pKa_acid=6.0, pKa_base=8.0):
    """
    Construct (pKa_full, acid_base, correct_mask) for an n_species × n_types CRN.

    Species layout (same convention as main.py):
        Particle k → species s = k // n_types
        s even → acid-like (charge < 0 at high pH)
        s odd  → base-like (charge > 0 at low pH)

    Correct bonds: same (acid, base) species pair AND same type index t.
        (A1–B1), (A2–B2), ..., (C1–D1), (C2–D2), ...

    Parameters
    ----------
    n_species : int   must be even, ≥ 2
    n_types   : int   ≥ 1
    pKa_acid  : float pKa for acid-like species
    pKa_base  : float pKa for base-like species
    """
    N = n_species * n_types
    acid_base = np.array([(k // n_types) % 2 for k in range(N)], dtype=int)
    pKa_full = np.where(acid_base == 0, pKa_acid, pKa_base).astype(float)
    correct_mask = np.zeros((N, N), dtype=bool)
    for pair_idx in range(n_species // 2):
        for t in range(n_types):
            i = 2 * pair_idx * n_types + t
            j = (2 * pair_idx + 1) * n_types + t
            correct_mask[i, j] = correct_mask[j, i] = True
    return pKa_full, acid_base, correct_mask


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_scan(phi_vals, J_vals, n_species_vals, n_types_vals, pH_vals,
             pKa_acid=6.0, pKa_base=8.0, beta=1.0):
    """Full Cartesian product sweep.  Returns list of row dicts."""
    combos = list(itertools.product(
        n_species_vals, n_types_vals, pH_vals, phi_vals, J_vals))
    total = len(combos)
    print(f"Running {total:,} combinations ...")
    rows = []
    for done, (n_species, n_types, pH, phi, J) in enumerate(combos, 1):
        pKa_full, acid_base, correct_mask = build_system(
            n_species, n_types, pKa_acid, pKa_base)
        d_c, d_i = boltzmann_equilibrium(
            pH, pKa_full, acid_base, correct_mask, phi, J, beta)
        total_dimer = d_c + d_i
        rows.append({
            'phi'             : round(float(phi), 6),
            'J'               : round(float(J),   6),
            'n_species'       : int(n_species),
            'n_types'         : int(n_types),
            'N'               : int(n_species * n_types),
            'n_correct_pairs' : int(n_species // 2 * n_types),
            'pH'              : round(float(pH),   4),
            'pKa_acid'        : float(pKa_acid),
            'pKa_base'        : float(pKa_base),
            'beta'            : float(beta),
            'sum_correct'     : d_c,
            'sum_incorrect'   : d_i,
            'correct_fraction': 2.0 * d_c,     # = 2·sum_correct / M, M≡1
            'selectivity'     : d_c / (total_dimer + 1e-15),
        })
        if done % 1000 == 0 or done == total:
            print(f"  {done:,}/{total:,}", end='\r', flush=True)
    print()
    return rows


def main():
    p = argparse.ArgumentParser(
        description='Sweep CRN Boltzmann equilibrium over parameter grids.')
    p.add_argument('--phi', type=float, nargs='+',
                   default=list(np.round(np.linspace(0.0, 1.0, 21), 4)),
                   help='phi values  [default: 21 pts in [0, 1]]')
    p.add_argument('--J', type=float, nargs='+',
                   default=list(np.round(np.linspace(0.1, 20.0, 20), 4)),
                   help='J values (kT)  [default: 20 pts in [0.5, 5.0]]')
    p.add_argument('--n_species', type=int, nargs='+',
                   default=[2, 4, 6, 8, 10],
                   help='n_species values (must be even ≥ 2)  [default: 2 4 6 8 10]')
    p.add_argument('--n_types', type=int, nargs='+',
                   default=[1, 2, 3, 4, 5],
                   help='n_types values  [default: 1 2 3 4 5]')
    p.add_argument('--pH', type=float, nargs='+',
                   default=[5.0, 7.0, 9.0],
                   help='pH values  [default: 5 7 9]')
    p.add_argument('--pKa_acid', type=float, default=6.0,
                   help='pKa for acid-like species  [default: 6.0]')
    p.add_argument('--pKa_base', type=float, default=8.0,
                   help='pKa for base-like species  [default: 8.0]')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Inverse temperature β  [default: 1.0]')
    p.add_argument('--outfile', default='boltzmann_scan.csv',
                   help='Output CSV filename  [default: boltzmann_scan.csv]')
    args = p.parse_args()

    for ns in args.n_species:
        if ns % 2 != 0 or ns < 2:
            p.error(f'n_species must be even and ≥ 2, got {ns}')
    for nt in args.n_types:
        if nt < 1:
            p.error(f'n_types must be ≥ 1, got {nt}')

    print("CRN Boltzmann Parameter Sweep")
    print(f"  phi       : {len(args.phi)} values in [{min(args.phi):.3g}, {max(args.phi):.3g}]")
    print(f"  J         : {len(args.J)} values in [{min(args.J):.3g}, {max(args.J):.3g}] kT")
    print(f"  n_species : {args.n_species}")
    print(f"  n_types   : {args.n_types}")
    print(f"  pH        : {args.pH}")
    print(f"  pKa_acid={args.pKa_acid}  pKa_base={args.pKa_base}  beta={args.beta}")

    rows = run_scan(args.phi, args.J, args.n_species, args.n_types, args.pH,
                    pKa_acid=args.pKa_acid, pKa_base=args.pKa_base,
                    beta=args.beta)

    fieldnames = list(rows[0].keys())
    with open(args.outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.outfile}  ({len(rows):,} rows, {len(fieldnames)} columns)")


if __name__ == '__main__':
    main()
