#!/usr/bin/env python3
"""
scan.py — Parameter sweep for CRN_AD.

Evaluates correct-dimer yield (target-schedule score) across the Cartesian
product of one or more swept parameter lists, then writes a labelled CSV and
optionally plots the results.

Example
-------
  python scan.py \
      --n_species 4 \
      --target_pH 9 5 7 \
      --duration 10 20 30 \
      --eval_phi 0.1 0.3 \
      --eval_J   2.0 3.5 \
      --eval_pKa 8.0 7.0 6.0 5.0 \
      --outdir   scan_outputs

Swept parameters (accept one or more values)
----------------------------------------------
  --duration, --equil_duration,
  --eval_phi, --eval_J, --eval_pKa (repeating block per species),
  --S_max, --beta, --k0

Fixed structural parameters (single values, define JIT-cache key)
-------------------------------------------------------------------
  --n_species, --n_types, --target_pH, --smooth_width,
  --specific_bonds, --n_points_sim, --n_points_equil

Outputs
-------
  <outdir>/scan_results.csv   — one row per parameter combination
  <outdir>/scan_plot.png      — bar / scatter overview plot
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crn_ad.dynamics  import (simulate_schedule_scan, make_initial_state,
                               make_triu_indices)
from crn_ad.training  import correct_bond_score, all_unique_permutations
from crn_ad.physics   import henderson_hasselbalch


# ---------------------------------------------------------------------------
# JIT-function cache  (keyed on structural parameters)
# ---------------------------------------------------------------------------

_FN_CACHE: dict = {}


def _build_static(n_species, T, smooth_width, specific_bonds,
                  n_pts_sim, n_pts_equil, target_sched):
    """Build the structural (non-swept) parts of the problem."""
    N = n_species * T
    acid_base_np = np.array([(k // T) % 2 for k in range(N)], dtype=int)

    correct_mask_np = np.zeros((N, N), dtype=bool)
    for pair_idx in range(n_species // 2):
        for t in range(T):
            i = 2 * pair_idx * T + t
            j = (2 * pair_idx + 1) * T + t
            correct_mask_np[i, j] = True
            correct_mask_np[j, i] = True

    species_pair_mask_np = np.zeros((N, N), dtype=bool)
    for pair_idx in range(n_species // 2):
        for t1 in range(T):
            for t2 in range(T):
                i = 2 * pair_idx * T + t1
                j = (2 * pair_idx + 1) * T + t2
                species_pair_mask_np[i, j] = True
                species_pair_mask_np[j, i] = True

    i_idx, j_idx = make_triu_indices(N)
    correct_triu_idx = np.array([
        pos for pos, (ii, jj) in enumerate(zip(i_idx, j_idx))
        if correct_mask_np[ii, jj]
    ])
    allowed_mask = jnp.array(species_pair_mask_np) if specific_bonds else None

    all_scheds   = all_unique_permutations(target_sched)
    target_idx   = all_scheds.index(target_sched)
    all_pH_array = jnp.array(all_scheds, dtype=float)

    return {
        'N'               : N,
        'acid_base'       : jnp.array(acid_base_np),
        'correct_mask'    : jnp.array(correct_mask_np),
        'i_idx'           : i_idx,
        'j_idx'           : j_idx,
        'correct_triu_idx': jnp.array(correct_triu_idx),
        'allowed_mask'    : allowed_mask,
        'n_pts_sim'       : n_pts_sim,
        'n_pts_equil'     : n_pts_equil,
        'smooth_width'    : smooth_width,
        'all_pH_array'    : all_pH_array,
        'target_idx'      : target_idx,
        'all_scheds'      : all_scheds,
    }


def _get_jit_fn(cache_key, st, has_entropy, duration_py, equil_duration_py,
                equil_ramp_py, beta_py, k0_py):
    """
    Return (and cache) a JIT-compiled function:

        fn(pKa_full, phi, J, mono_s)  →  scores array  (n_schedules,)

    Numeric scalars (duration, equil_duration, equil_ramp, beta, k0) are
    baked into the compiled function as Python-level constants so that
    simulate_schedule_scan receives plain Python floats for its internal
    Python-if decisions (e.g. beta-ramp branch selection).  Different values
    of these scalars produce different cache entries and compile separately.
    """
    if cache_key in _FN_CACHE:
        return _FN_CACHE[cache_key]

    sw            = float(st['smooth_width'])
    initial_state = make_initial_state(st['N'])
    allowed_mask  = st['allowed_mask']
    all_pH_array  = st['all_pH_array']
    n             = st['N']
    acid_base     = st['acid_base']
    correct_mask  = st['correct_mask']
    i_idx         = st['i_idx']
    j_idx         = st['j_idx']
    correct_tidx  = st['correct_triu_idx']
    n_pts_sim     = st['n_pts_sim']
    n_pts_equil   = st['n_pts_equil']

    # Capture all Python-float scalars in the closure.
    _dur      = float(duration_py)
    _equil    = float(equil_duration_py)
    _eramp    = float(equil_ramp_py)
    _beta     = float(beta_py)
    _k0       = float(k0_py)
    _mono_kw  = dict(monomer_entropy=None)

    if has_entropy:
        @jax.jit
        def _fn(pKa_full, phi, J, mono_s):
            equil = simulate_schedule_scan(
                initial_state, jnp.array([7.0]), _equil,
                pKa_full, acid_base, phi, J,
                _beta, _k0, correct_mask, n, i_idx, j_idx,
                n_points=n_pts_equil, smooth_width=sw,
                monomer_entropy=mono_s, ph_initial=7.0,
                allowed_mask=allowed_mask, beta_ramp_duration=_eramp,
            )
            def score_one(pH_sched):
                final = simulate_schedule_scan(
                    equil, pH_sched, _dur,
                    pKa_full, acid_base, phi, J,
                    _beta, _k0, correct_mask, n, i_idx, j_idx,
                    n_points=n_pts_sim, smooth_width=sw,
                    monomer_entropy=mono_s, allowed_mask=allowed_mask,
                )
                return correct_bond_score(final, n, correct_tidx)
            return jax.vmap(score_one)(all_pH_array)
    else:
        @jax.jit
        def _fn(pKa_full, phi, J, mono_s):
            equil = simulate_schedule_scan(
                initial_state, jnp.array([7.0]), _equil,
                pKa_full, acid_base, phi, J,
                _beta, _k0, correct_mask, n, i_idx, j_idx,
                n_points=n_pts_equil, smooth_width=sw,
                monomer_entropy=None, ph_initial=7.0,
                allowed_mask=allowed_mask, beta_ramp_duration=_eramp,
            )
            def score_one(pH_sched):
                final = simulate_schedule_scan(
                    equil, pH_sched, _dur,
                    pKa_full, acid_base, phi, J,
                    _beta, _k0, correct_mask, n, i_idx, j_idx,
                    n_points=n_pts_sim, smooth_width=sw,
                    monomer_entropy=None, allowed_mask=allowed_mask,
                )
                return correct_bond_score(final, n, correct_tidx)
            return jax.vmap(score_one)(all_pH_array)

    _FN_CACHE[cache_key] = _fn
    return _fn


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(args):
    os.makedirs(args.outdir, exist_ok=True)

    target_sched = [float(x) for x in args.target_pH]
    n_species    = args.n_species

    # --- Build lists of pKa blocks ---
    # --eval_pKa can be:
    #   (a) n_species values  → fixed pKa across sweep
    #   (b) k * n_species values  → k different pKa configurations to sweep
    pKa_raw = args.eval_pKa
    if len(pKa_raw) % n_species != 0:
        print(f'ERROR: --eval_pKa length ({len(pKa_raw)}) must be a multiple '
              f'of --n_species ({n_species}).')
        sys.exit(1)
    pKa_blocks = [pKa_raw[i*n_species:(i+1)*n_species]
                  for i in range(len(pKa_raw) // n_species)]

    # --- Sweep axes ---
    sweep_axes = {
        'pKa'           : pKa_blocks,
        'phi'           : args.eval_phi,
        'J'             : args.eval_J,
        'S_max'         : args.S_max_list,
        'duration'      : args.duration,
        'equil_duration': args.equil_duration,
        'beta'          : args.beta_list,
        'k0'            : args.k0_list,
        'n_types'       : args.n_types_list,
    }

    # Count combinations
    n_combos = 1
    for v in sweep_axes.values():
        n_combos *= len(v)
    print(f'Sweep: {n_combos} combinations')
    print(f'  n_species={n_species}, target_pH={target_sched}')
    for k, v in sweep_axes.items():
        print(f'  {k}: {v}')
    print()

    # CSV setup
    csv_path    = os.path.join(args.outdir, 'scan_results.csv')
    field_names = ['pKa', 'phi', 'J', 'S_max', 'duration', 'equil_duration',
                   'beta', 'k0', 'n_types',
                   'target_score', 'best_other_score', 'selectivity']
    all_rows    = []

    t0 = time.time()
    total_done = 0

    for (pKa_vals, phi, J, S_max, dur, equil_dur, beta, k0, T) in itertools.product(
            sweep_axes['pKa'],
            sweep_axes['phi'],
            sweep_axes['J'],
            sweep_axes['S_max'],
            sweep_axes['duration'],
            sweep_axes['equil_duration'],
            sweep_axes['beta'],
            sweep_axes['k0'],
            sweep_axes['n_types']):

        T   = int(T)
        phi = float(phi)
        J   = float(J)

        # Build / reuse the static structure for this (n_species, T, …)
        st_key = (n_species, T, tuple(target_sched),
                  args.smooth_width, args.specific_bonds,
                  args.n_points_sim, args.n_points_equil)

        # Scalar physics values are baked into each compiled function.
        equil_ramp = float(equil_dur) / 2.0
        cache_key = st_key + (S_max > 0.0, float(dur), float(equil_dur),
                               equil_ramp, float(beta), float(k0))

        if not hasattr(run_sweep, '_st_cache'):
            run_sweep._st_cache = {}
        if st_key not in run_sweep._st_cache:
            run_sweep._st_cache[st_key] = _build_static(
                n_species, T, args.smooth_width, args.specific_bonds,
                args.n_points_sim, args.n_points_equil, target_sched)
        st = run_sweep._st_cache[st_key]

        has_entropy = S_max > 0.0
        fn = _get_jit_fn(cache_key, st, has_entropy,
                         dur, equil_dur, equil_ramp, beta, k0)

        # Build JAX inputs (only the truly swept arrays)
        pKa_np   = np.array(pKa_vals, dtype=float)
        pKa_full = jnp.repeat(jnp.array(pKa_np), T) if T > 1 else jnp.array(pKa_np)
        mono_s   = jnp.array([S_max]) if has_entropy else jnp.zeros(1)

        # Run (triggers JIT compilation on first call for this cache_key)
        if total_done == 0:
            print(f'  Compiling JIT for cache_key={cache_key[:5]}... '
                  f'dur={dur} equil={equil_dur} β={beta} k0={k0}', flush=True)

        scores = np.array(fn(
            pKa_full,
            jnp.array(phi),
            jnp.array(J),
            mono_s,
        ))

        target_idx   = st['target_idx']
        target_score = float(scores[target_idx])
        other_scores = [scores[i] for i in range(len(scores)) if i != target_idx]
        best_other   = float(max(other_scores)) if other_scores else 0.0
        selectivity  = target_score - best_other

        row = {
            'pKa'           : str(list(pKa_vals)),
            'phi'           : phi,
            'J'             : J,
            'S_max'         : S_max,
            'duration'      : dur,
            'equil_duration': equil_dur,
            'beta'          : beta,
            'k0'            : k0,
            'n_types'       : T,
            'target_score'  : round(target_score, 6),
            'best_other_score': round(best_other, 6),
            'selectivity'   : round(selectivity, 6),
        }
        all_rows.append(row)
        total_done += 1

        elapsed = time.time() - t0
        eta_str = ''
        if total_done > 1:
            eta = elapsed / total_done * (n_combos - total_done)
            eta_str = f'  ETA {eta:.0f}s'
        print(f'  [{total_done}/{n_combos}] '
              f'pKa={[round(x,2) for x in pKa_vals]} '
              f'phi={phi:.2f} J={J:.2f} S={S_max:.2f} '
              f'dur={dur:.0f} β={beta:.2f} k0={k0:.2f} T={T}'
              f'  →  target={target_score:.4f}  select={selectivity:.4f}'
              + eta_str, flush=True)

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nResults → {csv_path}')

    return all_rows, csv_path


# ---------------------------------------------------------------------------
# Simple result plot
# ---------------------------------------------------------------------------

def plot_results(all_rows, outdir, sweep_axes):
    """Produce a summary scatter / bar plot of target_score vs each swept axis."""
    import pandas as pd

    df = pd.DataFrame(all_rows)

    # Identify axes that actually vary
    varying = [k for k in ['phi', 'J', 'S_max', 'duration', 'equil_duration',
                            'beta', 'k0', 'n_types']
               if df[k].nunique() > 1]
    # pKa is a string column; check if it varies
    if df['pKa'].nunique() > 1:
        varying.insert(0, 'pKa')

    if not varying:
        print('No parameters vary — skipping plot.')
        return

    ncols = min(3, len(varying))
    nrows = (len(varying) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle('CRN scan — target score vs swept parameters', fontsize=13)

    for idx, param in enumerate(varying):
        ax = axes[idx // ncols][idx % ncols]
        if param == 'pKa':
            # Use string labels
            groups = df.groupby('pKa')['target_score']
            labels = list(groups.groups.keys())
            means  = [groups.get_group(l).mean() for l in labels]
            x_pos  = range(len(labels))
            ax.bar(x_pos, means, color='#3498db', alpha=0.8)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=7)
        else:
            x = df[param].astype(float)
            sc = ax.scatter(x, df['target_score'],
                            c=df['selectivity'], cmap='RdYlGn',
                            vmin=-0.5, vmax=0.5, edgecolors='none', s=30)
            plt.colorbar(sc, ax=ax, label='selectivity')
        ax.set_xlabel(param)
        ax.set_ylabel('target score')
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color='grey', lw=0.7, ls='--')
        ax.grid(True, alpha=0.3)

    # Hide empty axes
    for idx in range(len(varying), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(outdir, 'scan_plot.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f'Plot      → {plot_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='CRN_AD parameter sweep — evaluate correct-dimer yield '
                    'across a Cartesian product of parameter lists.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Fixed structural parameters
    p.add_argument('--n_species',    type=int,   default=4)
    p.add_argument('--target_pH',    nargs='+',  type=float, default=[9.0, 5.0, 7.0])
    p.add_argument('--smooth_width', type=float, default=0.0)
    p.add_argument('--specific_bonds', action='store_true')
    p.add_argument('--n_points_sim',   type=int, default=40)
    p.add_argument('--n_points_equil', type=int, default=60)
    p.add_argument('--outdir', default='scan_outputs')
    p.add_argument('--no_plot', action='store_true',
                   help='Skip generating the summary plot.')

    # Swept parameters — accept one or more values
    p.add_argument('--n_types', dest='n_types_list',
                   nargs='+', type=int,   default=[1],
                   metavar='T',
                   help='Types per species (structural; triggers recompilation per value).')
    p.add_argument('--duration', nargs='+', type=float, default=[30.0],
                   metavar='D',
                   help='Duration per pH segment (time units).')
    p.add_argument('--equil_duration', nargs='+', type=float, default=[80.0],
                   metavar='E',
                   help='Pre-equilibration duration at pH 7.')
    p.add_argument('--eval_pKa', nargs='+', type=float, required=True,
                   metavar='V',
                   help='pKa values.  Must be a multiple of --n_species.  '
                        'Multiple blocks of n_species values are swept.')
    p.add_argument('--eval_phi', nargs='+', type=float, default=[0.1],
                   metavar='P',
                   help='Steric mismatch factor φ ∈ [0, 1].')
    p.add_argument('--eval_J', nargs='+', type=float, default=[2.0],
                   metavar='J',
                   help='Electrostatic coupling J (kT), scalar.')
    p.add_argument('--S_max', dest='S_max_list',
                   nargs='+', type=float, default=[0.0],
                   metavar='S',
                   help='Monomer entropy (kT).  0 = disabled.')
    p.add_argument('--beta', dest='beta_list',
                   nargs='+', type=float, default=[1.0],
                   metavar='B',
                   help='Inverse temperature β.')
    p.add_argument('--k0', dest='k0_list',
                   nargs='+', type=float, default=[1.0],
                   metavar='K',
                   help='Base rate constant k₀.')
    return p


def main():
    args = build_parser().parse_args()

    all_rows, csv_path = run_sweep(args)

    if not args.no_plot:
        try:
            import pandas
            sweep_axes = {
                'phi'           : args.eval_phi,
                'J'             : args.eval_J,
                'S_max'         : args.S_max_list,
                'duration'      : args.duration,
                'equil_duration': args.equil_duration,
                'beta'          : args.beta_list,
                'k0'            : args.k0_list,
                'n_types'       : args.n_types_list,
            }
            plot_results(all_rows, args.outdir, sweep_axes)
        except ImportError:
            print('(pandas not installed — skipping plot)')


if __name__ == '__main__':
    main()
