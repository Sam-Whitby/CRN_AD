#!/usr/bin/env python3
"""
CRN_AD — main entry point.

Default behaviour
-----------------
  python main.py

  Trains a 4-species CRN to fold correctly only under the target schedule
  [9, 5, 7], evaluates all 6 permutations, and saves a single summary PNG
  which is then opened automatically.  No animation is produced by default
  (pass --animate to enable GIFs).

Usage examples
--------------
  python main.py                              # default 4-species run
  python main.py --n_species 6 --n_epochs 400
  python main.py --target_pH 9 5 7
  python main.py --animate                    # also produce animated GIFs
  python main.py --mode animate               # load saved params + make plots
"""

import argparse
import json
import os
import platform
import subprocess
import sys

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crn_ad.training  import (train, constrain_params, all_unique_permutations,
                               correct_bond_score, total_monomer_content)
from crn_ad.dynamics  import (simulate_schedule, simulate_schedule_scan,
                               make_initial_state, make_triu_indices)
from crn_ad.visualize import (plot_summary, animate_crn,
                               plot_final_concentrations, SPECIES_NAMES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='CRN_AD: pH-responsive Chemical Reaction Network trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mode', choices=['train', 'animate', 'both'],
                   default='train',
                   help='What to run.  Use "both" to also generate animations.')
    p.add_argument('--n_species', type=int, default=4,
                   help='Number of species (even, max 10).')
    p.add_argument('--target_pH', nargs='+', type=float,
                   default=[9.0, 5.0, 7.0],
                   help='Target pH schedule, one value per segment.')
    p.add_argument('--duration', type=float, default=30.0,
                   help='Duration of each pH segment (time units).')
    p.add_argument('--equil_duration', type=float, default=80.0,
                   help='Duration of pH-7 pre-equilibration (time units).')
    p.add_argument('--n_epochs', type=int, default=300,
                   help='Training epochs.')
    p.add_argument('--lr', type=float, default=0.02,
                   help='Adam learning rate.')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Inverse temperature β = 1/k_BT.')
    p.add_argument('--k0', type=float, default=1.0,
                   help='Base rate constant k_0.')
    p.add_argument('--tau', type=float, default=6.0,
                   help='Softmax temperature for loss.')
    p.add_argument('--n_points_sim', type=int, default=40,
                   help='ODE time points per schedule segment.')
    p.add_argument('--n_points_equil', type=int, default=60,
                   help='ODE time points for pH-7 equilibration.')
    p.add_argument('--outdir', type=str, default='outputs',
                   help='Directory for all output files.')
    p.add_argument('--params_file', type=str, default='trained_params.json',
                   help='JSON file for saving/loading parameters.')
    p.add_argument('--animate', action='store_true',
                   help='Also generate animated GIFs (requires Pillow).')
    p.add_argument('--seed', type=int, default=42)
    # New flags
    p.add_argument('--J_max', type=float, default=3.5,
                   help='Hard upper cap on J (kT).  Increase carefully — large J '
                        'makes the ODE stiff.  Smooth-pH mode (--smooth_width) '
                        'helps stability when J_max > 3.5.')
    p.add_argument('--smooth_width', type=float, default=0.0,
                   help='If > 0, pH transitions are smoothed with a logistic sigmoid '
                        'over this many time units.  Reduces adjoint NaNs caused by '
                        'RHS discontinuities at segment boundaries.  Try 1–3.')
    p.add_argument('--S_max', type=float, default=0.0,
                   help='If > 0, add per-dimer conformational-entropy parameters '
                        'ΔS_ij ∈ [0, S_max] (kT) that are optimised by autodiff. '
                        'Positive ΔS makes dimerisation less favourable.')
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_file(path):
    """Open a file with the system default viewer."""
    try:
        if platform.system() == 'Darwin':
            subprocess.Popen(['open', path])
        elif platform.system() == 'Linux':
            subprocess.Popen(['xdg-open', path])
        elif platform.system() == 'Windows':
            os.startfile(path)
    except Exception:
        pass


def static_from_saved(pdata, beta, k0, n_points_sim=40,
                      n_points_equil=60, equil_duration=80.0, tau=5.0,
                      J_max=3.5, S_max=0.0, smooth_width=0.0):
    n = pdata['n_species']
    acid_base_np    = np.array([i % 2 for i in range(n)], dtype=int)
    correct_mask_np = np.zeros((n, n), dtype=bool)
    for k in range(n // 2):
        i, j = 2 * k, 2 * k + 1
        correct_mask_np[i, j] = True
        correct_mask_np[j, i] = True
    i_idx, j_idx = make_triu_indices(n)
    correct_triu_idx = np.array([
        pos for pos, (ii, jj) in enumerate(zip(i_idx, j_idx))
        if correct_mask_np[ii, jj]
    ])
    return {
        'n'               : n,
        'acid_base'       : jnp.array(acid_base_np),
        'acid_base_np'    : acid_base_np,
        'correct_mask'    : jnp.array(correct_mask_np),
        'correct_mask_np' : correct_mask_np,
        'i_idx'           : i_idx,
        'j_idx'           : j_idx,
        'correct_triu_idx': jnp.array(correct_triu_idx),
        'beta'            : float(beta),
        'k0'              : float(k0),
        'n_points_sim'    : int(n_points_sim),
        'n_points_equil'  : int(n_points_equil),
        'equil_duration'  : float(equil_duration),
        'tau'             : float(tau),
        'J_max'           : float(J_max),
        'S_max'           : float(S_max),
        'smooth_width'    : float(smooth_width),
    }


def get_equil_and_schedule_traj(p, static, target_sched, duration):
    """Run pH-7 equilibration then target schedule; return trajectories."""
    n            = static['n']
    initial      = make_initial_state(n)
    entropy_triu = _entropy_array(p, static)

    equil_final, equil_traj = simulate_schedule(
        initial, [7.0], static['equil_duration'],
        jnp.array(p['pKa']), static['acid_base'],
        jnp.array(p['phi']), jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n, static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        entropy_triu=entropy_triu,
    )
    equil_traj = equil_traj[0]

    final_state, schedule_trajs = simulate_schedule(
        equil_final, target_sched, duration,
        jnp.array(p['pKa']), static['acid_base'],
        jnp.array(p['phi']), jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n, static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        entropy_triu=entropy_triu,
    )
    return equil_traj, schedule_trajs, final_state


def compute_all_scores(p, static, all_schedules, duration):
    """Run all schedule permutations and return scores array."""
    n            = static['n']
    initial      = make_initial_state(n)
    pKa          = jnp.array(p['pKa'])
    phi          = jnp.array(p['phi'])
    J            = jnp.array(p['J'])
    entropy_triu = _entropy_array(p, static)

    equil_final, _ = simulate_schedule(
        initial, [7.0], static['equil_duration'],
        pKa, static['acid_base'], phi, J,
        static['beta'], static['k0'],
        static['correct_mask'], n, static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        entropy_triu=entropy_triu,
    )

    scores = []
    for sched in all_schedules:
        final, _ = simulate_schedule(
            equil_final, sched, duration,
            pKa, static['acid_base'], phi, J,
            static['beta'], static['k0'],
            static['correct_mask'], n, static['i_idx'], static['j_idx'],
            n_points=static['n_points_sim'],
            entropy_triu=entropy_triu,
        )
        scores.append(float(correct_bond_score(final, n, static['correct_triu_idx'])))
    return np.array(scores)


def _entropy_array(p, static):
    """Return entropy JAX array or None."""
    if static.get('S_max', 0.0) > 0.0 and 'entropy' in p and p['entropy'] is not None:
        return jnp.array(p['entropy'])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = build_parser().parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    params_path  = os.path.join(outdir, args.params_file)
    summary_path = os.path.join(outdir, 'summary.png')

    # =====================================================================
    # TRAIN
    # =====================================================================
    if args.mode in ('train', 'both'):
        config = dict(
            n_species          = args.n_species,
            target_pH_schedule = args.target_pH,
            duration_per_seg   = args.duration,
            equil_duration     = args.equil_duration,
            n_epochs           = args.n_epochs,
            learning_rate      = args.lr,
            beta               = args.beta,
            k0                 = args.k0,
            n_points_sim       = args.n_points_sim,
            n_points_equil     = args.n_points_equil,
            tau                = args.tau,
            seed               = args.seed,
            J_max              = args.J_max,
            smooth_width       = args.smooth_width,
            S_max              = args.S_max,
        )

        print('=' * 60)
        print('CRN_AD  —  Training')
        print('=' * 60)

        result = train(config)
        (raw_params, loss_history, score_history, param_history,
         static, all_schedules, target_idx, equil_state) = result

        p = constrain_params(raw_params, J_max=args.J_max, S_max=args.S_max)
        # Save params
        params_out = {
            'n_species'          : args.n_species,
            'target_pH_schedule' : args.target_pH,
            'pKa'                : np.array(p['pKa']).tolist(),
            'phi'                : float(p['phi']),
            'J'                  : float(p['J']),
            'beta'               : args.beta,
            'k0'                 : args.k0,
            'J_max'              : args.J_max,
            'S_max'              : args.S_max,
        }
        if args.S_max > 0.0 and 'entropy' in p:
            params_out['entropy'] = np.array(p['entropy']).tolist()
        with open(params_path, 'w') as f:
            json.dump(params_out, f, indent=2)
        print(f'Saved trained params → {params_path}')

        print('\n— Trained parameters —')
        for i in range(args.n_species):
            ab = 'base' if int(static['acid_base'][i]) == 1 else 'acid'
            print(f'  {SPECIES_NAMES[i]} ({ab:<4s}): pKa = {float(p["pKa"][i]):.3f}')
        print(f'  φ = {float(p["phi"]):.4f}')
        print(f'  J = {float(p["J"]):.4f}  kT')
        if args.S_max > 0.0 and 'entropy' in p:
            from crn_ad.dynamics import make_triu_indices as _mti
            _ii, _jj = _mti(args.n_species)
            for k, (ii, jj) in enumerate(zip(_ii, _jj)):
                print(f'  ΔS[{SPECIES_NAMES[ii]}–{SPECIES_NAMES[jj]}] = '
                      f'{float(p["entropy"][k]):.4f}  kT')

    # =====================================================================
    # Load params if animate-only mode
    # =====================================================================
    if args.mode == 'animate':
        if not os.path.exists(params_path):
            print(f'ERROR: params file not found: {params_path}')
            sys.exit(1)
        with open(params_path) as f:
            pdata = json.load(f)
        _J_max = float(pdata.get('J_max', args.J_max))
        _S_max = float(pdata.get('S_max', args.S_max))
        static = static_from_saved(
            pdata, pdata['beta'], pdata['k0'],
            args.n_points_sim, args.n_points_equil,
            args.equil_duration, args.tau,
            J_max=_J_max, S_max=_S_max,
            smooth_width=args.smooth_width,
        )
        p = {'pKa': pdata['pKa'], 'phi': pdata['phi'], 'J': pdata['J'],
             'entropy': pdata.get('entropy', None)}
        target_sched  = [float(x) for x in pdata['target_pH_schedule']]
        all_schedules = all_unique_permutations(target_sched)
        target_idx    = all_schedules.index(target_sched)
        loss_history  = []
        score_history = []
        param_history = [{'pKa': np.array(p['pKa']),
                          'phi': float(p['phi']), 'J': float(p['J'])}]
    else:
        target_sched = [float(x) for x in args.target_pH]

    # =====================================================================
    # SUMMARY PLOT
    # =====================================================================
    if args.mode in ('train', 'both', 'animate'):
        print('\nGenerating summary plot ...')

        if args.mode == 'animate':
            p_eval = p
        else:
            p_eval = constrain_params(raw_params, J_max=args.J_max, S_max=args.S_max)
            # Convert JAX arrays to plain numpy/python for downstream functions
            p_eval = {k: (np.array(v) if hasattr(v, 'shape') else float(v))
                      for k, v in p_eval.items()}

        equil_traj, schedule_trajs, _ = get_equil_and_schedule_traj(
            p_eval, static, target_sched, args.duration)

        print('  Scoring all schedule permutations ...')
        final_scores = compute_all_scores(p_eval, static, all_schedules, args.duration)

        if args.mode == 'animate':
            loss_history  = [0.0]
            score_history = [final_scores]
            param_history = [{'pKa': np.array(p_eval['pKa']),
                               'phi': float(p_eval['phi']),
                               'J':   float(p_eval['J'])}]

        trained_params = {
            'pKa'    : np.array(p_eval['pKa']),
            'phi'    : float(p_eval['phi']),
            'J'      : float(p_eval['J']),
            'entropy': (np.array(p_eval['entropy'])
                        if p_eval.get('entropy') is not None else None),
        }

        plot_summary(
            loss_history, score_history, param_history,
            all_schedules, target_idx,
            equil_traj, schedule_trajs, target_sched,
            args.equil_duration, args.duration,
            static, trained_params, final_scores,
            save_path=summary_path,
        )

        print(f'\nSummary plot → {summary_path}')
        open_file(summary_path)

        # Print final scores
        print('\n— Final scores (all schedules) —')
        for i, (sched, sc) in enumerate(zip(all_schedules, final_scores)):
            marker = ' ← TARGET' if i == target_idx else ''
            print(f'  {sched}  →  {sc:.4f}{marker}')

    # =====================================================================
    # OPTIONAL ANIMATIONS
    # =====================================================================
    if args.animate or args.mode == 'both':
        print('\nGenerating animations ...')
        n               = static['n']
        acid_base_np    = static['acid_base_np']
        correct_mask_np = static['correct_mask_np']
        p_anim          = trained_params if 'trained_params' in dir() else p_eval

        for s_idx, sched in enumerate([all_schedules[target_idx]] +
                                       [s for i, s in enumerate(all_schedules)
                                        if i != target_idx][:2]):
            label = 'target' if s_idx == 0 else f'perm{s_idx}'
            equil_t, sched_trajs, _ = get_equil_and_schedule_traj(
                p_anim, static, sched, args.duration
            )
            gif = os.path.join(outdir, f'animation_{label}.gif')
            try:
                all_trajs = [equil_t] + sched_trajs   # include equilibration in anim
                animate_crn(
                    all_trajs, n, acid_base_np, correct_mask_np,
                    [7.0] + list(sched),               # prepend pH-7 equil segment
                    args.duration,
                    pKa_visual=p_anim['pKa'],
                    output_path=gif,
                    fps=12,
                )
            except Exception as exc:
                print(f'  Warning: animation failed ({exc})')

    print('\nDone.')


if __name__ == '__main__':
    main()
