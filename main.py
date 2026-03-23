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
                      n_points_equil=60, equil_duration=200.0, tau=5.0):
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
    }


def get_equil_and_schedule_traj(p, static, target_sched, duration):
    """
    Run pH-7 equilibration then target schedule; return trajectories.

    Returns
    -------
    equil_traj     : array (n_pts, state_dim)
    schedule_trajs : list of arrays (n_pts, state_dim), one per segment
    final_state    : last state
    """
    n = static['n']
    initial = make_initial_state(n)

    # --- Equilibration ---
    equil_final, equil_traj = simulate_schedule(
        initial,
        [7.0],
        static['equil_duration'],
        jnp.array(p['pKa']), static['acid_base'],
        jnp.array(p['phi']), jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n,
        static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
    )
    equil_traj = equil_traj[0]   # single segment

    # --- Schedule ---
    final_state, schedule_trajs = simulate_schedule(
        equil_final,
        target_sched,
        duration,
        jnp.array(p['pKa']), static['acid_base'],
        jnp.array(p['phi']), jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n,
        static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
    )
    return equil_traj, schedule_trajs, final_state


def compute_all_scores(p, static, all_schedules, duration):
    """Run all schedule permutations and return scores array."""
    n            = static['n']
    initial      = make_initial_state(n)
    scores       = []
    pKa          = jnp.array(p['pKa'])
    phi          = jnp.array(p['phi'])
    J            = jnp.array(p['J'])

    # Shared equilibrium state
    equil_final, _ = simulate_schedule(
        initial, [7.0], static['equil_duration'],
        pKa, static['acid_base'], phi, J,
        static['beta'], static['k0'],
        static['correct_mask'], n, static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
    )
    equil_final = equil_final

    for sched in all_schedules:
        final, _ = simulate_schedule(
            equil_final, sched, duration,
            pKa, static['acid_base'], phi, J,
            static['beta'], static['k0'],
            static['correct_mask'], n, static['i_idx'], static['j_idx'],
            n_points=static['n_points_sim'],
        )
        scores.append(float(correct_bond_score(
            final, n, static['correct_triu_idx']
        )))
    return np.array(scores)


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
        )

        print('=' * 60)
        print('CRN_AD  —  Training')
        print('=' * 60)

        result = train(config)
        (raw_params, loss_history, score_history, param_history,
         static, all_schedules, target_idx, equil_state) = result

        p = constrain_params(raw_params)
        # Save params
        params_out = {
            'n_species'          : args.n_species,
            'target_pH_schedule' : args.target_pH,
            'pKa'                : np.array(p['pKa']).tolist(),
            'phi'                : float(p['phi']),
            'J'                  : float(p['J']),
            'beta'               : args.beta,
            'k0'                 : args.k0,
        }
        with open(params_path, 'w') as f:
            json.dump(params_out, f, indent=2)
        print(f'Saved trained params → {params_path}')

        # Print summary
        print('\n— Trained parameters —')
        for i in range(args.n_species):
            ab = 'base' if int(static['acid_base'][i]) == 1 else 'acid'
            print(f'  {SPECIES_NAMES[i]} ({ab:<4s}): pKa = {float(p["pKa"][i]):.3f}')
        print(f'  φ = {float(p["phi"]):.4f}')
        print(f'  J = {float(p["J"]):.4f}  kT')

    # =====================================================================
    # Load params if animate-only mode
    # =====================================================================
    if args.mode == 'animate':
        if not os.path.exists(params_path):
            print(f'ERROR: params file not found: {params_path}')
            sys.exit(1)
        with open(params_path) as f:
            pdata = json.load(f)
        static = static_from_saved(
            pdata, pdata['beta'], pdata['k0'],
            args.n_points_sim, args.n_points_equil,
            args.equil_duration, args.tau,
        )
        p = {'pKa': pdata['pKa'], 'phi': pdata['phi'], 'J': pdata['J']}
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
    # SUMMARY PLOT (always produced after training or in animate mode)
    # =====================================================================
    if args.mode in ('train', 'both', 'animate'):
        print('\nGenerating summary plot ...')

        # Run target schedule trajectory for concentration panel
        equil_traj, schedule_trajs, final_state = get_equil_and_schedule_traj(
            {'pKa': np.array(p['pKa']) if args.mode == 'animate'
                    else np.array(constrain_params(raw_params)['pKa']),
             'phi': float(p['phi']) if args.mode == 'animate'
                    else float(constrain_params(raw_params)['phi']),
             'J':   float(p['J']) if args.mode == 'animate'
                    else float(constrain_params(raw_params)['J'])},
            static, target_sched, args.duration,
        )

        # Scores for all schedules
        print('  Scoring all schedule permutations ...')
        p_eval = constrain_params(raw_params) if args.mode != 'animate' else \
                 {'pKa': p['pKa'], 'phi': p['phi'], 'J': p['J']}
        final_scores = compute_all_scores(
            p_eval, static, all_schedules, args.duration
        )

        # If we just trained, use the already-recorded param/score history
        if args.mode == 'animate':
            loss_history  = [0.0]
            score_history = [final_scores]
            param_history = [{'pKa': np.array(p_eval['pKa']),
                               'phi': float(p_eval['phi']),
                               'J':   float(p_eval['J'])}]

        # Trained params dict (constrained, numpy)
        trained_params = {
            'pKa': np.array(p_eval['pKa']),
            'phi': float(p_eval['phi']),
            'J':   float(p_eval['J']),
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
