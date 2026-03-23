#!/usr/bin/env python3
"""
CRN_AD — main entry point.

Modes
-----
  train    : Train parameters from scratch and save results.
  animate  : Load saved parameters and generate animations.
  both     : Train then animate (default).

Quick start
-----------
  python main.py                          # 4-species, 4-segment default
  python main.py --n_species 6 --n_epochs 400
  python main.py --mode animate --outdir outputs

Animate any parameter file
--------------------------
  python main.py --mode animate --params_file outputs/trained_params.json
"""

import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crn_ad.training  import (train, constrain_params, all_unique_permutations,
                               correct_bond_score)
from crn_ad.dynamics  import (simulate_schedule, equilibrate_denatured,
                               make_triu_indices)
from crn_ad.visualize import (plot_loss_curve, animate_crn,
                               plot_final_concentrations,
                               plot_charges_vs_pH, SPECIES_NAMES)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='CRN_AD: pH-responsive Chemical Reaction Network trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mode', choices=['train', 'animate', 'both'],
                   default='both',
                   help='What to run.')
    p.add_argument('--n_species', type=int, default=4,
                   help='Number of species (even, max 10).')
    p.add_argument('--target_pH', nargs='+', type=float,
                   default=[4.0, 7.0, 9.0, 11.0],
                   help='Target pH schedule, one value per segment.')
    p.add_argument('--duration', type=float, default=80.0,
                   help='Duration of each pH segment (time units).')
    p.add_argument('--n_epochs', type=int, default=250,
                   help='Training epochs.')
    p.add_argument('--lr', type=float, default=0.05,
                   help='Adam learning rate.')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Inverse temperature β = 1/k_BT.')
    p.add_argument('--k0', type=float, default=1.0,
                   help='Base rate constant k_0.')
    p.add_argument('--tau', type=float, default=6.0,
                   help='Softmax temperature for loss.')
    p.add_argument('--n_points_sim', type=int, default=40,
                   help='ODE time points per segment (accuracy vs speed).')
    p.add_argument('--outdir', type=str, default='outputs',
                   help='Directory for all output files.')
    p.add_argument('--params_file', type=str, default='trained_params.json',
                   help='JSON file for saving/loading trained parameters.')
    p.add_argument('--no_animation', action='store_true',
                   help='Skip animation (faster if only plots are needed).')
    p.add_argument('--anim_fps', type=int, default=15,
                   help='Animation frames per second.')
    p.add_argument('--seed', type=int, default=42)
    return p


# ---------------------------------------------------------------------------
# Helper: rebuild static dict from saved params
# ---------------------------------------------------------------------------

def static_from_params(pdata, beta, k0, n_points_sim=40, tau=5.0):
    n = pdata['n_species']
    acid_base_np   = np.array([i % 2 for i in range(n)], dtype=int)
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
        'tau'             : float(tau),
    }


# ---------------------------------------------------------------------------
# Animate one schedule
# ---------------------------------------------------------------------------

def run_animation(label, pH_sched, duration, p, static, denatured_state,
                  outdir, fps, n_points_anim=60):
    n               = static['n']
    acid_base_np    = np.array(static['acid_base'])
    correct_mask_np = static['correct_mask_np']
    i_idx           = static['i_idx']
    j_idx           = static['j_idx']

    print(f'  Simulating schedule {pH_sched} ...', flush=True)
    final_state, traj_list = simulate_schedule(
        denatured_state, pH_sched, duration,
        jnp.array(p['pKa']),
        static['acid_base'],
        jnp.array(p['phi']),
        jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n, i_idx, j_idx,
        n_points=n_points_anim,
    )

    # Score
    score = float(correct_bond_score(
        final_state, n, static['correct_triu_idx']
    ))
    print(f'    Correct-bond fraction: {score:.4f}')

    # Final concentration plot
    fig = plot_final_concentrations(
        final_state, n, acid_base_np, correct_mask_np,
        title=f'pH schedule {pH_sched}  |  correct-bond fraction = {score:.3f}',
        save_path=os.path.join(outdir, f'final_conc_{label}.png'),
    )
    plt.close(fig)

    # Animation
    gif_path = os.path.join(outdir, f'animation_{label}.gif')
    try:
        animate_crn(
            traj_list, n, acid_base_np, correct_mask_np,
            pH_sched, duration,
            pKa_visual=p['pKa'],
            output_path=gif_path,
            fps=fps,
        )
    except Exception as exc:
        print(f'    Warning: animation failed ({exc}). Skipping GIF.')

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = build_parser().parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    params_path = os.path.join(outdir, args.params_file)

    # =======================================================================
    # TRAIN
    # =======================================================================
    if args.mode in ('train', 'both'):
        config = dict(
            n_species         = args.n_species,
            target_pH_schedule= args.target_pH,
            duration_per_seg  = args.duration,
            n_epochs          = args.n_epochs,
            learning_rate     = args.lr,
            beta              = args.beta,
            k0                = args.k0,
            n_points_sim      = args.n_points_sim,
            tau               = args.tau,
            seed              = args.seed,
        )

        print('=' * 60)
        print('CRN_AD  —  Training')
        print('=' * 60)

        result = train(config)
        (raw_params, loss_history, score_history,
         static, all_schedules, target_idx, denatured_state) = result

        # Save trained params
        p = constrain_params(raw_params)
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
        print(f'\nSaved trained params → {params_path}')

        # Training curve
        fig = plot_loss_curve(
            loss_history, score_history, target_idx, all_schedules,
            save_path=os.path.join(outdir, 'training_curve.png'),
        )
        plt.close(fig)

        # Charge curves
        fig = plot_charges_vs_pH(
            np.array(p['pKa']),
            np.array(static['acid_base']),
            save_path=os.path.join(outdir, 'charge_curves.png'),
        )
        plt.close(fig)

        # Print summary
        print('\n— Trained parameters —')
        for i in range(args.n_species):
            ab = 'base' if int(static['acid_base'][i]) == 1 else 'acid'
            print(f'  {SPECIES_NAMES[i]} ({ab:<4s}): pKa = {float(p["pKa"][i]):.3f}')
        print(f'  φ   = {float(p["phi"]):.4f}   (steric mismatch factor)')
        print(f'  J   = {float(p["J"]):.4f}   (coupling, kT units)')

    # =======================================================================
    # ANIMATE
    # =======================================================================
    if args.mode in ('animate', 'both') and not args.no_animation:

        # Load params if not coming from training
        if args.mode == 'animate':
            if not os.path.exists(params_path):
                print(f'ERROR: params file not found: {params_path}')
                sys.exit(1)
            with open(params_path) as f:
                pdata = json.load(f)
            static = static_from_params(pdata, pdata['beta'], pdata['k0'],
                                        args.n_points_sim, args.tau)
            p = {k: pdata[k] for k in ('pKa', 'phi', 'J')}
            target_sched  = [float(x) for x in pdata['target_pH_schedule']]
            all_schedules = all_unique_permutations(target_sched)
            target_idx    = all_schedules.index(target_sched)
            duration      = args.duration

            print('Equilibrating denatured state ...')
            denatured_state = equilibrate_denatured(
                static['n'], static['acid_base'], static['correct_mask'],
                static['i_idx'], static['j_idx'],
                J=float(p['J']), k0=static['k0'],
                ref_pH=7.0, duration=300.0, n_points=150,
            )
        else:
            p = {k: np.array(v) if k == 'pKa' else float(v)
                 for k, v in constrain_params(raw_params).items()}
            duration = args.duration

        print('\n— Animations —')

        # Target schedule
        target_sched = all_schedules[target_idx]
        score_target = run_animation(
            'target', target_sched, duration, p, static, denatured_state,
            outdir, args.anim_fps,
        )

        # Up to 3 non-target permutations for comparison
        others = [s for i, s in enumerate(all_schedules) if i != target_idx]
        for k, sched in enumerate(others[:3]):
            run_animation(
                f'perm{k+1}', sched, duration, p, static, denatured_state,
                outdir, args.anim_fps,
            )

        print(f'\nTarget schedule correct-bond fraction : {score_target:.4f}')

    print(f'\nAll outputs in: {outdir}/')


if __name__ == '__main__':
    main()
