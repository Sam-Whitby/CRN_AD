#!/usr/bin/env python3
"""
CRN_AD — main entry point.

Default behaviour
-----------------
  python main.py

  Trains a 4-species CRN to fold correctly only under the target schedule
  [9, 5, 7], evaluates all 6 permutations, and saves a single summary PNG
  which is then opened automatically.  No animation by default (--animate).

Usage examples
--------------
  python main.py                              # default 4-species run
  python main.py --n_species 6 --n_epochs 400
  python main.py --target_pH 9 5 7
  python main.py --animate                    # also produce animated GIFs
  python main.py --mode animate               # load saved params + make plots
  python main.py --J_max 5.0 --smooth_width 2.0    # larger J, smooth pH ramps
  python main.py --S_max 2.0                 # monomer entropy (shared value)
  python main.py --S_max 2.0 --per_monomer_entropy  # per-monomer entropy
"""

import argparse
import json
import os
import platform
import subprocess
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # must be before any JAX computation
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')

from crn_ad.training  import (train, constrain_params, all_unique_permutations,
                               correct_bond_score, compute_scores_fast)
from crn_ad.dynamics  import (simulate_schedule, make_initial_state, make_triu_indices)
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
    p.add_argument('--mode', choices=['train', 'animate', 'both', 'eval'], default='train',
                   help='train: train and plot | animate: load saved params and plot | '
                        'eval: plot for fully-specified parameters given on the CLI | '
                        'both: train + animate')
    p.add_argument('--n_species', type=int, default=4,
                   help='Number of species types (even). E.g. 4 → A,B,C,D.')
    p.add_argument('--n_types', type=int, default=1,
                   help='Number of types per species (T). With T>1 species X '
                        'becomes X1…XT. Correct bonds are Ai-Bi for matching type. '
                        'All types of a species share the same pKa.')
    p.add_argument('--target_pH', nargs='+', type=float, default=[9.0, 5.0, 7.0],
                   help='Target pH schedule, one value per segment.')
    p.add_argument('--duration', type=float, default=30.0,
                   help='Duration of each pH segment (time units).')
    p.add_argument('--equil_duration', type=float, default=80.0,
                   help='Duration of pH-7 pre-equilibration (time units).')
    p.add_argument('--n_epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.02, help='Adam learning rate.')
    p.add_argument('--beta', type=float, default=1.0, help='Inverse temperature β.')
    p.add_argument('--k0', type=float, default=1.0, help='Base rate constant k₀.')
    p.add_argument('--tau', type=float, default=6.0, help='Softmax loss temperature.')
    p.add_argument('--n_points_sim', type=int, default=40,
                   help='ODE time points per schedule segment.')
    p.add_argument('--n_points_equil', type=int, default=60,
                   help='ODE time points for pH-7 equilibration.')
    p.add_argument('--outdir', type=str, default='outputs')
    p.add_argument('--params_file', type=str, default='trained_params.json')
    p.add_argument('--animate', action='store_true',
                   help='Also generate animated GIFs (requires Pillow).')
    p.add_argument('--seed', type=int, default=42)
    # Stability / physics flags
    p.add_argument('--J_max', type=float, default=3.5,
                   help='Hard cap on the coupling constant J (kT).  '
                        'Larger values allow stronger electrostatic binding '
                        'but increase ODE stiffness.  Use --smooth_width '
                        'together with J_max > 3.5 for stability.')
    p.add_argument('--smooth_width', type=float, default=0.0,
                   help='If > 0, pH transitions between segments are smoothed '
                        'with a logistic sigmoid over this many time units. '
                        'Eliminates RHS discontinuities that cause adjoint NaN. '
                        'Recommended: 1–3 for J_max > 3.5 or lr > 0.05.')
    # Monomer entropy flags
    p.add_argument('--S_max', type=float, default=0.0,
                   help='If > 0, enable per-monomer conformational-entropy '
                        'parameters s_i ∈ [0, S_max] kT.  Each monomer '
                        'contributes s_i to the dimerisation free energy: '
                        'ΔG_ij += s_i + s_j.  Optimised by autodiff.')
    p.add_argument('--per_monomer_entropy', action='store_true',
                   help='If set, train a separate entropy value per monomer '
                        '(n values).  Default: one shared value for all.')
    p.add_argument('--specific_bonds', action='store_true',
                   help='If set, only species of the correct pairing can interact '
                        '(e.g. any A with any B, but A cannot bind C, D, or another A). '
                        'Equivalent to phi=0 between wrong-species pairs. '
                        'Within the correct species pair, phi still controls the '
                        'binding strength of type mismatches (A1-B2 etc.).')
    p.add_argument('--no_self_bonds', action='store_true',
                   help='If set, identical particles have zero interaction energy '
                        '(ΔG=0 for A-A, B-B, etc.; or A1-A1, B2-B2 with n_types>1). '
                        'Cross-type interactions (A1-A2, B1-B2, A1-B2 …) are unaffected.')
    p.add_argument('--fixed_phi', type=float, default=None,
                   help='If set, fix phi at this value in [0, 1] for the entire run '
                        'and do not train it.  If omitted, phi is a free parameter.')
    p.add_argument('--n_restarts', type=int, default=1,
                   help='Run training N times from different random starting points '
                        'and report the best result (lowest final loss).  Runs are '
                        'parallelised via ProcessPoolExecutor when possible, '
                        'otherwise sequential.')
    p.add_argument('--wide_init', action='store_true',
                   help='If set, all restarts (including the first) use wide uniform '
                        'initialisation: pKa ~ U[3.1, 9.9], φ ~ U[0.05, 0.95], '
                        'J ~ U[0.55, J_max].  Without this flag, all restarts use '
                        'the standard pH-guided initialisation (pKa centred near '
                        'target pH, φ ~ 0.2, J ~ 1.5).')
    # ---- Gradient clipping (JAX custom_vjp approach) ----
    p.add_argument('--grad_clip', type=float, default=None,
                   help='If set, clip the L2 norm of gradients flowing back through each '
                        'ODE call to this value, using a JAX custom_vjp wrapper.  '
                        'Helps prevent NaN gradients from the adjoint ODE.  '
                        'Try 1.0–10.0; smaller = more aggressive clipping.')
    # ---- Eval mode: specify all parameters explicitly ----
    p.add_argument('--eval_pKa', nargs='+', type=float, default=None,
                   help='(--mode eval) pKa values, one per species.')
    p.add_argument('--eval_phi', type=float, default=None,
                   help='(--mode eval) Steric mismatch factor φ ∈ [0, 1].')
    p.add_argument('--eval_J', nargs='+', type=float, default=None,
                   help='(--mode eval) Electrostatic coupling J (kT). '
                        'One value → same J for all correct pairs. '
                        'n_species/2 values → one J per correct species pair '
                        '(e.g. --eval_J 3.0 1.5 for two pairs A-B and C-D).')
    p.add_argument('--eval_monomer_entropy', nargs='+', type=float, default=None,
                   help='(--mode eval) Monomer conformational entropy s (kT). '
                        'One value → shared across all species. '
                        'n_species values (with --per_monomer_entropy) → one per species. '
                        'If omitted and --S_max > 0, defaults to S_max for all species.')
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_file(path):
    try:
        if platform.system() == 'Darwin':
            subprocess.Popen(['open', path])
        elif platform.system() == 'Linux':
            subprocess.Popen(['xdg-open', path])
        elif platform.system() == 'Windows':
            os.startfile(path)
    except Exception:
        pass


def _run_one_restart(config_seed):
    """Top-level worker for ProcessPoolExecutor — one training restart.

    Defined at module level so it can be pickled by multiprocessing.
    config_seed is a (config, seed, wide_init) tuple.
    Returns a dict of numpy-serialisable results.
    """
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    config, seed, wide_init = config_seed
    config = {**config, 'seed': seed, 'wide_init': wide_init, 'verbose': False}
    from crn_ad.training import train as _train, constrain_params as _cp
    import numpy as _np

    result = _train(config)
    (raw_params, loss_history, score_history, param_history,
     *_, init_phys_np, nan_stopped) = result
    p = _cp(raw_params,
            J_max=config.get('J_max', 3.5),
            S_max=config.get('S_max', 0.0),
            fixed_phi=config.get('fixed_phi'))
    return {
        'seed'              : seed,
        'wide_init'         : wide_init,
        'final_loss'        : float(loss_history[-1]),
        'epochs_completed'  : len(loss_history),
        'nan_stopped'       : bool(nan_stopped),
        'init_params'       : init_phys_np,
        'raw_params'        : {k: _np.array(v) for k, v in raw_params.items()},
        'p_eval'            : {k: (_np.array(v) if hasattr(v, 'shape') else float(v))
                               for k, v in p.items()},
        'loss_history'      : loss_history,
        'score_history'     : [_np.array(s) for s in score_history],
        'param_history'     : param_history,
    }


def _static_dict(n_species, T, beta, k0, n_points_sim, n_points_equil,
                 equil_duration, tau, J_max, S_max, smooth_width,
                 per_monomer_entropy=False, specific_bonds=False,
                 no_self_bonds=False):
    N = n_species * T
    acid_base_np    = np.array([(k // T) % 2 for k in range(N)], dtype=int)
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
    allowed_mask_jax = jnp.array(species_pair_mask_np) if specific_bonds else None
    return {
        'n'                   : N,
        'n_species'           : n_species,
        'T'                   : T,
        'acid_base'           : jnp.array(acid_base_np),
        'acid_base_np'        : acid_base_np,
        'correct_mask'        : jnp.array(correct_mask_np),
        'correct_mask_np'     : correct_mask_np,
        'species_pair_mask_np': species_pair_mask_np,
        'i_idx'               : i_idx,
        'j_idx'               : j_idx,
        'correct_triu_idx'    : jnp.array(correct_triu_idx),
        'beta'                : float(beta),
        'k0'                  : float(k0),
        'n_points_sim'        : int(n_points_sim),
        'n_points_equil'      : int(n_points_equil),
        'equil_duration'      : float(equil_duration),
        'equil_ramp_duration' : float(equil_duration) / 2.0,
        'tau'                 : float(tau),
        'J_max'               : float(J_max),
        'S_max'               : float(S_max),
        'smooth_width'        : float(smooth_width),
        'per_monomer_entropy' : bool(per_monomer_entropy),
        'specific_bonds'      : bool(specific_bonds),
        'no_self_bonds'       : bool(no_self_bonds),
        'allowed_mask'        : allowed_mask_jax,
    }


def get_equil_and_schedule_traj(p, static, target_sched, duration):
    """Run pH-7 equilibration then target schedule; return trajectories."""
    n            = static['n']
    T            = static.get('T', 1)
    allowed_mask  = static.get('allowed_mask', None)
    no_self_bonds = bool(static.get('no_self_bonds', False))
    mono_s        = _get_mono(p, static)

    # Expand species-level pKa (n_species,) → particle-level (N,)
    pKa_arr  = jnp.array(p['pKa'])
    pKa_full = jnp.repeat(pKa_arr, T) if T > 1 else pKa_arr
    if mono_s is not None and static.get('per_monomer_entropy', False) and T > 1:
        mono_s = jnp.repeat(mono_s, T)

    equil_ramp = float(static.get('equil_ramp_duration', 0.0))
    equil_final, equil_traj = simulate_schedule(
        make_initial_state(n), [7.0], static['equil_duration'],
        pKa_full, static['acid_base'],
        jnp.array(p['phi']), jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n, static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        monomer_entropy=mono_s,
        allowed_mask=allowed_mask,
        beta_ramp_duration=equil_ramp,
        no_self_bonds=no_self_bonds,
    )

    final_state, schedule_trajs = simulate_schedule(
        equil_final, target_sched, duration,
        pKa_full, static['acid_base'],
        jnp.array(p['phi']), jnp.array(p['J']),
        static['beta'], static['k0'],
        static['correct_mask'], n, static['i_idx'], static['j_idx'],
        n_points=static['n_points_equil'],
        monomer_entropy=mono_s,
        allowed_mask=allowed_mask,
        no_self_bonds=no_self_bonds,
    )
    return equil_traj[0], schedule_trajs, final_state


def _get_mono(p, static):
    """Return monomer_entropy as JAX array or None."""
    s = p.get('monomer_entropy', None)
    if s is None or static.get('S_max', 0.0) == 0.0:
        return None
    return jnp.atleast_1d(jnp.array(s))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_J_matrix(J_pairs, n_species, T):
    """Build an N×N J matrix for eval mode with per-correct-pair coupling.

    Correct pair p uses J_pairs[p].  All other interactions (wrong species,
    homodimers) use the mean of J_pairs as a baseline before phi is applied.
    """
    N = n_species * T
    n_pairs = n_species // 2
    J_mean = float(np.mean(J_pairs))
    J_mat = np.full((N, N), J_mean, dtype=float)
    for pair_idx in range(n_pairs):
        J_p = float(J_pairs[pair_idx])
        for t1 in range(T):
            for t2 in range(T):
                i = 2 * pair_idx * T + t1
                j = (2 * pair_idx + 1) * T + t2
                J_mat[i, j] = J_p
                J_mat[j, i] = J_p
    return jnp.array(J_mat)


def _j_scalar_for_history(J):
    """Return a scalar J suitable for param_history (mean if matrix)."""
    J_arr = np.array(J)
    return float(J_arr.mean()) if J_arr.ndim > 0 else float(J_arr)


def _print_param_table(p_eval, static, fixed_phi=None, title='Parameters'):
    """Print a formatted parameter table to stdout."""
    n_species    = static.get('n_species', static['n'])
    T            = static.get('T', 1)
    acid_base_np = np.array(static['acid_base_np'])
    pKa          = np.array(p_eval['pKa'])
    phi          = float(p_eval['phi'])
    J            = p_eval['J']
    S_max        = float(static.get('S_max', 0.0))
    W = 60

    print('─' * W)
    print(f'  {title}')
    print('─' * W)
    print(f'  {"Species":<12}  {"Role":<6}  {"pKa":>7}')
    print(f'  {"─"*12}  {"─"*6}  {"─"*7}')
    for i in range(n_species):
        ab = 'base' if int(acid_base_np[i * T]) == 1 else 'acid'
        print(f'  {SPECIES_NAMES[i]:<12}  {ab:<6}  {float(pKa[i]):>7.4f}')

    phi_tag = '  (fixed)' if fixed_phi is not None else ''
    print()
    print(f'  {"φ (steric factor)":<28} {phi:.4f}{phi_tag}')

    J_arr = np.array(J)
    if J_arr.ndim == 0:
        print(f'  {"J (coupling)":<28} {float(J_arr):.4f}  kT')
    else:
        print(f'  J (coupling, per pair):')
        for pair_idx in range(n_species // 2):
            A   = SPECIES_NAMES[2 * pair_idx]
            B   = SPECIES_NAMES[2 * pair_idx + 1]
            i0  = 2 * pair_idx * T
            j0  = (2 * pair_idx + 1) * T
            j_v = float(J_arr[i0, j0])
            print(f'    {A}–{B}: {j_v:.4f}  kT')

    if S_max > 0.0 and p_eval.get('monomer_entropy') is not None:
        s = np.atleast_1d(np.array(p_eval['monomer_entropy']))
        if len(s) == 1:
            print(f'  {"s (entropy, shared)":<28} {float(s[0]):.4f}  kT')
        else:
            for i, sv in enumerate(s):
                print(f'  {"s(" + SPECIES_NAMES[i] + ")":<28} {float(sv):.4f}  kT')

    print()
    print(f'  {"Equilibration":<28} {static["equil_duration"]:.0f}  time units  (pH 7)')
    print(f'  {"β (inv. temperature)":<28} {static["beta"]:.2f}')
    print(f'  {"k₀ (base rate)":<28} {static["k0"]:.2f}')
    print('─' * W)


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
            n_species            = args.n_species,
            n_types              = args.n_types,
            target_pH_schedule   = args.target_pH,
            duration_per_seg     = args.duration,
            equil_duration       = args.equil_duration,
            n_epochs             = args.n_epochs,
            learning_rate        = args.lr,
            beta                 = args.beta,
            k0                   = args.k0,
            n_points_sim         = args.n_points_sim,
            n_points_equil       = args.n_points_equil,
            tau                  = args.tau,
            seed                 = args.seed,
            J_max                = args.J_max,
            smooth_width         = args.smooth_width,
            S_max                = args.S_max,
            per_monomer_entropy  = args.per_monomer_entropy,
            specific_bonds       = args.specific_bonds,
            no_self_bonds        = args.no_self_bonds,
            wide_init            = args.wide_init,
            fixed_phi            = args.fixed_phi,
            grad_clip            = args.grad_clip,
        )

        print('=' * 60)
        print('CRN_AD  —  Training')
        print('=' * 60)

        n_restarts = int(args.n_restarts)

        if n_restarts > 1:
            init_tag = 'wide' if args.wide_init else 'standard'
            print(f'Running {n_restarts} restarts ({init_tag} init) ...')
            seeds = [args.seed + i for i in range(n_restarts)]
            pairs = [(config, s, args.wide_init) for s in seeds]

            try:
                from concurrent.futures import ProcessPoolExecutor
                print('  Using parallel execution (ProcessPoolExecutor) ...', flush=True)
                with ProcessPoolExecutor() as executor:
                    restart_results = list(executor.map(_run_one_restart, pairs))
                print('  Parallel restarts complete.\n')
            except Exception as exc:
                print(f'  Parallel execution unavailable ({exc}); running sequentially.\n')
                restart_results = [_run_one_restart(p) for p in pairs]

            # Print summary table
            n_epochs_req = int(config['n_epochs'])
            S_max_cfg    = float(config.get('S_max', 0.0))
            print(f'  {"seed":>5}  {"init":>8}  {"start pKa":<{12 * config["n_species"] // 2}}  '
                  f'{"φ":>5}  {"J":>5}'
                  + (f'  {"s̄":>5}' if S_max_cfg > 0 else '')
                  + f'  {"epochs":>11}  {"loss":>8}')
            print('  ' + '─' * (70 + (8 if S_max_cfg > 0 else 0)))
            for r in restart_results:
                ip       = r['init_params']
                pKa_str  = ' '.join(f'{float(v):5.2f}' for v in ip['pKa'])
                phi_str  = f'{float(ip["phi"]):5.3f}'
                J_str    = f'{float(ip["J"]):5.3f}'
                s_str    = (f'  {float(np.mean(ip["monomer_entropy"])):5.3f}'
                            if S_max_cfg > 0 and 'monomer_entropy' in ip else
                            (f'  {"—":>5}' if S_max_cfg > 0 else ''))
                ep_str   = f'{r["epochs_completed"]:4d}/{n_epochs_req}'
                nan_flag = '  [NaN stop]' if r['nan_stopped'] else ''
                loss_str = f'{r["final_loss"]:8.4f}'
                print(f'  {r["seed"]:>5}  {("wide" if r["wide_init"] else "std"):>8}'
                      f'  pKa=[{pKa_str}]  φ={phi_str}  J={J_str}{s_str}'
                      f'  {ep_str} epochs  {loss_str}{nan_flag}')

            best = min(restart_results, key=lambda r: r['final_loss'])
            print(f'\nBest restart: seed={best["seed"]}  '
                  f'final_loss={best["final_loss"]:.4f}')

            # Re-run the best restart with verbose output.  Use the same wide_init
            # so the trajectory is reproduced exactly (same seed + same init strategy).
            print('\nRe-running best restart with full output ...')
            best_config = {**config, 'seed': best['seed'],
                           'wide_init': best['wide_init'], 'verbose': True}
            (raw_params, loss_history, score_history, param_history,
             static, all_schedules, target_idx, *_) = train(best_config)
        else:
            (raw_params, loss_history, score_history, param_history,
             static, all_schedules, target_idx, *_) = train(config)

        p_eval = constrain_params(raw_params, J_max=args.J_max, S_max=args.S_max,
                                  fixed_phi=args.fixed_phi)
        p_eval = {k: (np.array(v) if hasattr(v, '__len__') else float(v))
                  for k, v in p_eval.items()}

        # Save params
        params_out = {
            'n_species'         : args.n_species,
            'n_types'           : args.n_types,
            'target_pH_schedule': args.target_pH,
            'pKa'               : p_eval['pKa'].tolist(),
            'phi'               : float(p_eval['phi']),
            'J'                 : float(p_eval['J']),
            'beta'              : args.beta,
            'k0'                : args.k0,
            'J_max'             : args.J_max,
            'S_max'             : args.S_max,
            'per_monomer_entropy': args.per_monomer_entropy,
            'specific_bonds'     : args.specific_bonds,
            'no_self_bonds'      : args.no_self_bonds,
            'fixed_phi'          : args.fixed_phi,
        }
        if args.S_max > 0.0 and 'monomer_entropy' in p_eval:
            params_out['monomer_entropy'] = np.atleast_1d(
                p_eval['monomer_entropy']).tolist()

        with open(params_path, 'w') as f:
            json.dump(params_out, f, indent=2)
        print(f'Saved trained params → {params_path}')

        _print_param_table(p_eval, static,
                           fixed_phi=args.fixed_phi,
                           title='Trained Parameters')

        target_sched = [float(x) for x in args.target_pH]

    # =====================================================================
    # Load params for animate-only mode
    # =====================================================================
    if args.mode == 'animate':
        if not os.path.exists(params_path):
            print(f'ERROR: params file not found: {params_path}')
            sys.exit(1)
        with open(params_path) as f:
            pdata = json.load(f)
        _J_max  = float(pdata.get('J_max', args.J_max))
        _S_max  = float(pdata.get('S_max', args.S_max))
        _permon = bool(pdata.get('per_monomer_entropy', False))
        _T      = int(pdata.get('n_types', 1))
        _specb  = bool(pdata.get('specific_bonds', False))
        _nsb = bool(pdata.get('no_self_bonds', False))
        static = _static_dict(
            pdata['n_species'], _T,
            pdata['beta'], pdata['k0'],
            args.n_points_sim, args.n_points_equil,
            args.equil_duration, args.tau,
            _J_max, _S_max, args.smooth_width, _permon, _specb, _nsb,
        )
        p_eval = {
            'pKa': np.array(pdata['pKa']),
            'phi': float(pdata['phi']),
            'J'  : float(pdata['J']),
            'monomer_entropy': (np.array(pdata['monomer_entropy'])
                                if 'monomer_entropy' in pdata else None),
        }
        target_sched  = [float(x) for x in pdata['target_pH_schedule']]
        all_schedules = all_unique_permutations(target_sched)
        target_idx    = all_schedules.index(target_sched)
        loss_history  = [0.0]
        score_history = None
        param_history = [{'pKa': p_eval['pKa'], 'phi': p_eval['phi'], 'J': p_eval['J']}]

    # =====================================================================
    # EVAL — fully-specified parameters from CLI, no training
    # =====================================================================
    if args.mode == 'eval':
        missing = [n for n, v in [('--eval_pKa', args.eval_pKa),
                                   ('--eval_phi', args.eval_phi),
                                   ('--eval_J',   args.eval_J)] if v is None]
        if missing:
            print(f'ERROR: --mode eval requires {", ".join(missing)}')
            sys.exit(1)
        if len(args.eval_pKa) != args.n_species:
            print(f'ERROR: --eval_pKa must have exactly {args.n_species} values '
                  f'(got {len(args.eval_pKa)}) — set --n_species accordingly.')
            sys.exit(1)

        n_pairs = args.n_species // 2
        j_raw   = args.eval_J
        if len(j_raw) == 1:
            J_eval = float(j_raw[0])          # scalar: same J for all pairs
        elif len(j_raw) == n_pairs:
            J_eval = _build_J_matrix(j_raw, args.n_species, args.n_types)
        else:
            print(f'ERROR: --eval_J must have 1 value (same for all pairs) or '
                  f'{n_pairs} values (one per correct pair). Got {len(j_raw)}.')
            sys.exit(1)

        # --- Monomer entropy for eval mode ---
        # If --eval_monomer_entropy is given, use those values.
        # If omitted but --S_max > 0, default to S_max as the shared value.
        # Validate count: must be 1 (shared) or n_species (per-species).
        me_raw = args.eval_monomer_entropy
        if me_raw is not None:
            if len(me_raw) == 1:
                mono_eval = np.array(me_raw, dtype=float)        # shape (1,) → shared
            elif len(me_raw) == args.n_species:
                mono_eval = np.array(me_raw, dtype=float)        # shape (n_species,)
            else:
                print(f'ERROR: --eval_monomer_entropy must have 1 value (shared) or '
                      f'{args.n_species} values (one per species). Got {len(me_raw)}.')
                sys.exit(1)
            # S_max must cover the given entropy so _get_mono doesn't silently drop it.
            # If user forgot --S_max, infer it from the given values.
            S_max_eval = max(args.S_max, float(np.max(mono_eval)))
        elif args.S_max > 0.0:
            # --S_max given, no explicit entropy → use S_max as the shared entropy value
            mono_eval  = np.array([args.S_max], dtype=float)
            S_max_eval = args.S_max
        else:
            mono_eval  = None
            S_max_eval = 0.0

        static = _static_dict(
            args.n_species, args.n_types,
            args.beta, args.k0,
            args.n_points_sim, args.n_points_equil,
            args.equil_duration, args.tau,
            args.J_max, S_max_eval, args.smooth_width,
            args.per_monomer_entropy, args.specific_bonds, args.no_self_bonds,
        )
        p_eval = {
            'pKa'            : np.array(args.eval_pKa),
            'phi'            : float(args.eval_phi),
            'J'              : J_eval,
            'monomer_entropy': mono_eval,   # None or array
        }
        target_sched  = [float(x) for x in args.target_pH]
        all_schedules = all_unique_permutations(target_sched)
        target_idx    = all_schedules.index(target_sched)
        loss_history  = [0.0]
        score_history = None
        # param_history stores scalar J for the evolution plots (trivially flat)
        param_history = [{'pKa': np.array(args.eval_pKa),
                          'phi': float(args.eval_phi),
                          'J'  : _j_scalar_for_history(J_eval)}]
        print('=' * 60)
        print('CRN_AD  —  Eval (fixed parameters)')
        print('=' * 60)
        _print_param_table(p_eval, static, title='Eval Parameters')

    # =====================================================================
    # SUMMARY PLOT
    # =====================================================================
    if args.mode in ('train', 'both', 'animate', 'eval'):
        print('\nGenerating summary plot ...')

        equil_traj, schedule_trajs, _ = get_equil_and_schedule_traj(
            p_eval, static, target_sched, args.duration)

        # Fast vmap-based scoring — compiles once, runs in parallel
        print('  Scoring all schedule permutations ...')
        all_schedules_local = all_schedules if args.mode == 'animate' else all_schedules
        final_scores = compute_scores_fast(
            p_eval, all_schedules_local, args.duration, static)

        if args.mode == 'animate' or score_history is None:
            score_history = [final_scores]
            param_history = param_history  # already set

        trained_params = {
            'pKa'           : np.array(p_eval['pKa']),
            'phi'           : float(p_eval['phi']),
            'J'             : p_eval['J'],          # scalar or N×N matrix
            'monomer_entropy': (np.atleast_1d(np.array(p_eval['monomer_entropy']))
                                if p_eval.get('monomer_entropy') is not None else None),
        }

        plot_summary(
            loss_history, score_history, param_history,
            all_schedules_local, target_idx,
            equil_traj, schedule_trajs, target_sched,
            args.equil_duration, args.duration,
            static, trained_params, final_scores,
            save_path=summary_path,
        )

        print(f'\nSummary plot → {summary_path}')
        open_file(summary_path)

        print('\n— Final scores (all schedules) —')
        for i, (sched, sc) in enumerate(zip(all_schedules_local, final_scores)):
            marker = ' ← TARGET' if i == target_idx else ''
            print(f'  {sched}  →  {sc:.4f}{marker}')

    # =====================================================================
    # OPTIONAL ANIMATIONS
    # =====================================================================
    if args.animate or args.mode == 'both':
        print('\nGenerating animations ...')
        n               = static['n']
        T_val           = static.get('T', 1)
        acid_base_np    = static['acid_base_np']
        correct_mask_np = static['correct_mask_np']
        # pKa for animation: expand from n_species to N particles
        pKa_vis = np.array(p_eval['pKa'])
        pKa_vis = np.repeat(pKa_vis, T_val) if T_val > 1 else pKa_vis

        for s_idx, sched in enumerate(
                [all_schedules[target_idx]] +
                [s for i, s in enumerate(all_schedules) if i != target_idx][:2]):
            label = 'target' if s_idx == 0 else f'perm{s_idx}'
            equil_t, sched_trajs, _ = get_equil_and_schedule_traj(
                p_eval, static, sched, args.duration)
            gif = os.path.join(outdir, f'animation_{label}.gif')
            try:
                animate_crn(
                    [equil_t] + sched_trajs, n, acid_base_np, correct_mask_np,
                    [7.0] + list(sched), args.duration,
                    pKa_visual=pKa_vis,
                    output_path=gif, fps=12,
                )
            except Exception as exc:
                print(f'  Warning: animation failed ({exc})')

    print('\nDone.')


if __name__ == '__main__':
    main()
