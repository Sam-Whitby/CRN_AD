#!/usr/bin/env python3
"""
CRN_AD — main entry point.

Default behaviour
-----------------
  python main.py

  Trains a CRN with N=4 acid + 4 base species (M=2 classifier pairs,
  2 roughness pairs) to fold correctly only under the target schedule
  [9, 5, 7], evaluates all permutations, and saves a summary PNG.

Usage examples
--------------
  python main.py                              # default N=4, M=2
  python main.py --N 6 --M 3 --n_epochs 400
  python main.py --N 8 --M 2                 # 2 classifier + 6 roughness pairs
  python main.py --target_pH 9 5 7
  python main.py --animate                    # also produce animated GIFs
  python main.py --mode animate               # load saved params + make plots
  python main.py --J_max 5.0 --smooth_width 2.0
  python main.py --S_max 2.0                 # monomer entropy (shared value)
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
    p.add_argument('--N', type=int, default=4,
                   help='Total number of acid species (and base species). '
                        '2N monomers in total.')
    p.add_argument('--M', type=int, default=2,
                   help='Number of classifier pairs (M ≤ N). '
                        'Acids 0..M-1 form correct bonds with bases N..N+M-1. '
                        'Acids M..N-1 and bases N+M..2N-1 are roughness-only species.')
    p.add_argument('--target_pH', nargs='+', type=float, default=[9.0, 5.0, 7.0],
                   help='Target pH schedule, one value per segment.')
    p.add_argument('--duration', type=float, default=30.0,
                   help='Duration of each pH segment in units of 1/k0.  '
                        'Together with --k0 only the product k0×duration matters '
                        '(rescaling k0 and duration by the same factor is exactly equivalent).')
    p.add_argument('--equil_duration', type=float, default=80.0,
                   help='Duration of pH-7 pre-equilibration in units of 1/k0.  '
                        'Should satisfy k0×equil_duration ≫ exp(J) to reach '
                        'thermodynamic equilibrium (e.g. J=3.5→≫33, J=10→≫22000).')
    p.add_argument('--n_epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.02, help='Adam learning rate.')
    p.add_argument('--beta', type=float, default=1.0,
                   help='Inverse temperature β.  When J is trained, β is degenerate '
                        'with J (only β·J matters); changing β is equivalent to '
                        'rescaling J. Meaningful as a true temperature knob only when '
                        'J is fixed via --fixed_J.  Default 1.0 (energies in kT).')
    p.add_argument('--k0', type=float, default=1.0,
                   help='Base rate constant k₀.  Only the products k0×duration and '
                        'k0×equil_duration matter physically; --k0 is a convenience '
                        'multiplier that is absorbed into the durations internally.  '
                        'Small k0 → kinetic trapping; large k0 → thermodynamic limit.')
    p.add_argument('--tau', type=float, default=6.0,
                   help='Softmax temperature τ in the training loss.  '
                        'Higher τ sharpens discrimination between schedules.  '
                        'Does not affect ODE physics.')
    p.add_argument('--n_points_sim', type=int, default=None,
                   help='ODE output points per schedule segment (visualisation only, '
                        'does not affect ODE accuracy).  Default: auto = max(20, 2×duration).')
    p.add_argument('--n_points_equil', type=int, default=None,
                   help='ODE output points for equilibration (visualisation only).  '
                        'Default: auto = max(30, 2×equil_duration).')
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
    p.add_argument('--fixed_J', type=float, default=None,
                   help='If set, fix J (kT) at this value for the entire run '
                        'and do not train it.  If omitted, J is a free parameter.')
    p.add_argument('--pka_default', action='store_true',
                   help='Fix pKa values and do not train them.  Acid-like species '
                        '(even index) get pKa=--pKa_acid; base-like (odd) get pKa=--pKa_base.')
    p.add_argument('--pKa_acid', type=float, default=6.0,
                   help='pKa for acid-like species when --pka_default is set  [default: 6.0]')
    p.add_argument('--pKa_base', type=float, default=8.0,
                   help='pKa for base-like species when --pka_default is set  [default: 8.0]')
    p.add_argument('--no_baseline', action='store_true',
                   help='Exclude the pH-7 equilibrium baseline from the loss function. '
                        'By default the baseline score is an additional negative class '
                        'in the softmax loss.  With this flag only schedule permutations '
                        'are compared.  For a single-pH target this makes the loss a '
                        'direct maximisation of the correct-dimer fraction.')
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
    p.add_argument('--J_init_max', action='store_true',
                   help='If set, initialise J at J_max rather than the default ~1.5 kT. '
                        'Overrides both standard and --wide_init J sampling. '
                        'Intended to explore solutions with high kinetic frustration.')
    p.add_argument('--phi_init_max', action='store_true',
                   help='If set, initialise φ at 1.0 (maximum steric mismatch penalty) '
                        'rather than the default ~0.2. Overrides both standard and '
                        '--wide_init φ sampling.')
    # ---- Optimiser options ----
    p.add_argument('--weight_decay', type=float, default=0.0,
                   help='AdamW L2 weight decay on raw (unconstrained) parameters.  '
                        'Provides a restoring force pulling raw params toward 0 '
                        '(the sigmoid midpoint), preventing boundary sticking.  '
                        'Try 1e-4 to 1e-2.  Default 0 (plain Adam).')
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
    _N_acids = int(config.get('N_total', 2))
    _pka_default = config.get('pka_default', False)
    _pKa_acid = config.get('pKa_acid', 6.0)
    _pKa_base = config.get('pKa_base', 8.0)
    _fixed_pKa = ([_pKa_acid]*_N_acids + [_pKa_base]*_N_acids
                  if _pka_default else None)
    p = _cp(raw_params,
            J_max=config.get('J_max', 3.5),
            S_max=config.get('S_max', 0.0),
            fixed_phi=config.get('fixed_phi'),
            fixed_J=config.get('fixed_J'),
            fixed_pKa=_fixed_pKa)
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


def _static_dict(N_acids, M, beta, k0, n_points_sim, n_points_equil,
                 equil_duration, tau, J_max, S_max, smooth_width,
                 no_self_bonds=False):
    N = 2 * N_acids  # total particle count
    acid_base_np    = np.array([0]*N_acids + [1]*N_acids, dtype=int)
    correct_mask_np = np.zeros((N, N), dtype=bool)
    for i in range(M):
        correct_mask_np[i, N_acids + i] = True
        correct_mask_np[N_acids + i, i] = True
    i_idx, j_idx = make_triu_indices(N)
    correct_triu_idx = np.array([
        pos for pos, (ii, jj) in enumerate(zip(i_idx, j_idx))
        if correct_mask_np[ii, jj]
    ])
    return {
        'n'                   : N,
        'n_species'           : N,  # for visualize.py compat: 2*N_acids pKa values
        'N_total'             : N_acids,
        'M_classifier'        : M,
        'acid_base'           : jnp.array(acid_base_np),
        'acid_base_np'        : acid_base_np,
        'correct_mask'        : jnp.array(correct_mask_np),
        'correct_mask_np'     : correct_mask_np,
        'i_idx'               : i_idx,
        'j_idx'               : j_idx,
        'correct_triu_idx'    : jnp.array(correct_triu_idx),
        'beta'                : float(beta),
        'k0'                  : float(k0),
        'n_points_sim'        : (int(n_points_sim) if n_points_sim is not None
                                  else 40),
        'n_points_equil'      : (int(n_points_equil) if n_points_equil is not None
                                  else max(30, int(2 * float(equil_duration)))),
        'equil_duration'      : float(equil_duration),
        'equil_ramp_duration' : float(equil_duration) / 2.0,
        'tau'                 : float(tau),
        'J_max'               : float(J_max),
        'S_max'               : float(S_max),
        'smooth_width'        : float(smooth_width),
        'no_self_bonds'       : bool(no_self_bonds),
        'allowed_mask'        : None,
    }


def get_equil_and_schedule_traj(p, static, target_sched, duration):
    """Run pH-7 equilibration then target schedule; return trajectories."""
    n            = static['n']
    allowed_mask  = static.get('allowed_mask', None)
    no_self_bonds = bool(static.get('no_self_bonds', False))
    mono_s        = _get_mono(p, static)

    # pKa already has one value per particle (2*N_acids elements)
    pKa_full = jnp.array(p['pKa'])

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

def _build_J_matrix(J_pairs, N_acids, M):
    """Build a 2N×2N J matrix for eval mode with per-classifier-pair coupling.

    Classifier pair i uses J_pairs[i].  All other interactions use the mean
    of J_pairs as a baseline (before phi is applied to wrong bonds).
    """
    N = 2 * N_acids
    J_mean = float(np.mean(J_pairs))
    J_mat = np.full((N, N), J_mean, dtype=float)
    for i in range(M):
        J_p = float(J_pairs[i])
        J_mat[i, N_acids + i] = J_p
        J_mat[N_acids + i, i] = J_p
    return jnp.array(J_mat)


def _j_scalar_for_history(J):
    """Return a scalar J suitable for param_history (mean if matrix)."""
    J_arr = np.array(J)
    return float(J_arr.mean()) if J_arr.ndim > 0 else float(J_arr)


def _print_param_table(p_eval, static, fixed_phi=None, fixed_J=None,
                       fixed_pKa=False, title='Parameters'):
    """Print a formatted parameter table to stdout."""
    N_acids      = static.get('N_total', static['n'] // 2)
    M_clf        = static.get('M_classifier', N_acids)
    acid_base_np = np.array(static['acid_base_np'])
    pKa          = np.array(p_eval['pKa'])
    phi          = float(p_eval['phi'])
    J            = p_eval['J']
    S_max        = float(static.get('S_max', 0.0))
    W = 60

    print('─' * W)
    print(f'  {title}')
    print(f'  N={N_acids} acids + {N_acids} bases  |  M={M_clf} classifier pairs  |  '
          f'{N_acids-M_clf} roughness acid/base pairs')
    print('─' * W)
    pKa_tag = '  (fixed)' if fixed_pKa else ''
    print(f'  {"Particle":<14}  {"Role":<10}  {"Type":<12}  {"pKa":>7}{pKa_tag}')
    print(f'  {"─"*14}  {"─"*10}  {"─"*12}  {"─"*7}')
    for i in range(N_acids):
        role = 'classifier' if i < M_clf else 'roughness'
        print(f'  {SPECIES_NAMES[i]:<14}  {"acid":<10}  {role:<12}  {float(pKa[i]):>7.4f}')
    for i in range(N_acids):
        j = N_acids + i
        role = 'classifier' if i < M_clf else 'roughness'
        label = SPECIES_NAMES[i].lower() if i < 26 else f'b{i}'
        print(f'  {label:<14}  {"base":<10}  {role:<12}  {float(pKa[j]):>7.4f}')

    phi_tag = '  (fixed)' if fixed_phi is not None else ''
    print()
    print(f'  {"φ (steric factor)":<28} {phi:.4f}{phi_tag}')

    J_arr = np.array(J)
    J_fixed_tag = '  kT  (fixed)' if fixed_J is not None else '  kT'
    if J_arr.ndim == 0:
        print(f'  {"J (coupling)":<28} {float(J_arr):.4f}{J_fixed_tag}')
    else:
        print(f'  J (coupling, per classifier pair):')
        for i in range(M_clf):
            j_v = float(J_arr[i, N_acids + i])
            print(f'    {SPECIES_NAMES[i]}–{SPECIES_NAMES[i].lower()}: {j_v:.4f}  kT')

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
        # k0 is absorbed into the durations: only k0×duration matters physically.
        # Internally we always use k0=1 so that times are in natural units (1/k0).
        _k0 = float(args.k0)
        config = dict(
            N_total              = args.N,
            M_classifier         = args.M,
            target_pH_schedule   = args.target_pH,
            duration_per_seg     = args.duration * _k0,
            equil_duration       = args.equil_duration * _k0,
            n_epochs             = args.n_epochs,
            learning_rate        = args.lr,
            beta                 = args.beta,
            k0                   = 1.0,
            n_points_sim         = args.n_points_sim,    # None → auto in training.py
            n_points_equil       = args.n_points_equil,  # None → auto in training.py
            tau                  = args.tau,
            seed                 = args.seed,
            J_max                = args.J_max,
            smooth_width         = args.smooth_width,
            S_max                = args.S_max,
            per_monomer_entropy  = args.per_monomer_entropy,
            no_self_bonds        = args.no_self_bonds,
            wide_init            = args.wide_init,
            j_init_max           = args.J_init_max,
            phi_init_max         = args.phi_init_max,
            fixed_phi            = args.fixed_phi,
            fixed_J              = args.fixed_J,
            pka_default          = args.pka_default,
            pKa_acid             = args.pKa_acid,
            pKa_base             = args.pKa_base,
            no_baseline          = args.no_baseline,
            weight_decay         = args.weight_decay,
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
            print(f'  {"seed":>5}  {"init":>8}  {"start pKa (2N values)":<30}  '
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

        _fixed_pKa_eval = ([args.pKa_acid]*args.N + [args.pKa_base]*args.N
                            if args.pka_default else None)
        p_eval = constrain_params(raw_params, J_max=args.J_max, S_max=args.S_max,
                                  fixed_phi=args.fixed_phi, fixed_J=args.fixed_J,
                                  fixed_pKa=_fixed_pKa_eval)
        p_eval = {k: (np.array(v) if hasattr(v, '__len__') else float(v))
                  for k, v in p_eval.items()}

        # Save params
        params_out = {
            'N_total'           : args.N,
            'M_classifier'      : args.M,
            'target_pH_schedule': args.target_pH,
            'pKa'               : p_eval['pKa'].tolist(),
            'phi'               : float(p_eval['phi']),
            'J'                 : float(p_eval['J']),
            'beta'              : args.beta,
            'k0'                : 1.0,
            'duration_per_seg'  : float(config['duration_per_seg']),
            'equil_duration'    : float(config['equil_duration']),
            'J_max'             : args.J_max,
            'S_max'             : args.S_max,
            'per_monomer_entropy': args.per_monomer_entropy,
            'no_self_bonds'      : args.no_self_bonds,
            'fixed_phi'          : args.fixed_phi,
            'fixed_J'            : args.fixed_J,
            'pka_default'        : args.pka_default,
        }
        if args.S_max > 0.0 and 'monomer_entropy' in p_eval:
            params_out['monomer_entropy'] = np.atleast_1d(
                p_eval['monomer_entropy']).tolist()

        with open(params_path, 'w') as f:
            json.dump(params_out, f, indent=2)
        print(f'Saved trained params → {params_path}')

        _print_param_table(p_eval, static,
                           fixed_phi=args.fixed_phi,
                           fixed_J=args.fixed_J,
                           fixed_pKa=args.pka_default,
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
        _nsb    = bool(pdata.get('no_self_bonds', False))
        _N_anim = int(pdata['N_total'])
        _M_anim = int(pdata['M_classifier'])
        # Load effective durations (k0-multiplied) from params file if present.
        _k0_anim = float(pdata.get('k0', 1.0))
        _anim_equil_dur = float(pdata.get('equil_duration',
                                          args.equil_duration * _k0_anim))
        _anim_duration  = float(pdata.get('duration_per_seg',
                                          args.duration * _k0_anim))
        static = _static_dict(
            _N_anim, _M_anim,
            pdata['beta'], 1.0,
            args.n_points_sim, args.n_points_equil,
            _anim_equil_dur, args.tau,
            _J_max, _S_max, args.smooth_width, _nsb,
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
        if len(args.eval_pKa) != 2 * args.N:
            print(f'ERROR: --eval_pKa must have exactly {2*args.N} values '
                  f'(got {len(args.eval_pKa)}) — set --N accordingly.')
            sys.exit(1)

        n_pairs = args.M
        j_raw   = args.eval_J
        if len(j_raw) == 1:
            J_eval = float(j_raw[0])          # scalar: same J for all pairs
        elif len(j_raw) == n_pairs:
            J_eval = _build_J_matrix(j_raw, args.N, args.M)
        else:
            print(f'ERROR: --eval_J must have 1 value (same for all pairs) or '
                  f'{n_pairs} values (one per classifier pair). Got {len(j_raw)}.')
            sys.exit(1)

        # --- Monomer entropy for eval mode ---
        me_raw = args.eval_monomer_entropy
        if me_raw is not None:
            if len(me_raw) == 1:
                mono_eval = np.array(me_raw, dtype=float)        # shared
            elif len(me_raw) == 2 * args.N:
                mono_eval = np.array(me_raw, dtype=float)        # per-particle
            else:
                print(f'ERROR: --eval_monomer_entropy must have 1 value (shared) or '
                      f'{2*args.N} values (one per particle). Got {len(me_raw)}.')
                sys.exit(1)
            S_max_eval = max(args.S_max, float(np.max(mono_eval)))
        elif args.S_max > 0.0:
            mono_eval  = np.array([args.S_max], dtype=float)
            S_max_eval = args.S_max
        else:
            mono_eval  = None
            S_max_eval = 0.0

        static = _static_dict(
            args.N, args.M,
            args.beta, args.k0,
            args.n_points_sim, args.n_points_equil,
            args.equil_duration, args.tau,
            args.J_max, S_max_eval, args.smooth_width,
            args.no_self_bonds,
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

        # Effective duration to use for visualization trajectories.
        # In train mode k0 was absorbed into the duration, so the static dict
        # uses k0=1 and the effective duration is config['duration_per_seg'].
        # In animate mode the effective duration was loaded from the params file.
        # In eval mode k0 in the static dict handles the scaling, so args.duration
        # is passed as-is (the ODE RHS multiplies by static['k0']).
        if args.mode in ('train', 'both'):
            _eff_duration = float(config['duration_per_seg'])
        elif args.mode == 'animate':
            _eff_duration = _anim_duration
        else:  # eval
            _eff_duration = args.duration

        equil_traj, schedule_trajs, _ = get_equil_and_schedule_traj(
            p_eval, static, target_sched, _eff_duration)

        # Fast vmap-based scoring — compiles once, runs in parallel
        print('  Scoring all schedule permutations ...')
        all_schedules_local = all_schedules if args.mode == 'animate' else all_schedules
        final_scores = compute_scores_fast(
            p_eval, all_schedules_local, _eff_duration, static)

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

        _plot_config = config if args.mode in ('train', 'both') else None
        plot_summary(
            loss_history, score_history, param_history,
            all_schedules_local, target_idx,
            equil_traj, schedule_trajs, target_sched,
            static['equil_duration'], _eff_duration,
            static, trained_params, final_scores,
            save_path=summary_path,
            config=_plot_config,
        )

        print(f'\nSummary plot → {summary_path}')
        open_file(summary_path)

        print('\n— Final scores —')
        n_scheds = len(all_schedules_local)
        for i, (sched, sc) in enumerate(zip(all_schedules_local, final_scores[:n_scheds])):
            marker = ' ← TARGET' if i == target_idx else ''
            print(f'  {sched}  →  {sc:.4f}{marker}')
        if len(final_scores) > n_scheds:
            print(f'  pH 7 (baseline)  →  {final_scores[n_scheds]:.4f}')

    # =====================================================================
    # OPTIONAL ANIMATIONS
    # =====================================================================
    if args.animate or args.mode == 'both':
        print('\nGenerating animations ...')
        n               = static['n']
        acid_base_np    = static['acid_base_np']
        correct_mask_np = static['correct_mask_np']
        pKa_vis = np.array(p_eval['pKa'])  # already one value per particle

        for s_idx, sched in enumerate(
                [all_schedules[target_idx]] +
                [s for i, s in enumerate(all_schedules) if i != target_idx][:2]):
            label = 'target' if s_idx == 0 else f'perm{s_idx}'
            equil_t, sched_trajs, _ = get_equil_and_schedule_traj(
                p_eval, static, sched, _eff_duration)
            gif = os.path.join(outdir, f'animation_{label}.gif')
            try:
                animate_crn(
                    [equil_t] + sched_trajs, n, acid_base_np, correct_mask_np,
                    [7.0] + list(sched), _eff_duration,
                    pKa_visual=pKa_vis,
                    output_path=gif, fps=12,
                )
            except Exception as exc:
                print(f'  Warning: animation failed ({exc})')

    print('\nDone.')


if __name__ == '__main__':
    main()
