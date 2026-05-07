"""
Visualisation: training curves, final concentration bar charts,
animated network diagrams, and the comprehensive summary PNG.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import string

from .physics import henderson_hasselbalch, interaction_energy_matrix
import jax.numpy as jnp

SPECIES_NAMES = list(string.ascii_uppercase)


def _particle_labels(n_species, T):
    """Return labels for all N = n_species*T particles.

    T=1: ['A', 'B', 'C', ...]
    T>1: ['A1', 'A2', 'B1', 'B2', ...]
    """
    if T == 1:
        return [SPECIES_NAMES[s] for s in range(n_species)]
    return [f'{SPECIES_NAMES[s]}{t + 1}' for s in range(n_species) for t in range(T)]

_CMAP_CHARGE = plt.cm.RdBu_r
_NORM_CHARGE = Normalize(vmin=-1.0, vmax=1.0)


def _charge_color(q):
    return _CMAP_CHARGE(_NORM_CHARGE(float(q)))


def _node_positions(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------------------------------------------------------------------------
# Colour palettes — enough for 26+ monomers and dimers
# ---------------------------------------------------------------------------

_TAB20B = plt.cm.tab20b(np.linspace(0, 1, 20))
_TAB20C = plt.cm.tab20c(np.linspace(0, 1, 20))
_PALETTE = np.concatenate([_TAB20B, _TAB20C], axis=0)

# Visually distinct colors for correct dimers — consistent across all panels
_CORRECT_DIMER_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
    '#c0392b', '#2980b9', '#27ae60', '#8e44ad',
    '#16a085', '#d35400', '#7f8c8d', '#2c3e50',
]


def _correct_dimer_color(correct_idx):
    """Color for the correct_idx-th correct dimer (0-based)."""
    return _CORRECT_DIMER_COLORS[correct_idx % len(_CORRECT_DIMER_COLORS)]


def _species_color(i):
    return _PALETTE[i % len(_PALETTE)]


def _dimer_color(ii, jj, n, correct_mask_np, T=1):
    if correct_mask_np[ii, jj]:
        greens = ['#27ae60', '#1abc9c', '#2ecc71', '#16a085', '#0e6655']
        pair_idx = min(ii, jj) // (2 * T)  # species pair index (robust to T>1)
        return greens[pair_idx % len(greens)]
    else:
        greys = ['#7f8c8d', '#95a5a6', '#bdc3c7', '#a04000',
                 '#784212', '#6e2f1a', '#717d7e', '#808b96']
        k = sum(1 for a in range(n) for b in range(a, n) if (a, b) < (ii, jj))
        return greys[k % len(greys)]


# ---------------------------------------------------------------------------
# pH(t) helper
# ---------------------------------------------------------------------------

def _ph_trace(t_arr, equil_duration, pH_schedule, duration_per_seg, smooth_width):
    """Compute pH at each time in t_arr (numpy array)."""
    ph = np.zeros_like(t_arr)
    for idx, t in enumerate(t_arr):
        if t <= equil_duration:
            ph[idx] = 7.0
        else:
            t_rel  = t - equil_duration
            seg    = int(t_rel / duration_per_seg)
            seg    = min(seg, len(pH_schedule) - 1)
            t_seg  = t_rel - seg * duration_per_seg
            target = float(pH_schedule[seg])
            prev   = float(pH_schedule[seg - 1]) if seg > 0 else 7.0
            if smooth_width > 0:
                w     = float(smooth_width)
                blend = 1.0 / (1.0 + np.exp(-(t_seg - w * 0.5) / (w * 0.2 + 1e-8)))
                ph[idx] = prev + (target - prev) * blend
            else:
                ph[idx] = target
    return ph


# ---------------------------------------------------------------------------
# Final-concentration bar chart
# ---------------------------------------------------------------------------

def plot_final_concentrations(state, n, acid_base, correct_mask_np,
                               title='', save_path=None,
                               n_species=None, T=1):
    from .dynamics import make_triu_indices
    if n_species is None:
        n_species = n
    plabels  = _particle_labels(n_species, T)
    i_idx, j_idx = make_triu_indices(n)
    free       = np.array(state[:n])
    dimer_triu = np.array(state[n:])
    labels, values, colors = [], [], []
    for i in range(n):
        labels.append(plabels[i])
        values.append(free[i])
        colors.append(_species_color(i))
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        labels.append(f'{plabels[ii]}–{plabels[jj]}')
        values.append(dimer_triu[k])
        colors.append(_dimer_color(ii, jj, n, correct_mask_np, T))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 4.5))
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8.5)
    ax.set_ylabel('Concentration')
    ax.set_title(title or 'Final concentrations')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Network animation
# ---------------------------------------------------------------------------

def animate_crn(traj_list, n, acid_base, correct_mask_np,
                pH_schedule, duration_per_seg,
                pKa_visual=None, output_path='animation.gif', fps=15,
                n_species=None, T=1):
    from .dynamics import make_triu_indices
    if n_species is None:
        n_species = n
    plabels  = _particle_labels(n_species, T)
    i_idx, j_idx = make_triu_indices(n)
    pos          = _node_positions(n)
    all_states   = np.concatenate([np.array(tr) for tr in traj_list], axis=0)
    n_frames     = all_states.shape[0]
    n_seg        = len(pH_schedule)
    t_all        = np.linspace(0.0, n_seg * duration_per_seg, n_frames)
    if pKa_visual is None:
        pKa_visual = np.array([5.5 if acid_base[i] == 1 else 8.5 for i in range(n)])

    fig = plt.figure(figsize=(14, 6), facecolor='#1a1a2e')
    ax_net  = fig.add_subplot(1, 2, 1, facecolor='#16213e')
    ax_conc = fig.add_subplot(1, 2, 2, facecolor='#16213e')

    for ax in (ax_net, ax_conc):
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
        ax.tick_params(colors='#cccccc', labelsize=8)
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.title.set_color('white')

    ax_net.set_xlim(-1.55, 1.55)
    ax_net.set_ylim(-1.55, 1.55)
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    ax_net.set_title('CRN Network', fontsize=13)

    edge_artists = {}
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        if ii == jj:
            continue
        col = _dimer_color(ii, jj, n, correct_mask_np)
        line, = ax_net.plot(
            [pos[ii, 0], pos[jj, 0]], [pos[ii, 1], pos[jj, 1]],
            '-', color=col, linewidth=0.1, alpha=0.85, zorder=1)
        edge_artists[(ii, jj)] = line

    homo_rings = {}
    for i in range(n):
        circ = plt.Circle(pos[i], 0.17, fill=False, edgecolor='#aaaaaa',
                          linewidth=0.3, linestyle='--', zorder=2)
        ax_net.add_patch(circ)
        homo_rings[i] = circ

    node_circles = []
    for i in range(n):
        circ = plt.Circle(pos[i], 0.13, zorder=3, linewidth=1.5, edgecolor='white')
        ax_net.add_patch(circ)
        node_circles.append(circ)
        ax_net.text(pos[i, 0], pos[i, 1], plabels[i],
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white', zorder=4)

    pH_txt   = ax_net.text(-1.5, -1.5, '', fontsize=12, color='#f39c12', zorder=5)
    time_txt = ax_net.text(-1.5,  1.45, '', fontsize=9,  color='#aaaaaa', zorder=5)

    sm = plt.cm.ScalarMappable(cmap=_CMAP_CHARGE, norm=_NORM_CHARGE)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_net, fraction=0.035, pad=0.01, shrink=0.6)
    cbar.set_label('Charge q', color='#cccccc', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='#cccccc', labelcolor='#cccccc')

    ax_conc.set_xlim(0, t_all[-1])
    ax_conc.set_ylim(0, 1.05)
    ax_conc.set_xlabel('Time', fontsize=11)
    ax_conc.set_ylabel('Concentration', fontsize=11)
    ax_conc.set_title('Species concentrations', fontsize=13)
    ax_conc.grid(alpha=0.15, color='#555577')

    conc_lines = []
    for i in range(n):
        ln, = ax_conc.plot([], [], '-', color=_species_color(i), linewidth=1.8,
                           label=f'[{plabels[i]}]', alpha=0.9)
        conc_lines.append(ln)
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        col = _dimer_color(ii, jj, n, correct_mask_np, T)
        lw  = 2.0 if correct_mask_np[ii, jj] else 0.7
        lbl = f'[{plabels[ii]}–{plabels[jj]}]' if correct_mask_np[ii, jj] else None
        ln, = ax_conc.plot([], [], '-', color=col, linewidth=lw, label=lbl,
                           alpha=0.85 if lbl else 0.4)
        conc_lines.append(ln)

    ax_conc.legend(fontsize=8, loc='upper right',
                   facecolor='#16213e', labelcolor='white',
                   framealpha=0.8, ncol=2)

    seg_colors = ['#2980b9', '#27ae60', '#c0392b', '#8e44ad', '#d35400', '#16a085']
    for s_i, pH_v in enumerate(pH_schedule):
        t0, t1 = s_i * duration_per_seg, (s_i + 1) * duration_per_seg
        ax_conc.axvspan(t0, t1, alpha=0.07, color=seg_colors[s_i % len(seg_colors)])
        ax_conc.text((t0 + t1) / 2, 0.99, f'pH {pH_v:.1f}',
                     ha='center', va='top', fontsize=7.5, color='white', alpha=0.65)

    time_line = ax_conc.axvline(0, color='#f39c12', linewidth=1.5, alpha=0.9)

    def init():
        for ln in conc_lines:
            ln.set_data([], [])
        time_line.set_xdata([0, 0])
        return conc_lines + [time_line, pH_txt, time_txt] + node_circles

    def update(frame):
        state      = all_states[frame]
        t          = t_all[frame]
        dimer_triu = state[n:]
        seg_idx    = min(int(t / (duration_per_seg + 1e-9)), n_seg - 1)
        current_pH = float(pH_schedule[seg_idx])
        pH_txt.set_text(f'pH = {current_pH:.1f}')
        time_txt.set_text(f't = {t:.1f}')
        time_line.set_xdata([t, t])
        charges = np.array(henderson_hasselbalch(
            jnp.array(pKa_visual), current_pH, jnp.array(acid_base)))
        for i, circ in enumerate(node_circles):
            circ.set_facecolor(_charge_color(charges[i]))
        max_d = float(np.max(dimer_triu)) + 1e-9
        for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
            if ii == jj:
                homo_rings[ii].set_linewidth(max(8.0 * float(dimer_triu[k]) / max_d, 0.1))
            else:
                edge_artists[(ii, jj)].set_linewidth(max(12.0 * float(dimer_triu[k]) / max_d, 0.05))
        t_sl = t_all[:frame + 1]
        s_sl = all_states[:frame + 1]
        for i, ln in enumerate(conc_lines[:n]):
            ln.set_data(t_sl, s_sl[:, i])
        for k, ln in enumerate(conc_lines[n:]):
            ln.set_data(t_sl, s_sl[:, n + k])
        return conc_lines + [time_line, pH_txt, time_txt] + node_circles

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=False,
        interval=max(10, 1000 // fps))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f'pH schedule: {pH_schedule}', color='white', fontsize=12)
    writer = (animation.PillowWriter(fps=fps) if output_path.lower().endswith('.gif')
              else animation.FFMpegWriter(fps=fps, bitrate=2000))
    ani.save(output_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f'  Animation saved: {output_path}')
    return ani


# ---------------------------------------------------------------------------
# Boltzmann equilibrium helper
# ---------------------------------------------------------------------------

def _boltzmann_equilibrium(pH, pKa_full, acid_base_np, correct_mask_np,
                            phi, J, beta, n, i_idx, j_idx,
                            monomer_entropy=None, allowed_mask_np=None,
                            no_self_bonds=False, n_iter=3000, tol=1e-12):
    """Thermodynamic equilibrium concentrations via fixed-point iteration.

    Solves  x_i = C_i / (1 + (K·x)_i + K_ii·x_i)  where K_ij = exp(-β·ΔG_ij)
    and C_i = 1/N (uniform initial content).  Damped iteration (x ← ½(x + F(x)))
    converges robustly even when K values are large.

    Returns (sum_correct_dimers, sum_incorrect_dimers) at thermodynamic equilibrium.
    """
    charges = np.array(henderson_hasselbalch(
        jnp.array(pKa_full, dtype=float), float(pH),
        jnp.array(acid_base_np, dtype=float)))
    me_jax = (jnp.array(monomer_entropy, dtype=float)
              if monomer_entropy is not None else None)
    am_jax = (jnp.array(allowed_mask_np, dtype=bool)
              if allowed_mask_np is not None else None)
    dG = np.array(interaction_energy_matrix(
        jnp.array(charges, dtype=float),
        jnp.array(correct_mask_np, dtype=bool),
        float(phi),
        jnp.array(J, dtype=float) if np.ndim(J) > 0 else float(J),
        monomer_entropy=me_jax, allowed_mask=am_jax))
    K = np.exp(-float(beta) * dG)
    if no_self_bonds:
        np.fill_diagonal(K, 0.0)
    C = np.ones(n) / n
    x = C.copy()
    K_diag = np.diag(K).copy()
    for _ in range(n_iter):
        Fx = C / (1.0 + K.dot(x) + K_diag * x)
        x_new = 0.5 * (x + Fx)  # damped: prevents oscillation when K is large
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    d_correct = d_incorrect = 0.0
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        d_k = float(K[ii, jj]) * float(x[ii]) * float(x[jj])
        if correct_mask_np[ii, jj]:
            d_correct += d_k
        else:
            d_incorrect += d_k
    return d_correct, d_incorrect


# ---------------------------------------------------------------------------
# Comprehensive summary PNG
# ---------------------------------------------------------------------------

def plot_summary(loss_history, score_history, param_history,
                 all_schedules, target_idx,
                 equil_traj, schedule_trajs, pH_schedule,
                 equil_duration, duration_per_seg,
                 static, trained_params,
                 final_scores,
                 save_path='summary.png',
                 config=None):
    """
    Summary figure.

    Rows (always):
      0 — Loss | pKa history | φ+J history
      1 — Concentration + pH trace (wide, 2/3) | ΔG vs pH
      2 — Bar chart (full width)

    Additional row if monomer_entropy trained:
      before bar chart: entropy history (wide)
    """
    n               = static['n']
    n_species       = static.get('n_species', n)  # = 2*N_acids in M/N model
    T               = static.get('T', 1)          # = 1 in M/N model
    plabels         = _particle_labels(n_species, T)
    i_idx, j_idx    = static['i_idx'], static['j_idx']
    correct_mask_np = static['correct_mask_np']
    acid_base       = np.array(static['acid_base'])
    pKa             = np.array(trained_params['pKa'])  # shape (n_species,) = (2*N_acids,)
    pKa_full        = pKa                               # one value per particle
    phi             = float(trained_params['phi'])
    J_raw           = trained_params['J']
    J_arr           = np.array(J_raw)
    J               = float(J_arr) if J_arr.ndim == 0 else None  # None → per-pair matrix
    S_max           = float(static.get('S_max', 0.0))
    sw              = float(static.get('smooth_width', 0.0))
    J_max           = float(static.get('J_max', 3.5))

    has_entropy = (S_max > 0.0 and 'monomer_entropy' in trained_params
                   and trained_params['monomer_entropy'] is not None)

    epochs     = np.arange(len(loss_history))
    score_arr  = np.array(score_history)
    pKa_hist   = np.array([p['pKa'] for p in param_history])
    phi_hist   = np.array([p['phi'] for p in param_history])
    J_hist     = np.array([float(np.mean(np.array(p['J']))) if np.ndim(np.array(p['J'])) > 0
                           else float(p['J']) for p in param_history])

    n_rows       = 4 if has_entropy else 3
    h_ratios     = ([1.0, 2.5, 1.2, 1.2] if has_entropy
                    else [1.0, 2.5, 1.2])
    fig_h        = sum(h_ratios) * 3.2

    fig = plt.figure(figsize=(20, fig_h))
    gs  = fig.add_gridspec(n_rows, 3,
                           height_ratios=h_ratios,
                           hspace=0.45, wspace=0.32,
                           left=0.06, right=0.97, top=0.96, bottom=0.04)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_pka  = fig.add_subplot(gs[0, 1])
    ax_phiJ = fig.add_subplot(gs[0, 2])
    ax_conc = fig.add_subplot(gs[1, 0:2])
    ax_dG   = fig.add_subplot(gs[1, 2])
    if has_entropy:
        ax_ent    = fig.add_subplot(gs[2, 0:2])
        ax_params = fig.add_subplot(gs[2, 2])
        ax_bar    = fig.add_subplot(gs[3, 0:3])
    else:
        ax_params = fig.add_subplot(gs[2, 2])
        ax_bar    = fig.add_subplot(gs[2, 0:2])

    ls_cycle = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    # -------------------------------------------------------------------
    # Panel 1 — Loss
    # -------------------------------------------------------------------
    ax_loss.plot(epochs, loss_history, color='#2c3e50', linewidth=2)
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss  (−log p_target)', fontsize=10)
    ax_loss.set_title('Training Loss', fontsize=12)
    ax_loss.grid(alpha=0.25)
    # Annotate learning rate and seed on the loss panel
    _lr   = (config or {}).get('learning_rate', None)
    _seed = (config or {}).get('seed', None)
    _loss_ann = []
    if _lr   is not None: _loss_ann.append(f'lr = {_lr}')
    if _seed is not None: _loss_ann.append(f'seed = {_seed}')
    if _loss_ann:
        ax_loss.text(0.98, 0.97, '  '.join(_loss_ann),
                     transform=ax_loss.transAxes, fontsize=7.5,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                               alpha=0.75, edgecolor='#cccccc'))

    # -------------------------------------------------------------------
    # Panel 2 — pKa evolution  (one curve per species, shared across types)
    # -------------------------------------------------------------------
    for i in range(n_species):
        ab = 'base' if acid_base[i] == 1 else 'acid'
        ax_pka.plot(epochs, pKa_hist[:, i],
                    color=_species_color(i),
                    linestyle=ls_cycle[i % len(ls_cycle)],
                    linewidth=2,
                    label=f'{SPECIES_NAMES[i]} ({ab})')
    ax_pka.set_ylim(3, 10)
    ax_pka.set_xlabel('Epoch', fontsize=11)
    ax_pka.set_ylabel('pKa', fontsize=11)
    ax_pka.set_title('pKa evolution', fontsize=12)
    ax_pka.legend(fontsize=8, loc='best')
    ax_pka.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 3 — φ and J evolution
    # -------------------------------------------------------------------
    _fixed_J_plot  = static.get('fixed_J')
    _fixed_phi_plot = static.get('fixed_phi')
    ax_phiJ.plot(epochs, phi_hist, color='#2980b9', linewidth=2, label='φ (steric)')
    ax_phiJ.set_ylim(0, 1.05)
    ax_phiJ.set_xlabel('Epoch', fontsize=11)
    ax_phiJ.set_ylabel('φ', color='#2980b9', fontsize=11)
    ax_phiJ.tick_params(axis='y', labelcolor='#2980b9')
    ax2 = ax_phiJ.twinx()
    j_lbl = 'mean J (kT)' if J is None else 'J (kT)'
    ax2.plot(epochs, J_hist, color='#c0392b', linewidth=2, label=j_lbl)
    if _fixed_J_plot is not None:
        j_axis_lbl = f'J  (kT, fixed={_fixed_J_plot:.2g})'
        j_lo = max(0.0, float(_fixed_J_plot) * 0.9 - 0.5)
        j_hi = float(_fixed_J_plot) * 1.1 + 0.5
        ax2.set_ylim(j_lo, j_hi)
    else:
        j_axis_lbl = f'J  (kT, cap {J_max:.1f})'
        ax2.set_ylim(0, J_max * 1.08)
    ax2.set_ylabel(j_axis_lbl, color='#c0392b', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax_phiJ.set_title('φ and J evolution', fontsize=12)
    h1, l1 = ax_phiJ.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_phiJ.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')
    ax_phiJ.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 4 — Concentration time series with pH overlay
    # -------------------------------------------------------------------
    equil_states = np.array(equil_traj)
    t_equil      = np.linspace(0.0, equil_duration, len(equil_states))
    sched_states = np.concatenate([np.array(tr) for tr in schedule_trajs], axis=0)
    t_sched      = np.linspace(equil_duration,
                               equil_duration + len(pH_schedule) * duration_per_seg,
                               len(sched_states))
    all_st  = np.concatenate([equil_states, sched_states], axis=0)
    t_all   = np.concatenate([t_equil, t_sched])
    n_triu  = len(i_idx)

    # Symlog scale: linear below 1e-3, log above
    ax_conc.set_yscale('symlog', linthresh=1e-3, linscale=0.4)
    ax_conc.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Correct dimers only — with distinct colors consistent across all panels
    correct_count = 0
    for k in range(n_triu):
        ii, jj = int(i_idx[k]), int(j_idx[k])
        if not correct_mask_np[ii, jj]:
            continue
        col = _correct_dimer_color(correct_count)
        ax_conc.plot(t_all, all_st[:, n + k],
                     color=col,
                     linestyle=ls_cycle[correct_count % len(ls_cycle)],
                     linewidth=2.2,
                     label=f'[{plabels[ii]}–{plabels[jj]}] ✓')
        correct_count += 1

    # Total monomer content M(t) ≡ 1
    M_t = all_st[:, :n].sum(1) + 2.0 * all_st[:, n:].sum(1)
    ax_conc.plot(t_all, M_t, 'k--', linewidth=2.0,
                 label='M(t) total (≡1)')

    # Aggregate dimer lines
    correct_triu   = [k for k, (ii, jj) in enumerate(zip(i_idx, j_idx))
                      if correct_mask_np[ii, jj]]
    incorrect_triu = [k for k, (ii, jj) in enumerate(zip(i_idx, j_idx))
                      if not correct_mask_np[ii, jj]]
    sum_correct    = (all_st[:, n:][:, correct_triu].sum(axis=1)
                      if correct_triu else np.zeros(len(t_all)))
    sum_incorrect  = (all_st[:, n:][:, incorrect_triu].sum(axis=1)
                      if incorrect_triu else np.zeros(len(t_all)))
    # Maximum possible: all classifier-pair monomer in correct dimers → M/(2N)
    M_clf   = static.get('M_classifier', n // 2)
    N_acids = static.get('N_total', n // 2)
    max_correct = M_t * M_clf / (2 * N_acids)

    ax_conc.plot(t_all, sum_correct,   color='#27ae60', linewidth=3.5,
                 label='Σ correct dimers', zorder=5)
    ax_conc.plot(t_all, sum_incorrect, color='#e74c3c', linewidth=3.5,
                 label='Σ incorrect dimers', zorder=5)
    ax_conc.plot(t_all, max_correct,   color='#aaaaaa', linewidth=2.5,
                 linestyle='--', label=f'max possible correct (M={M_clf}/2N={2*N_acids})', zorder=4)

    # Segment shading
    seg_colors = ['#2980b9', '#27ae60', '#c0392b', '#8e44ad', '#d35400']
    ax_conc.axvspan(0, equil_duration, alpha=0.04, color='grey')
    for s_i, pH_v in enumerate(pH_schedule):
        t0 = equil_duration + s_i * duration_per_seg
        t1 = equil_duration + (s_i + 1) * duration_per_seg
        ax_conc.axvspan(t0, t1, alpha=0.07, color=seg_colors[s_i % len(seg_colors)])
    ax_conc.axvline(equil_duration, color='grey', linewidth=1.0, linestyle=':', alpha=0.6)

    # --- Boltzmann thermodynamic equilibrium reference lines ----------------
    # For each pH segment (including equilibration at pH 7), solve the
    # mean-field equilibrium and overlay dotted horizontal lines so the user
    # can see how far the ODE trajectory is from the thermodynamic limit.
    _beta_eq = float(static.get('beta', 1.0))
    _no_sb   = bool(static.get('no_self_bonds', False))
    _am_eq   = (np.array(static['allowed_mask'])
                if static.get('allowed_mask') is not None else None)
    _phi_eq  = float(trained_params['phi'])
    _J_eq    = trained_params['J']
    _me_eq   = trained_params.get('monomer_entropy', None)

    ph_segs = [(7.0, 0.0, equil_duration)] + [
        (float(pH_schedule[si]),
         equil_duration + si * duration_per_seg,
         equil_duration + (si + 1) * duration_per_seg)
        for si in range(len(pH_schedule))]

    _eq_legend_done = False
    for ph_val, ts0, ts1 in ph_segs:
        eq_c, eq_i = _boltzmann_equilibrium(
            ph_val, pKa_full, acid_base, correct_mask_np,
            _phi_eq, _J_eq, _beta_eq, n, i_idx, j_idx,
            monomer_entropy=_me_eq, allowed_mask_np=_am_eq,
            no_self_bonds=_no_sb)
        lbl_c = 'Σ correct (eq. from dissoc.)'   if not _eq_legend_done else None
        lbl_i = 'Σ incorrect (eq. from dissoc.)' if not _eq_legend_done else None
        _eq_legend_done = True
        x_mid = 0.5 * (ts0 + ts1)
        ax_conc.plot([ts0, ts1], [eq_c, eq_c], color='#27ae60',
                     linestyle=':', linewidth=2.0, zorder=6, alpha=0.85,
                     label=lbl_c)
        ax_conc.plot([ts0, ts1], [eq_i, eq_i], color='#e74c3c',
                     linestyle=':', linewidth=2.0, zorder=6, alpha=0.85,
                     label=lbl_i)
        ax_conc.text(x_mid, eq_c, f'{eq_c:.3g}', color='#27ae60',
                     fontsize=6.5, ha='center', va='bottom', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.1',
                               facecolor='white', alpha=0.65, edgecolor='none'))
        ax_conc.text(x_mid, eq_i, f'{eq_i:.3g}', color='#e74c3c',
                     fontsize=6.5, ha='center', va='top', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.1',
                               facecolor='white', alpha=0.65, edgecolor='none'))
    # -----------------------------------------------------------------------

    ax_conc.set_xlabel('Time', fontsize=11)
    ax_conc.set_ylabel('Concentration (symlog)', fontsize=11)
    ax_conc.set_title(f'Concentrations — target schedule {pH_schedule}', fontsize=12)
    ax_conc.legend(fontsize=6.5, loc='upper right', ncol=3,
                   bbox_to_anchor=(1.0, 1.0), framealpha=0.85)
    ax_conc.grid(alpha=0.2, which='both')

    # pH twin axis — shows schedule and smoothing
    ax_pH = ax_conc.twinx()
    ph_values = _ph_trace(t_all, equil_duration, pH_schedule,
                          duration_per_seg, sw)
    ax_pH.plot(t_all, ph_values, color='#8e44ad', linewidth=2.5, alpha=0.8,
               linestyle='-', label='pH(t)', zorder=20)
    ax_pH.set_ylabel('pH', color='#8e44ad', fontsize=11)
    ax_pH.tick_params(axis='y', labelcolor='#8e44ad')
    all_pH_vals = [7.0] + list(pH_schedule)
    lo, hi = min(all_pH_vals) - 0.5, max(all_pH_vals) + 0.5
    ax_pH.set_ylim(lo, hi)
    # Small pH segment labels
    ax_pH.text(equil_duration / 2, hi - 0.1, 'equil pH 7',
               ha='center', va='top', fontsize=7, color='#8e44ad', alpha=0.7)
    for s_i, pH_v in enumerate(pH_schedule):
        t0 = equil_duration + s_i * duration_per_seg
        t1 = equil_duration + (s_i + 1) * duration_per_seg
        ax_pH.text((t0 + t1) / 2, hi - 0.1, f'pH {pH_v:.0f}',
                   ha='center', va='top', fontsize=8, color='#8e44ad')

    # -------------------------------------------------------------------
    # Panel 5 — ΔG vs pH
    # -------------------------------------------------------------------
    pHs = np.linspace(2, 12, 300)
    mono_entropy = trained_params.get('monomer_entropy', None)
    mono_arr     = (np.array(mono_entropy) if mono_entropy is not None else None)

    correct_count_dG = 0
    for k in range(n_triu):
        ii, jj = int(i_idx[k]), int(j_idx[k])
        # Only show correct bonds in the dG panel
        if not correct_mask_np[ii, jj]:
            continue
        qs = np.array([
            float(henderson_hasselbalch(jnp.array(pKa_full), ph, jnp.array(acid_base))[ii]) *
            float(henderson_hasselbalch(jnp.array(pKa_full), ph, jnp.array(acid_base))[jj])
            for ph in pHs
        ])
        J_ij    = float(J_arr[ii, jj]) if J is None else J
        dG_line = J_ij * qs
        if mono_arr is not None:
            si = float(mono_arr[0] if len(mono_arr) == 1 else mono_arr[ii])
            sj = float(mono_arr[0] if len(mono_arr) == 1 else mono_arr[jj])
            dG_line = dG_line + si + sj
        col = _correct_dimer_color(correct_count_dG)
        lbl = f'{plabels[ii]}–{plabels[jj]} ✓'
        ax_dG.plot(pHs, dG_line, color=col, linewidth=2.2, linestyle='-', label=lbl)
        correct_count_dG += 1
    ax_dG.axhline(0, color='black', linewidth=0.7)
    for pH_v in pH_schedule:
        ax_dG.axvline(pH_v, color='#e74c3c', linewidth=0.9, linestyle=':', alpha=0.7)
    ax_dG.set_xlabel('pH', fontsize=11)
    ax_dG.set_ylabel('ΔG  (kT)', fontsize=11)
    ax_dG.set_title('Dimer free energy vs pH', fontsize=11)
    ax_dG.legend(fontsize=7, loc='best', ncol=2)
    ax_dG.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 6 (optional) — entropy history
    # -------------------------------------------------------------------
    if has_entropy:
        ent_hist = np.array([p.get('monomer_entropy', np.zeros(1)) for p in param_history])
        for k in range(ent_hist.shape[1]):
            lbl = (SPECIES_NAMES[k] if ent_hist.shape[1] > 1 else 'shared')
            col = (_species_color(k) if ent_hist.shape[1] > 1 else '#8e44ad')
            ax_ent.plot(epochs, ent_hist[:, k], color=col,
                        linestyle=ls_cycle[k % len(ls_cycle)],
                        linewidth=1.8, label=f's({lbl})')
        ax_ent.set_xlabel('Epoch', fontsize=11)
        ax_ent.set_ylabel('Monomer entropy  s  (kT)', fontsize=11)
        ax_ent.set_title(f'Conformational entropy parameters (max {S_max:.1f} kT)', fontsize=12)
        ax_ent.legend(fontsize=8, loc='best', ncol=max(1, n_species // 4))
        ax_ent.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Bar chart — all schedule permutations + pH-7 baseline
    # final_scores layout: [sched_0, ..., sched_K, baseline]
    # -------------------------------------------------------------------
    n_scheds      = len(all_schedules)
    sched_scores  = final_scores[:n_scheds]
    baseline_val  = float(final_scores[n_scheds]) if len(final_scores) > n_scheds else None

    all_vals    = list(sched_scores) + ([baseline_val] if baseline_val is not None else [])
    all_labels  = [str(s) for s in all_schedules] + (['pH 7 (baseline)'] if baseline_val is not None else [])
    colors_bar  = ['#27ae60' if i == target_idx else '#e74c3c'
                   for i in range(n_scheds)]
    if baseline_val is not None:
        colors_bar.append('#95a5a6')   # grey for baseline

    xs   = np.arange(len(all_vals))
    bars = ax_bar.bar(xs, all_vals, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(all_labels, rotation=40, ha='right', fontsize=9)
    ax_bar.set_ylabel('Correct-bond fraction', fontsize=11)
    ax_bar.set_title('Response to all pH-schedule permutations and pH-7 baseline  '
                     '(green = target, red = others, grey = baseline)', fontsize=12)
    ax_bar.set_ylim(0, min(1.05, max(all_vals) * 1.25 + 0.02))
    ax_bar.grid(axis='y', alpha=0.25)
    for b, v in zip(bars, all_vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    legend_handles = [
        mpatches.Patch(color='#27ae60', label='Target schedule'),
        mpatches.Patch(color='#e74c3c', label='Other permutations'),
    ]
    if baseline_val is not None:
        legend_handles.append(mpatches.Patch(color='#95a5a6', label='pH 7 baseline'))
    ax_bar.legend(handles=legend_handles, fontsize=9)

    # -------------------------------------------------------------------
    # Parameter table panel (ax_params)
    # -------------------------------------------------------------------
    ax_params.axis('off')
    cfg = config or {}
    _fp  = static.get('fixed_phi')
    _fJ  = static.get('fixed_J')
    _fpK = static.get('fixed_pKa')   # list or None
    _nb  = bool(static.get('no_baseline', False))

    def _fv(val, fmt='.4g'):
        return format(float(val), fmt) if val is not None else '—'

    # Trained final values
    _phi_val = float(trained_params['phi'])
    _J_val   = float(np.mean(np.array(trained_params['J'])))
    _pKa_np  = np.array(trained_params['pKa'])

    N_total = static.get('N_total', n_species // 2)
    M_clf   = static.get('M_classifier', N_total)
    lines = ['─' * 28,
             ' Parameters',
             '─' * 28,
             '',
             ' System',
             f'   N_acids     {N_total}',
             f'   M_clf_pairs {M_clf}',
             f'   N_roughness {N_total - M_clf}',
             f'   n_particles {n}',
             '',
             ' Schedule',
             f'   target pH   {list(pH_schedule)}',
             f'   equil dur   {static["equil_duration"]:.0f}',
             f'   seg dur     {duration_per_seg:.0f}',
             f'   n_segs      {len(pH_schedule)}',
             '',
             ' Training',
             f'   epochs      {len(loss_history)}',
             f'   lr          {cfg.get("learning_rate", "—")}',
             f'   seed        {cfg.get("seed", "—")}',
             f'   tau         {static.get("tau", 5.0):.1f}',
             '',
             ' Physics',
             f'   beta        {static["beta"]:.2f}',
             f'   k0          {static["k0"]:.3g}',
             f'   J_max       {J_max:.2f}  kT',
             f'   smooth_w    {float(static.get("smooth_width", 0.0)):.2f}',
             '',
             ' Parameters (final)',
    ]
    # pKa lines — n_species = 2*N_total in M/N model, T=1
    for i in range(n_species):
        role = 'base' if acid_base[i] == 1 else 'acid'
        tag  = '  [fixed]' if _fpK is not None else ''
        lines.append(f'   pKa {SPECIES_NAMES[i % 26]} ({role})  {float(_pKa_np[i]):.4f}{tag}')
    lines.append(f'   phi         {_phi_val:.4f}'
                 + ('  [fixed]' if _fp is not None else ''))
    lines.append(f'   J           {_J_val:.4f}  kT'
                 + ('  [fixed]' if _fJ is not None else ''))
    lines += [
        '',
        ' Flags',
        f'   no_baseline {_nb}',
        f'   pka_default {_fpK is not None}',
        '─' * 28,
    ]

    ax_params.text(0.04, 0.98, '\n'.join(lines),
                   transform=ax_params.transAxes,
                   fontsize=7.2, va='top', ha='left',
                   family='monospace',
                   bbox=dict(boxstyle='round,pad=0.4',
                             facecolor='#f8f9fa', alpha=0.92,
                             edgecolor='#cccccc', linewidth=0.8))

    fig.suptitle('CRN_AD — Training Summary', fontsize=15, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    plt.close(fig)
    return save_path
