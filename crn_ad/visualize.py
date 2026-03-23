"""
Visualisation: training curves, final concentration bar charts,
and animated network diagrams.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import string

from .physics import henderson_hasselbalch
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIES_NAMES = list(string.ascii_uppercase)   # A, B, C, ...

_CMAP_CHARGE = plt.cm.RdBu_r                   # red = positive, blue = negative
_NORM_CHARGE = Normalize(vmin=-1.0, vmax=1.0)

COLOR_FREE    = '#95a5a6'
COLOR_CORRECT = '#27ae60'
COLOR_WRONG   = '#e67e22'
COLOR_BASE    = '#3498db'   # blue for base-like species label
COLOR_ACID    = '#e74c3c'   # red for acid-like species label


def _charge_color(q):
    """Map fractional charge q ∈ (−1, 1) to RGBA via diverging colourmap."""
    return _CMAP_CHARGE(_NORM_CHARGE(float(q)))


def _node_positions(n):
    """Equally spaced on a unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------

def plot_loss_curve(loss_history, score_history, target_idx, all_schedules,
                    save_path=None):
    """
    Two-panel figure: (top) loss vs epoch, (bottom) per-schedule score.

    The target schedule is drawn bold; all others are thin/translucent.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    epochs = np.arange(len(loss_history))

    # --- Loss ---
    axes[0].plot(epochs, loss_history, 'k-', linewidth=2, label='Loss')
    axes[0].set_ylabel('Loss  (−log p_target)', fontsize=12)
    axes[0].set_title('CRN Training — Softmax Cross-Entropy Loss', fontsize=13)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=10)

    # --- Scores ---
    score_arr = np.array(score_history)   # (n_epochs, n_schedules)
    palette   = plt.cm.tab20(np.linspace(0, 1, len(all_schedules)))

    for s_idx, sched in enumerate(all_schedules):
        is_target = (s_idx == target_idx)
        label     = f'TARGET  {sched}' if is_target else None
        axes[1].plot(
            epochs, score_arr[:, s_idx],
            linewidth=2.5 if is_target else 0.7,
            linestyle='-'  if is_target else '--',
            alpha=1.0       if is_target else 0.35,
            color='#e74c3c' if is_target else palette[s_idx],
            label=label,
            zorder=10       if is_target else 1,
        )

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Correct-bond fraction', fontsize=12)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Final-concentration bar chart
# ---------------------------------------------------------------------------

def plot_final_concentrations(state, n, acid_base, correct_mask_np,
                               title='', save_path=None):
    """
    Horizontal bar chart showing concentrations of all free monomers
    and all dimers at the end of a simulation run.
    """
    from .dynamics import make_triu_indices
    i_idx, j_idx = make_triu_indices(n)

    free       = np.array(state[:n])
    dimer_triu = np.array(state[n:])

    labels, values, colors = [], [], []

    # Free monomers
    for i in range(n):
        labels.append(SPECIES_NAMES[i])
        values.append(free[i])
        colors.append(COLOR_FREE)

    # Dimers
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        lbl = f'{SPECIES_NAMES[ii]}–{SPECIES_NAMES[jj]}'
        labels.append(lbl)
        values.append(dimer_triu[k])
        colors.append(COLOR_CORRECT if correct_mask_np[ii, jj] else COLOR_WRONG)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 4.5))
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8.5)
    ax.set_ylabel('Concentration')
    ax.set_title(title or 'Final concentrations')
    ax.set_ylim(0, max(values) * 1.15 + 1e-4)
    ax.grid(axis='y', alpha=0.25)

    patches = [
        mpatches.Patch(color=COLOR_FREE,    label='Free monomer'),
        mpatches.Patch(color=COLOR_CORRECT, label='Correct dimer'),
        mpatches.Patch(color=COLOR_WRONG,   label='Incorrect dimer'),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Network animation
# ---------------------------------------------------------------------------

def animate_crn(traj_list, n, acid_base, correct_mask_np,
                pH_schedule, duration_per_seg,
                pKa_visual=None,
                output_path='animation.gif',
                fps=15):
    """
    Animated two-panel figure:
      Left  — network graph (nodes = species, edges = dimers, width ∝ concentration)
      Right — concentration time series

    pKa_visual : array (n,) of pKa values used to colour nodes (actual trained
                 values if available, else defaults based on acid/base type).

    Saves to output_path (.gif requires Pillow; .mp4 requires ffmpeg).
    """
    from .dynamics import make_triu_indices

    i_idx, j_idx = make_triu_indices(n)
    pos          = _node_positions(n)

    # Flatten all segments into one continuous trajectory
    all_states = np.concatenate([np.array(tr) for tr in traj_list], axis=0)
    n_frames   = all_states.shape[0]
    n_seg      = len(pH_schedule)
    t_all      = np.linspace(0.0, n_seg * duration_per_seg, n_frames)

    # Default pKa for node colour
    if pKa_visual is None:
        pKa_visual = np.array([5.5 if acid_base[i] == 1 else 8.5 for i in range(n)])
    pKa_visual = np.array(pKa_visual)

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Network panel
    # ------------------------------------------------------------------
    ax_net.set_xlim(-1.55, 1.55)
    ax_net.set_ylim(-1.55, 1.55)
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    ax_net.set_title('CRN Network', fontsize=13)

    # Edge lines (drawn before nodes so nodes sit on top)
    edge_artists = {}   # (ii, jj) -> Line2D
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        if ii == jj:
            continue  # skip self-loops in main panel
        col = COLOR_CORRECT if correct_mask_np[ii, jj] else COLOR_WRONG
        line, = ax_net.plot(
            [pos[ii, 0], pos[jj, 0]],
            [pos[ii, 1], pos[jj, 1]],
            '-', color=col, linewidth=0.1, alpha=0.85, zorder=1,
            solid_capstyle='round',
        )
        edge_artists[(ii, jj)] = line

    # Homodimer rings (small circles around nodes)
    homo_rings = {}
    for i in range(n):
        circ = plt.Circle(pos[i], 0.17, fill=False,
                          edgecolor='#aaaaaa', linewidth=0.3,
                          linestyle='--', zorder=2)
        ax_net.add_patch(circ)
        homo_rings[i] = circ

    # Node circles
    node_circles = []
    for i in range(n):
        circ = plt.Circle(pos[i], 0.13, zorder=3,
                          linewidth=1.5, edgecolor='white')
        ax_net.add_patch(circ)
        node_circles.append(circ)
        ax_net.text(
            pos[i, 0], pos[i, 1], SPECIES_NAMES[i],
            ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', zorder=4,
        )

    pH_txt  = ax_net.text(-1.5, -1.5, '', fontsize=12, color='#f39c12', zorder=5)
    time_txt = ax_net.text(-1.5,  1.45, '', fontsize=9,  color='#aaaaaa', zorder=5)

    # Colorbar legend for node charge
    sm = plt.cm.ScalarMappable(cmap=_CMAP_CHARGE, norm=_NORM_CHARGE)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_net, fraction=0.035, pad=0.01, shrink=0.6)
    cbar.set_label('Charge q', color='#cccccc', fontsize=8)
    cbar.ax.yaxis.set_tick_params(color='#cccccc', labelcolor='#cccccc')

    # ------------------------------------------------------------------
    # Concentration time-series panel
    # ------------------------------------------------------------------
    ax_conc.set_xlim(0, t_all[-1])
    ax_conc.set_ylim(0, 1.05)
    ax_conc.set_xlabel('Time', fontsize=11)
    ax_conc.set_ylabel('Concentration', fontsize=11)
    ax_conc.set_title('Species concentrations', fontsize=13)
    ax_conc.grid(alpha=0.15, color='#555577')

    conc_lines = []
    # Free monomers
    for i in range(n):
        c = COLOR_BASE if acid_base[i] == 1 else COLOR_ACID
        ln, = ax_conc.plot([], [], '-', color=c, linewidth=1.8,
                           label=f'[{SPECIES_NAMES[i]}]', alpha=0.9)
        conc_lines.append(ln)
    # Dimers
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        if correct_mask_np[ii, jj]:
            col = COLOR_CORRECT
            lw  = 2.0
            lbl = f'[{SPECIES_NAMES[ii]}–{SPECIES_NAMES[jj]}]'
        else:
            col = '#6c6c6c'
            lw  = 0.6
            lbl = None
        ln, = ax_conc.plot([], [], '-', color=col, linewidth=lw,
                           label=lbl, alpha=0.75 if lbl else 0.35)
        conc_lines.append(ln)

    ax_conc.legend(fontsize=8, loc='upper right',
                   facecolor='#16213e', labelcolor='white',
                   framealpha=0.8, ncol=2)

    # Segment shading and pH annotations
    seg_colors = ['#2980b9', '#27ae60', '#c0392b', '#8e44ad',
                  '#d35400', '#16a085', '#2c3e50', '#7f8c8d']
    for s_i, pH_v in enumerate(pH_schedule):
        t0 = s_i * duration_per_seg
        t1 = (s_i + 1) * duration_per_seg
        ax_conc.axvspan(t0, t1, alpha=0.07,
                        color=seg_colors[s_i % len(seg_colors)])
        ax_conc.text((t0 + t1) / 2, 0.99, f'pH {pH_v:.1f}',
                     ha='center', va='top', fontsize=7.5,
                     color='white', alpha=0.65)

    # Vertical time indicator
    time_line = ax_conc.axvline(0, color='#f39c12', linewidth=1.5, alpha=0.9)

    # ------------------------------------------------------------------
    # Animation functions
    # ------------------------------------------------------------------
    def init():
        for ln in conc_lines:
            ln.set_data([], [])
        time_line.set_xdata([0, 0])
        return conc_lines + [time_line, pH_txt, time_txt] + node_circles

    def update(frame):
        state      = all_states[frame]
        t          = t_all[frame]
        free       = state[:n]
        dimer_triu = state[n:]

        # Current pH segment
        seg_idx    = min(int(t / (duration_per_seg + 1e-9)), n_seg - 1)
        current_pH = float(pH_schedule[seg_idx])

        # Text overlays
        pH_txt.set_text(f'pH = {current_pH:.1f}')
        time_txt.set_text(f't = {t:.1f}')
        time_line.set_xdata([t, t])

        # Node colours from trained pKa
        charges = np.array(henderson_hasselbalch(
            jnp.array(pKa_visual), current_pH, jnp.array(acid_base)
        ))
        for i, circ in enumerate(node_circles):
            circ.set_facecolor(_charge_color(charges[i]))

        # Edge widths ∝ dimer concentration
        max_d = float(np.max(dimer_triu)) + 1e-9
        for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
            if ii == jj:
                # Update homodimer ring linewidth
                lw = 8.0 * float(dimer_triu[k]) / max_d
                homo_rings[ii].set_linewidth(max(lw, 0.1))
            else:
                lw = 12.0 * float(dimer_triu[k]) / max_d
                edge_artists[(ii, jj)].set_linewidth(max(lw, 0.05))

        # Concentration time series (up to current frame)
        t_sl = t_all[:frame + 1]
        s_sl = all_states[:frame + 1]
        for i, ln in enumerate(conc_lines[:n]):
            ln.set_data(t_sl, s_sl[:, i])
        for k, ln in enumerate(conc_lines[n:]):
            ln.set_data(t_sl, s_sl[:, n + k])

        return conc_lines + [time_line, pH_txt, time_txt] + node_circles

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=False, interval=max(10, 1000 // fps)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f'pH schedule: {pH_schedule}', color='white', fontsize=12)

    if output_path.lower().endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)

    ani.save(output_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f'  Animation saved: {output_path}')
    return ani


# ---------------------------------------------------------------------------
# Quick-look static plot (no training needed)
# ---------------------------------------------------------------------------

def plot_charges_vs_pH(pKa, acid_base, pH_range=(2, 12), save_path=None):
    """
    Plot how each species' charge varies with pH.
    Useful for understanding what the trained pKa values imply.
    """
    n      = len(pKa)
    pHs    = np.linspace(pH_range[0], pH_range[1], 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n):
        qs = np.array([
            float(henderson_hasselbalch(
                jnp.array([pKa[i]]), ph, jnp.array([acid_base[i]])
            )[0])
            for ph in pHs
        ])
        col  = COLOR_BASE if acid_base[i] == 1 else COLOR_ACID
        ax.plot(pHs, qs, color=col, linewidth=2, label=f'{SPECIES_NAMES[i]} (pKa={pKa[i]:.2f})')

    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax.set_xlabel('pH', fontsize=12)
    ax.set_ylabel('Fractional charge q', fontsize=12)
    ax.set_title('Henderson-Hasselbalch charge curves', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# ---------------------------------------------------------------------------
# Comprehensive single-screen summary (static PNG)
# ---------------------------------------------------------------------------

def plot_summary(loss_history, score_history, param_history,
                 all_schedules, target_idx,
                 equil_traj, schedule_trajs, pH_schedule,
                 equil_duration, duration_per_seg,
                 static, trained_params,
                 final_scores,
                 save_path='summary.png'):
    """
    Six-panel summary figure saved as a single PNG.

    Panels
    ------
    (0,0) Training loss vs epoch
    (0,1) pKa values vs epoch (one line per species)
    (0,2) φ and J vs epoch
    (1,0:2) Concentration time series for target schedule
            (includes pH-7 pre-equilibration + schedule segments)
            Black dashed line = total monomer content M(t) — should stay = 1
    (1,2)  Free energy ΔG_{ij}(pH) for all pairs, trained parameters
    (2,0:3) Bar chart: correct-bond fraction under all schedule permutations
    """
    from .dynamics import make_triu_indices
    from .training import total_monomer_content

    n               = static['n']
    i_idx, j_idx    = static['i_idx'], static['j_idx']
    correct_mask_np = static['correct_mask_np']
    acid_base       = np.array(static['acid_base'])
    pKa             = np.array(trained_params['pKa'])
    phi             = float(trained_params['phi'])
    J               = float(trained_params['J'])

    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_pka  = fig.add_subplot(gs[0, 1])
    ax_phiJ = fig.add_subplot(gs[0, 2])
    ax_conc = fig.add_subplot(gs[1, 0:2])
    ax_dG   = fig.add_subplot(gs[1, 2])
    ax_bar  = fig.add_subplot(gs[2, 0:3])

    epochs     = np.arange(len(loss_history))
    score_arr  = np.array(score_history)    # (n_epochs, n_schedules)
    pKa_hist   = np.array([p['pKa'] for p in param_history])  # (n_epochs, n)
    phi_hist   = np.array([p['phi'] for p in param_history])
    J_hist     = np.array([p['J']   for p in param_history])

    # -------------------------------------------------------------------
    # Panel 1: Loss
    # -------------------------------------------------------------------
    ax_loss.plot(epochs, loss_history, 'k-', linewidth=2)
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss  (−log p_target)', fontsize=10)
    ax_loss.set_title('Training Loss', fontsize=12)
    ax_loss.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 2: pKa evolution
    # -------------------------------------------------------------------
    colors_n = plt.cm.tab10(np.linspace(0, 0.9, n))
    for i in range(n):
        ab  = 'base' if acid_base[i] == 1 else 'acid'
        ls  = '-' if acid_base[i] == 1 else '--'
        ax_pka.plot(epochs, pKa_hist[:, i], color=colors_n[i], linestyle=ls,
                    linewidth=2, label=f'{SPECIES_NAMES[i]} ({ab})')
    ax_pka.set_ylim(3, 10)
    ax_pka.set_xlabel('Epoch', fontsize=11)
    ax_pka.set_ylabel('pKa', fontsize=11)
    ax_pka.set_title('pKa evolution', fontsize=12)
    ax_pka.legend(fontsize=8, loc='best')
    ax_pka.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 3: φ and J evolution
    # -------------------------------------------------------------------
    ax_phiJ.plot(epochs, phi_hist, color='#2980b9', linewidth=2, label='φ (steric)')
    ax_phiJ.set_ylim(0, 1.05)
    ax_phiJ.set_xlabel('Epoch', fontsize=11)
    ax_phiJ.set_ylabel('φ', color='#2980b9', fontsize=11)
    ax_phiJ.tick_params(axis='y', labelcolor='#2980b9')
    ax_phiJ2 = ax_phiJ.twinx()
    ax_phiJ2.plot(epochs, J_hist, color='#c0392b', linewidth=2, label='J (kT)')
    ax_phiJ2.set_ylabel('J  (kT)', color='#c0392b', fontsize=11)
    ax_phiJ2.tick_params(axis='y', labelcolor='#c0392b')
    ax_phiJ.set_title('φ and J evolution', fontsize=12)
    lines1, labs1 = ax_phiJ.get_legend_handles_labels()
    lines2, labs2 = ax_phiJ2.get_legend_handles_labels()
    ax_phiJ.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='best')
    ax_phiJ.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 4: Concentration time series (equil + schedule)
    # -------------------------------------------------------------------
    # Build time axis
    equil_states = np.array(equil_traj)    # (n_pts, state_dim)
    t_equil      = np.linspace(0.0, equil_duration, len(equil_states))

    sched_states = np.concatenate([np.array(tr) for tr in schedule_trajs], axis=0)
    n_sched_pts  = len(sched_states)
    t_sched      = np.linspace(equil_duration,
                               equil_duration + len(pH_schedule) * duration_per_seg,
                               n_sched_pts)

    all_states_conc = np.concatenate([equil_states, sched_states], axis=0)
    t_all_conc      = np.concatenate([t_equil, t_sched])

    # Free monomers
    for i in range(n):
        c = COLOR_BASE if acid_base[i] == 1 else COLOR_ACID
        ax_conc.plot(t_all_conc, all_states_conc[:, i],
                     color=c, linewidth=1.8, label=f'[{SPECIES_NAMES[i]}]')

    # Dimers
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        lbl = f'[{SPECIES_NAMES[ii]}–{SPECIES_NAMES[jj]}]'
        if correct_mask_np[ii, jj]:
            ax_conc.plot(t_all_conc, all_states_conc[:, n + k],
                         color=COLOR_CORRECT, linewidth=2.2, label=lbl)
        else:
            ax_conc.plot(t_all_conc, all_states_conc[:, n + k],
                         color='#aaaaaa', linewidth=0.7, alpha=0.5)

    # Mass conservation line  M(t) = Σ[Xi] + 2·Σ[XiXj]
    M_t = (all_states_conc[:, :n].sum(axis=1) +
           2.0 * all_states_conc[:, n:].sum(axis=1))
    ax_conc.plot(t_all_conc, M_t, 'k--', linewidth=1.8,
                 label='M(t) = Σ[Xᵢ] + 2·Σ[XᵢXⱼ]  (should = 1)')

    # Segment shading
    seg_colors = ['#2980b9', '#27ae60', '#c0392b', '#8e44ad', '#d35400']
    ax_conc.axvspan(0, equil_duration, alpha=0.05, color='grey',
                    label='pH 7 equilibration')
    for s_i, pH_v in enumerate(pH_schedule):
        t0 = equil_duration + s_i * duration_per_seg
        t1 = equil_duration + (s_i + 1) * duration_per_seg
        ax_conc.axvspan(t0, t1, alpha=0.08, color=seg_colors[s_i % len(seg_colors)])
        ax_conc.text((t0 + t1) / 2, ax_conc.get_ylim()[1] * 0.97,
                     f'pH {pH_v:.0f}', ha='center', va='top',
                     fontsize=9, color='#333333')

    ax_conc.axvline(equil_duration, color='grey', linewidth=1.0,
                    linestyle=':', alpha=0.7)
    ax_conc.set_xlabel('Time', fontsize=11)
    ax_conc.set_ylabel('Concentration', fontsize=11)
    ax_conc.set_title(f'Concentrations — target schedule {pH_schedule}', fontsize=12)
    ax_conc.legend(fontsize=7.5, loc='upper right', ncol=2)
    ax_conc.grid(alpha=0.2)

    # -------------------------------------------------------------------
    # Panel 5: Free energy ΔG_{ij}(pH)
    # -------------------------------------------------------------------
    pHs = np.linspace(2, 12, 300)
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        qs = np.array([
            float(henderson_hasselbalch(
                jnp.array(pKa), ph, jnp.array(acid_base)
            )[ii]) *
            float(henderson_hasselbalch(
                jnp.array(pKa), ph, jnp.array(acid_base)
            )[jj])
            for ph in pHs
        ])
        phi_fac = 1.0 if correct_mask_np[ii, jj] else phi
        dG_line = J * phi_fac * qs
        lbl = f'{SPECIES_NAMES[ii]}–{SPECIES_NAMES[jj]}'
        if correct_mask_np[ii, jj]:
            ax_dG.plot(pHs, dG_line, linewidth=2.2, color=COLOR_CORRECT,
                       label=lbl + ' ✓')
        else:
            ax_dG.plot(pHs, dG_line, linewidth=0.8, linestyle='--',
                       color='#aaaaaa', alpha=0.6)

    ax_dG.axhline(0, color='black', linewidth=0.7)
    for pH_v in pH_schedule:
        ax_dG.axvline(pH_v, color='#e74c3c', linewidth=0.9, linestyle=':',
                      alpha=0.7)
    ax_dG.set_xlabel('pH', fontsize=11)
    ax_dG.set_ylabel('ΔG  (kT)', fontsize=11)
    ax_dG.set_title('Dimer free energy vs pH\n(trained params, schedule pH shown ·)', fontsize=11)
    ax_dG.legend(fontsize=8)
    ax_dG.grid(alpha=0.25)

    # -------------------------------------------------------------------
    # Panel 6: Bar chart — response to all schedules
    # -------------------------------------------------------------------
    xs     = np.arange(len(all_schedules))
    colors_bar = ['#27ae60' if i == target_idx else '#e74c3c'
                  for i in range(len(all_schedules))]
    bars = ax_bar.bar(xs, final_scores, color=colors_bar, edgecolor='white',
                      linewidth=0.5)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(
        [str(s) for s in all_schedules],
        rotation=40, ha='right', fontsize=9
    )
    ax_bar.set_ylabel('Correct-bond fraction', fontsize=11)
    ax_bar.set_title('Response to all pH-schedule permutations  '
                     '(green = target, red = others)', fontsize=12)
    ax_bar.set_ylim(0, min(1.05, max(final_scores) * 1.25 + 0.02))
    ax_bar.grid(axis='y', alpha=0.25)
    # Annotate bars
    for b, v in zip(bars, final_scores):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    patches = [
        mpatches.Patch(color='#27ae60', label='Target schedule'),
        mpatches.Patch(color='#e74c3c', label='Other permutations'),
    ]
    ax_bar.legend(handles=patches, fontsize=9)

    # -------------------------------------------------------------------
    fig.suptitle('CRN_AD — Training Summary', fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {save_path}')
    plt.close(fig)
    return save_path
