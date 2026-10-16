"""Render convergence, selected locations, and transition-time densities."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.ticker import FormatStrFormatter


def plot_figures(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = ['MC', 'AL']
    colors = ['#808080', '#c92332']
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 18, 'axes.labelsize': 22, 'axes.titlesize': 22,
                         'legend.fontsize': 17, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
                         'axes.linewidth': .8, 'axes.spines.top': True, 'axes.spines.right': True,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'text.usetex': False})

    def save(fig, name):
        fig.savefig(output/(name+'.png'), dpi=450, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    counts = data['sample_counts']
    if int(data['budget']) != 30 or int(data['trial']) != 7:
        raise ValueError('Figure data must describe the 30-sample, trial-7 comparison')
    assert data['log_pdf_error'].shape == (2, 10, len(counts))
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout='constrained')
    for i in range(2):
        values = np.log10(np.maximum(data['log_pdf_error'][i], 1e-15))
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax.fill_between(counts, mean-sd, mean+sd, color=colors[i], alpha=.12, lw=0)
        ax.plot(counts, mean, label=labels[i], color=colors[i], lw=2.2)
    ax.set(xlabel='# of samples', ylabel=r'$\log_{10}(\mathrm{LPE})$',
           xlim=(6, 30), xticks=[6, 10, 15, 20, 25, 30])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(frameon=False)
    save(fig, '01_convergence')

    points = data['points']
    tri = Triangulation(points[:, 0], points[:, 1]/.001, triangles=data['triangles'])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained', sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        im = ax.tricontourf(tri, data['target'], levels=np.linspace(100, 300, 41), cmap='RdBu_r', extend='both')
        selected = data['selected_indices'][i]
        assert len(set(selected)) == 30
        q = points[selected]
        ax.scatter(q[:, 0], q[:, 1]/.001, s=38, c='#171717', edgecolors='white', linewidths=.85, zorder=3, clip_on=False)
        ax.set(title=labels[i], xlabel=r'$\theta$', xlim=(0, np.pi), ylim=(1, 2),
               xticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               xticklabels=['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    axes[0].set_ylabel(r'Amplitude $(\times10^{-3})$')
    fig.colorbar(im, ax=list(axes), label=r'$T_{\mathrm{tr}}$', ticks=[100, 150, 200, 250, 300], format='%d', shrink=.92, pad=.025)
    save(fig, '02_sample_locations')

    grid, reference = data['pdf_grid'], data['reference_pdf']
    fig, ax = plt.subplots(figsize=(5.8, 4), layout='constrained')
    ax.semilogy(grid, np.where(reference >= 1e-8, reference, np.nan), color='black', lw=2, label='Reference')
    for i in range(2):
        density = data['center_pdf'][i]
        ax.semilogy(grid, np.where(density >= 1e-8, density, np.nan), color=colors[i], lw=2, label=labels[i])
    ax.set(xlabel=r'$T_{\mathrm{tr}}$', ylabel='PDF', ylim=(1e-8, .08), xlim=(80, 270), xticks=[80, 120, 160, 200, 240])
    ax.legend(frameon=False)
    save(fig, '03_transition_time_pdf')
    print('Saved three PNG figures to', output, flush=True)
