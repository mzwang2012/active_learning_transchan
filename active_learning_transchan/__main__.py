"""Command-line workflows; all default paths are relative to the project root."""
import argparse
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'active-learning-transchan-mpl'))
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'


def main():
    parser = argparse.ArgumentParser(description='Learn transition-time probability distributions from channel-flow data.')
    parser.add_argument('action', choices=['prepare', 'run', 'plot'])
    parser.add_argument('--data', type=Path, default=ROOT/'data/Re_tau_amp_theta.mat')
    parser.add_argument('--workdir', type=Path, default=ROOT/'work')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    parser.add_argument('--task-id', type=int, choices=range(20), help='0-9: MC trials; 10-19: AL trials; omitted: all trials')
    parser.add_argument('--from-runs', action='store_true', help='Plot newly computed trials instead of bundled figure data')
    args = parser.parse_args()
    if args.action == 'prepare':
        from .data import prepare
        prepare(args.data, args.workdir)
    elif args.action == 'run':
        from .sampling import run_task
        for task_id in (range(20) if args.task_id is None else [args.task_id]):
            run_task(args.workdir, task_id)
    else:
        import numpy as np
        from .plotting import plot_figures
        if args.from_runs:
            from .sampling import collect
            figure_data = collect(args.workdir)
            np.savez_compressed(args.workdir/'figure_data.npz', **figure_data)
        else:
            figure_data = np.load(ROOT/'data/figure_data.npz')
        plot_figures(figure_data, args.output)


if __name__ == '__main__':
    main()
