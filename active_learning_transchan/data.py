"""Extract midpoint transition times and the input probability quadrature."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .density import kde, evaluate_density

AMPLITUDES = np.array([.001, .0012, .0014, .0016, .0018, .002])
LAMINAR = 100.7712786948711
TURBULENT = 234.94738933945877
THRESHOLD = .5 * (LAMINAR + TURBULENT)
SEED = 2026
TRIALS = 10
INITIAL = 6
BUDGET = 30


def cell_weights(nodes, lower, upper):
    nodes = np.asarray(nodes)
    if not np.all(np.diff(nodes) > 0) or nodes[0] < lower or nodes[-1] > upper:
        raise ValueError("Invalid quadrature nodes")
    edges = np.r_[lower, .5 * (nodes[:-1] + nodes[1:]), upper]
    return np.diff(edges) / (upper - lower)


def first_crossings(time, histories):
    if np.any(histories[0] >= THRESHOLD):
        raise ValueError("A history starts above the transition threshold")
    targets = np.full(histories.shape[1], np.nan)
    for j in np.flatnonzero(np.any(histories >= THRESHOLD, axis=0)):
        k = np.flatnonzero(histories[:, j] >= THRESHOLD)[0]
        fraction = (THRESHOLD - histories[k-1, j]) / (histories[k, j] - histories[k-1, j])
        targets[j] = time[k-1] + fraction * (time[k] - time[k-1])
    return targets


def cartesian_inputs(points):
    theta, amplitude = np.asarray(points).T
    return np.column_stack([amplitude * np.cos(theta), amplitude * np.sin(theta)]) / .002


def input_density(points):
    return .002**2 / (np.pi * .001 * points[:, 1])


def initial_indices(points, seed):
    rng = np.random.default_rng(seed)
    lhs = np.column_stack([(rng.permutation(INITIAL) + rng.random(INITIAL)) / INITIAL for _ in range(2)])
    normalized = (points - [0., .001]) / [np.pi, .001]
    selected = []
    for location in lhs:
        distance = np.sum((normalized - location)**2, axis=1)
        distance[selected] = np.inf
        selected.append(int(np.argmin(distance)))
    return selected


def prepare(data_path, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    raw = loadmat(str(data_path), variable_names=['a', 'theta', 'time', 'Re_tau_avg', 'member_id', 'usable_for_sampling',
                                                'Re_tau_avg_exception_indices', 'Re_tau_avg_exception_values'])
    # Rare overflow values in failed DNS runs are retained as double-precision
    # exceptions, with MATLAB-style 1-based (time, case) indices.
    raw['Re_tau_avg'] = raw['Re_tau_avg'].astype(np.float64)
    if 'Re_tau_avg_exception_indices' in raw:
        indices = raw['Re_tau_avg_exception_indices'].astype(int) - 1
        raw['Re_tau_avg'][indices[:, 0], indices[:, 1]] = raw['Re_tau_avg_exception_values'].ravel()
    amplitudes = raw['a'].ravel()
    theta = np.clip(raw['theta'].ravel(), 0, np.pi)
    rows = np.flatnonzero(np.any(np.isclose(amplitudes[:, None], AMPLITUDES[None, :], rtol=0, atol=1e-10), axis=1))
    rows = rows[np.lexsort((theta[rows], amplitudes[rows]))]
    points = np.column_stack([theta[rows], amplitudes[rows]])
    complete = raw['usable_for_sampling'].ravel()[rows].astype(bool)
    targets = np.full(len(rows), np.nan)
    # Histories are stored as float32 to keep the MAT file compact. Perform
    # crossing-time interpolation in float64 (also accepts legacy double data).
    histories = raw['Re_tau_avg'][:, rows[complete]].astype(np.float64)
    if not np.isfinite(histories).all():
        raise ValueError("Nonfinite values in a usable history")
    targets[complete] = first_crossings(raw['time'].ravel(), histories)
    imputed = ~np.isfinite(targets)
    weights = np.zeros(len(rows))
    for amplitude, aw in zip(AMPLITUDES, cell_weights(AMPLITUDES, .001, .002)):
        indices = np.flatnonzero(np.isclose(points[:, 1], amplitude, rtol=0, atol=1e-10))
        good = indices[~imputed[indices]]
        bad = indices[imputed[indices]]
        if len(bad):
            if not (points[bad, 0].min() > points[good, 0].min() and points[bad, 0].max() < points[good, 0].max()):
                raise ValueError("Missing targets would require extrapolation")
            targets[bad] = np.interp(points[bad, 0], points[good, 0], targets[good])
        weights[indices] = aw * cell_weights(points[indices, 0], 0, np.pi)
    weights /= weights.sum()
    estimator = kde(targets, weights=weights)
    bandwidth = float(estimator.bw)
    core_grid = np.linspace(targets.min(), targets.max(), 1024)
    pdf_grid = np.linspace(core_grid[0] - 3*bandwidth, core_grid[-1] + 3*bandwidth, 2048)
    pool = dict(points=points, target=targets, weights=weights, imputed=imputed, complete=complete,
                member_id=raw['member_id'].ravel()[rows].astype(int), threshold=THRESHOLD,
                source_columns_matlab=rows+1, bandwidth=bandwidth, core_grid=core_grid,
                core_reference=evaluate_density(estimator, core_grid), pdf_grid=pdf_grid,
                reference_pdf=evaluate_density(estimator, pdf_grid))
    np.savez_compressed(workdir/'pool.npz', **pool)
    with (workdir/'transition_times.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['theta', 'amplitude', 'member_id', 'transition_time', 'status'])
        for i in range(len(rows)):
            status = 'observed' if not imputed[i] else ('interpolated_non_crossing' if complete[i] else 'interpolated_failure')
            writer.writerow([*points[i], pool['member_id'][i], targets[i], status])
    setup = dict(threshold=THRESHOLD, laminar=LAMINAR, turbulent=TURBULENT, cases=len(rows),
                 observed=int((~imputed).sum()), imputed_failed=int((~complete).sum()),
                 imputed_non_crossing=int((complete & imputed).sum()), seed=SEED,
                 initial=INITIAL, max_samples=BUDGET, trials=TRIALS, methods=['MC', 'AL'])
    (workdir/'setup.json').write_text(json.dumps(setup, indent=2)+'\n')
    print(json.dumps(setup, indent=2), flush=True)
