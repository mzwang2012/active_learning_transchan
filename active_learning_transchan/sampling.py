"""Five-member neural ensembles with random or likelihood-weighted selection."""
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from .data import cartesian_inputs, input_density, initial_indices, SEED, INITIAL, BUDGET, TRIALS
from .density import kde, evaluate_density, acquisition, log_pdf_error


def fit_ensemble(points, target, selected, center, scale, seed):
    x = cartesian_inputs(points[selected])
    z = (target[selected] - center) / scale
    models, iterations = [], []
    nonconverged = 0
    for member in range(5):
        model = MLPRegressor(hidden_layer_sizes=(32, 32), activation='relu', solver='lbfgs',
                             alpha=1e-4, max_iter=1500, max_fun=50000, tol=1e-7,
                             random_state=seed + 1009*member)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, z)
        nonconverged += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        models.append(model)
        iterations.append(int(model.n_iter_))

    def predict(query):
        values = np.array([model.predict(cartesian_inputs(query))*scale + center for model in models])
        return values.mean(axis=0), values.var(axis=0, ddof=1)

    training_mse = float(np.mean((predict(points[selected])[0] - target[selected])**2))
    return predict, dict(iterations=iterations, nonconverged=nonconverged, training_mse=training_mse)


def run_task(workdir, task_id):
    if not 0 <= task_id < 2*TRIALS:
        raise ValueError('task-id must be between 0 and 19')
    workdir = Path(workdir)
    method = ['MC', 'AL'][task_id // TRIALS]
    trial = task_id % TRIALS
    p = np.load(workdir/'pool.npz')
    points, target, weights = p['points'], p['target'], p['weights']
    selected = initial_indices(points, SEED+trial)
    center, scale = float(target[selected].mean()), max(float(target[selected].std()), 1e-6)
    rng = np.random.default_rng(SEED+trial+1000000)
    budgets = np.arange(INITIAL, BUDGET+1)
    errors, means, densities, diagnostics = [], [], [], []
    px = input_density(points)
    for n in budgets:
        predict, diagnostic = fit_ensemble(points, target, selected, center, scale, SEED+10000*trial+int(n))
        mean, variance = predict(points)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError('Nonfinite ensemble prediction')
        estimator = kde(mean, weights=weights, bandwidth=float(p['bandwidth']))
        errors.append(log_pdf_error(evaluate_density(estimator, p['core_grid']), p['core_reference'], p['core_grid']))
        densities.append(evaluate_density(estimator, p['pdf_grid']))
        means.append(mean)
        diagnostics.append(diagnostic)
        print(method, 'trial', trial+1, 'samples', n, 'LPE', errors[-1], flush=True)
        if n == BUDGET:
            break
        available = np.ones(len(target), bool)
        available[selected] = False
        if method == 'MC':
            probabilities = weights * available
            index = int(rng.choice(len(target), p=probabilities/probabilities.sum()))
        else:
            score = acquisition(mean, variance, weights, px)
            if not np.isfinite(score).all():
                raise ValueError('Nonfinite acquisition score')
            score[~available] = -np.inf
            index = int(np.argmax(score))
        selected.append(index)
    assert len(set(selected)) == BUDGET
    destination = workdir/'trials'
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/('%s_%02d.npz' % (method, trial)), sample_counts=budgets,
                        log_pdf_error=errors, predicted_pdf=densities, prediction_mean=means,
                        selected_indices=selected, center=center, scale=scale)
    (destination/('%s_%02d.json' % (method, trial))).write_text(json.dumps(diagnostics, indent=2)+'\n')
    print('FINISHED', method, trial+1, flush=True)


def collect(workdir):
    """Build compact figure data after all twenty experiments finish."""
    from matplotlib.tri import Triangulation
    workdir = Path(workdir)
    p = np.load(workdir/'pool.npz')
    runs = [[np.load(workdir/'trials'/('%s_%02d.npz' % (method, trial)))
             for trial in range(TRIALS)] for method in ['MC', 'AL']]
    for trial in range(TRIALS):
        np.testing.assert_array_equal(runs[0][trial]['selected_indices'][:INITIAL], runs[1][trial]['selected_indices'][:INITIAL])
    points = p['points']
    return dict(points=points, target=p['target'], weights=p['weights'], imputed=p['imputed'],
                triangles=Triangulation(points[:, 0], points[:, 1]/.001).triangles,
                sample_counts=runs[0][0]['sample_counts'],
                log_pdf_error=np.array([[r['log_pdf_error'] for r in method] for method in runs]),
                center_pdf=np.array([method[6]['predicted_pdf'][-1] for method in runs]),
                selected_indices=np.array([method[6]['selected_indices'] for method in runs]),
                pdf_grid=p['pdf_grid'], reference_pdf=p['reference_pdf'], budget=BUDGET, trial=7, seed=2032)
