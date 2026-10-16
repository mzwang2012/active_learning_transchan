# active_learning_transchan

Active learning of the **probability distribution of transition time** in
transitional channel flow, using an existing collection of direct numerical
simulations (DNS). The code compares random sampling (MC) with
likelihood-weighted uncertainty sampling (AL), using an ensemble of neural
networks as the surrogate model.

## How it works

1. Read the top/bottom-wall-averaged friction Reynolds number histories and
   extract the first upward crossing of a fixed midpoint:
   `Re_tau = 168`.
   Crossing times are linearly interpolated between saved time steps.
2. Use six amplitudes, `0.001:0.0002:0.002`, and phase `theta` in `[0, pi]`.
   The original prior is independently uniform in amplitude and phase. 
3. Start with six matched cases and train five independently initialized ReLU
   networks, each with two 32-neuron hidden layers. Output normalization is
   fixed from those initial cases. Networks are refit with L-BFGS after each
   added sample; their prediction variance supplies the acquisition uncertainty.
4. Select from the remaining DNS cases. MC draws according to the input
   quadrature masses. AL maximizes
   `variance(x) * p_X(x) / estimated_p_Y(mean(x))`. In the normalized Cartesian
   coordinates, `p_X = 0.002^2 / (pi * 0.001 * a)`. The output-density estimate
   uses the original probability weights; the Jacobian is not applied twice.
5. Repeat for ten matched seeds, from six to 30 selected cases. Estimate the
   output PDF by weighted KDE of surrogate predictions over the 800-case input
   pool, and compare it with the weighted DNS-based reference.

This is retrospective selection from a fixed dataset; the code does not launch
new DNS. The benchmark contains 800 observed transitions.

## Contents

- `data/Re_tau_amp_theta.mat`: wall-averaged Reynolds-number histories and parameters.
- `data/figure_data.npz`: compact numerical data for immediate figure regeneration.
- `active_learning_transchan/`: data preparation, sampling, density estimation,
  and plotting code; no external project checkout is needed.
- `scripts/run_array.sh`: optional cluster job array with no hardcoded account.
- `results/`: the three PNG figures.

## Installation

The pinned environment matches the Python 3.8 environment used for these results:

```bash
conda env create -f environment.yml
conda activate active_learning_transchan
```

Alternatively, in a Python 3.8 environment, run `python -m pip install -r requirements.txt`.
No GPU, TensorFlow, GPy, or external LaTeX installation is required.

## Regenerate the figures

Run from the repository root:

```bash
python -m active_learning_transchan plot
```

This uses bundled numerical results without retraining. It writes:

1. `results/01_convergence.png`: mean and ±1 sample standard deviation of
   `log10(LPE)` across ten trials, through 30 samples.
2. `results/02_sample_locations.png`: the 30 selected cases over the common
   transition-time contour for MC and AL, displayed in phase/amplitude coordinates.
3. `results/03_transition_time_pdf.png`: reference, MC, and AL PDFs at 30 samples.



## Rerun learning from the histories

```bash
python -m active_learning_transchan prepare
python -m active_learning_transchan run
python -m active_learning_transchan plot --from-runs
```

`prepare` writes transition targets and imputation flags into `work/`.
`run` executes twenty experiments (ten MC and ten AL) and records predictions,
selected cases, errors, and optimizer diagnostics there. `plot --from-runs`
requires all twenty experiments and regenerates the same three panels from
the new results. A single experiment can be run with `run --task-id 0`:
IDs 0–9 are MC; IDs 10–19 are AL. Floating-point libraries and dependency
versions can change nonlinear optimization and subsequent selections.

For Slurm, activate the environment, prepare the data, and submit from the root:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION scripts/run_array.sh
# After all array tasks succeed:
python -m active_learning_transchan plot --from-runs
```

Optional CLI arguments include `--data PATH`, `--workdir PATH`, and
`--output PATH`. `work/` and cluster logs are excluded by `.gitignore`.

## Acknowledgment and reference

This code was adapted from Bianca Champenois and Themistoklis P. Sapsis's
[LikelihoodWeightedActiveSampling](https://github.com/biancach/LikelihoodWeightedActiveSampling)
repository. In particular, the weighted KDE and likelihood-weighted acquisition
follow that implementation; this project specializes the workflow to
transition-time distributions in channel flow.

Champenois, B. and Sapsis, T. P. **What are the most informative data points for
predicting extreme events?** *Nonlinear Dynamics* **113**, 34167–34189 (2025).
[https://doi.org/10.1007/s11071-025-11825-6](https://doi.org/10.1007/s11071-025-11825-6).
