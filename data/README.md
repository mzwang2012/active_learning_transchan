# Data

`Re_tau_amp_theta.mat` is a compressed MATLAB dataset containing only the
wall-averaged histories and required metadata. All 20,000 time steps and 800
cases are retained. Histories use single precision; time and parameter arrays
retain their original precision. The learning workflow reads these fields:

| Field | Meaning |
| --- | --- |
| `time` | Saved DNS times |
| `Re_tau_avg` | Top/bottom-wall-averaged friction Reynolds number, time by case, float32 |
| `a`, `theta` | Physical wave amplitude and phase (radians), one per case |
| `member_id` | Ensemble member identifier within an amplitude group |
| `usable_for_sampling` | Flag identifying usable completed histories |

`a_values`, `a_range`, and `theta_range` also record the amplitude levels and
parameter domain. Separate wall histories are not included. The original
double-precision histories were rounded to float32, not downsampled. Calculations
promote these values to float64; this does not recover the discarded precision.
Finite values exceeding float32 range in failed simulations are preserved in
`Re_tau_avg_exception_values` (float64). Their 1-based `(time row, case column)`
locations are in `Re_tau_avg_exception_indices`; the corresponding entries in
`Re_tau_avg` are NaN placeholders. The Python loader restores these exceptions
before processing. In MATLAB, restore them with:

```matlab
s = load('Re_tau_amp_theta.mat');
R = double(s.Re_tau_avg);
ij = double(s.Re_tau_avg_exception_indices);
R(sub2ind(size(R), ij(:,1), ij(:,2))) = s.Re_tau_avg_exception_values;
```

The preparation command extracts midpoint crossing times. It interpolates
missing crossing times in phase within each amplitude group, without modifying
the raw histories. Its CSV output records whether each target was observed,
interpolated from a failed run, or interpolated from a non-crossing run.

`figure_data.npz` contains the numerical results needed to regenerate the three
bundled figures without training. `points` has columns phase and physical
amplitude; `target`, `weights`, and `imputed` describe the 800-case reference.
`log_pdf_error` has axes method (MC, AL), trial, and sample count.
`selected_indices` and `center_pdf` contain the two methods' 30-sample results
for trial 7; indices are zero-based rows of `points`. `pdf_grid` and
`reference_pdf` describe the common density grid and DNS-based reference.
`triangles` preserves the reference contour triangulation.

`manifest.json` records file sizes, SHA-256 checksums, the upstream code snapshot,
and the displayed trial and sample budget.
