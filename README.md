# Koopman-Based Centroidal State Estimation

**Data-driven centroidal dynamics and state estimation for dynamic legged
locomotion.**

This repository demonstrates how a linear model of a legged robot's centroidal
dynamics can be learned from data using Dynamic Mode Decomposition with control
(DMDc) and then used for centroidal-state estimation. The estimated state
contains the center-of-mass (CoM) position, linear momentum, and angular
momentum.

Two estimators use the same learned Koopman/DMDc model:

- **Kalman filter (KF):** recursive state estimation using the learned linear
  dynamics.
- **Moving horizon estimator (MHE):** constrained estimation over a sliding
  window, formulated as a quadratic program and solved with OSQP.

A pretrained model and four test motions are included so that the estimation
demos can be run directly after installation. All provided data were generated
with the Solo12 quadruped performing dynamic locomotion in PyBullet.

---

## Key features

- General Koopman-embedding learning with DMD, DMDc, EDMDc, and EDMDcKIC,
  using configurable identity, polynomial, Fourier, radial-basis, tensor, and
  time-delay observables
- Koopman-model integration with a Kalman filter and an MHE
- Estimation of CoM position, linear momentum, and angular momentum
- Included pretrained model and test trajectories for trot, fast trot, jump,
  and bound motions
- RMSE reporting, interactive plotting, and headless figure export
- Unit tests for the Koopman models and estimators

---

## Repository structure

- `src/modeling/`: general Koopman-embedding models and configurable state and
  input observables
- `src/estimation/`: Kalman filter and moving horizon estimator
- `demo/demo_dmdc.py`: example of learning and validating the DMDc model
- `demo/demo_state_est.py`: runnable Koopman-based centroidal estimation demo
- `tests/`: numerical and interface tests
- `data/`: pretrained model and test trajectories
- `environment.yml`: Conda environment specification

---

## Installation

### (1) Clone the repository

```bash
git clone https://github.com/ShahramKhorshidi/koopman.git
cd koopman
```

### (2) Create the Conda environment

```bash
conda env create -f environment.yml
conda activate koopman_modeling
```

The environment installs Python, NumPy, SciPy, Matplotlib, CVXPY, OSQP,
scikit-learn, and dill.

---

## Run the centroidal-state estimation demos

The included `data/koopman_dmdc_model.pkl` contains the learned dynamics used by
both estimators. No training step is required to run these demos.

### (1) Kalman filter

Run the Kalman filter on the default trot trajectory:

```bash
python demo/demo_state_est.py --estimator kf --motion trot
```

### (2) Moving horizon estimator

Run the MHE on the jump trajectory:

```bash
python demo/demo_state_est.py --estimator mhe --motion jump
```

The available values for `--motion` are:

- `trot`
- `jump`
- `bound`

The MHE is more computationally demanding than the Kalman filter. During a
quick check, the number of processed samples and the estimation horizon can be
reduced:

```bash
python demo/demo_state_est.py --estimator mhe --motion jump \
    --samples 1000 --horizon 30
```

The demo reports the measurement and estimation RMSE for each centroidal-state
block. It also plots the clean reference, noisy measurement, and estimated
state.

List all options with:

```bash
python demo/demo_state_est.py --help
```

---

## Learning the Koopman/DMDc model

`demo/demo_dmdc.py` shows the complete learning procedure:

1. Construct snapshot pairs \(x_k, x_{k+1}\) and inputs \(u_k\) from training
   trajectories.
2. Fit the controlled linear model
   \(x_{k+1} = A x_k + B u_k\) using DMDc.
3. Estimate the process-noise covariance from the training residuals.
4. Save the learned model and validate it by open-loop prediction on a separate
   trajectory.

The included pretrained Solo12 model is sufficient for running
`demo/demo_state_est.py` with the provided PyBullet trajectories.

To retrain the model with your own data, provide tab-separated matrices with one
sample per column. The state and input dimensions are inferred automatically
from the number of rows, so they do not need to match the included centroidal
example. The combined-data layout expected by the demo is:

```text
<data_dir>/training_data_state.dat        # x_k,       shape: n_x x N
<data_dir>/training_data_state_prime.dat  # x_{k+1},   shape: n_x x N
<data_dir>/training_data_input.dat        # u_k,       shape: n_u x N
```

Open-loop validation is optional. To enable it with `--trajectory`, use the
following generic filenames in the same directory:

```text
<data_dir>/<trajectory>_test_state.dat # shape: n_x x T
<data_dir>/<trajectory>_test_input.dat # shape: n_u x T
```

Then run:

```bash
python demo/demo_dmdc.py \
    --data_dir /path/to/training_and_validation_data \
    --training_source combined \
    --trajectory trajectory_01 \
    --model_path results/koopman_dmdc_model.pkl \
    --save results/dmdc_validation.png \
    --no_show
```

Use `--state_dim` and `--input_dim` when you want the demo to check explicit
dimensions instead of only inferring them. For example:

```bash
python demo/demo_dmdc.py \
    --data_dir /path/to/data \
    --state_dim 6 \
    --input_dim 4 \
    --skip_validation
```

`--skip_validation` trains and saves a model without requiring a separate test
trajectory. For validation files that do not follow the default naming scheme,
use `--validation_state_file` and `--validation_input_file`. The validation
plots and per-state RMSE output automatically adapt to `n_x`; use `--dt` to set
the sampling period used on the plot's time axis.

Alternatively, `--training_source trajectories` constructs the snapshot
matrices from numbered files such as `0_state.dat` and `0_input.dat`. Use
`--first_trajectory`, `--last_trajectory`, and
`--trim_start` to select and trim those trajectories.

```bash
python demo/demo_dmdc.py \
    --training_source trajectories \
    --first_trajectory 0 \
    --last_trajectory 30 \
    --trim_start 500
```

This command only trains and saves the model. Open-loop validation runs when
`--trajectory` or both explicit validation-file arguments are also supplied.

List all learning options with:

```bash
python demo/demo_dmdc.py --help
```

The reusable learning demo and the classes under `src/modeling/` support
arbitrary state and input dimensions. `demo/demo_state_est.py` remains specific
to the provided centroidal-estimation example because its state definitions,
sensor-noise values, and plots are physical quantities defined for the included
9-state, 24-input model.

---

## Included test data

Each estimator trajectory contains:

- `<motion>_test_cent_state.dat`: clean centroidal-state reference
- `<motion>_test_cent_state_noisy.dat`: noisy centroidal-state measurement
- `<motion>_test_input_vector_noisy.dat`: noisy contact-force and foot-position
  inputs

The centroidal state has 9 rows: CoM position, linear momentum, and angular
momentum, with three Cartesian components each. The model input has 24 rows:
12 contact-force components followed by 12 foot-position components relative to
the CoM. All provided trajectories are sampled at 1 kHz.

The numbered training trajectories and the combined training matrices are not
included in version control.

---

## Tests

Run the numerical and interface tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

---

## Citation

If you use this repository in academic work, please cite: [paper](https://arxiv.org/abs/2403.13366), [doi](https://doi.org/10.1109/IROS58592.2024.10801750)

```bibtex
@inproceedings{khorshidi24iros,
  title={Centroidal State Estimation based on the Koopman Embedding for Dynamic Legged Locomotion}, 
  author={Khorshidi, Shahram and Dawood, Murad and Bennewitz, Maren},
  booktitle={Proc. of the IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS)}, 
  year={2024}
}
```
