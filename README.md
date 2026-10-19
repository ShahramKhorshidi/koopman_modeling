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
demos can be run directly after installation. The data used to train the model
are intentionally not included.

---

## Key features

- DMDc-based learning of a controlled linear centroidal-dynamics model
- Koopman-model integration with a Kalman filter and an MHE
- Estimation of CoM position, linear momentum, and angular momentum
- Included pretrained model and test trajectories for trot, fast trot, jump,
  and bound motions
- RMSE reporting, interactive plotting, and headless figure export
- Unit tests for the Koopman models and estimators

---

## Repository structure

- `src/modeling/`: DMD, DMDc, extended DMDc, and observable definitions
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
- `trot_fast`
- `jump`
- `bound`

The MHE is more computationally demanding than the Kalman filter. During a
quick check, the number of processed samples and the estimation horizon can be
reduced:

```bash
python demo/demo_state_est.py --estimator mhe --motion jump \
    --samples 1000 --horizon 30
```

### (3) Save results without opening a plot window

```bash
python demo/demo_state_est.py --estimator kf --motion trot_fast \
    --save results/trot_fast_kf.png \
    --save_estimate results/trot_fast_kf.dat \
    --no_show
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
   motion.

The original training data are not distributed with this repository. This
keeps the download small and separates the code and reproducible estimation
examples from the dataset used to identify the model. The included pretrained
model is sufficient for running `demo/demo_state_est.py`.

To retrain the model with your own data, provide tab-separated matrices with one
sample per column. The combined-data layout expected by the demo is:

```text
<data_dir>/training_data_state.dat        # x_k,       shape: 9 x N
<data_dir>/training_data_state_prime.dat  # x_{k+1},   shape: 9 x N
<data_dir>/training_data_input.dat        # u_k,       shape: 24 x N
```

For open-loop validation, the same directory must also contain:

```text
<data_dir>/<motion>_test_cent_state.dat   # shape: 9 x T
<data_dir>/<motion>_test_input_vector.dat # shape: 24 x T
```

Then run:

```bash
python demo/demo_dmdc.py \
    --data_dir /path/to/training_and_validation_data \
    --training_source combined \
    --motion trot \
    --model_path results/koopman_dmdc_model.pkl \
    --save results/dmdc_validation.png \
    --no_show
```

Alternatively, `--training_source trajectories` constructs the snapshot
matrices from numbered files such as `0_cent_state.dat` and
`0_input_vector.dat`. Use `--first_trajectory`, `--last_trajectory`, and
`--trim_start` to select and trim those trajectories.

List all learning options with:

```bash
python demo/demo_dmdc.py --help
```

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

If you use this repository in academic work, please cite:

- [Paper on arXiv](https://arxiv.org/abs/2403.13366)
- [DOI](https://doi.org/10.1109/IROS58592.2024.10801750)

```bibtex
@inproceedings{khorshidi2024centroidal,
  author={Khorshidi, Shahram and Dawood, Murad and Bennewitz, Maren},
  title={Centroidal State Estimation Based on the Koopman Embedding for Dynamic Legged Locomotion},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={12832--12839},
  year={2024},
  doi={10.1109/IROS58592.2024.10801750}
}
```
