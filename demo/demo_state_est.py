import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Allow this file to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from src.modeling.koopman import DMDC
from src.estimation.kalman_filter import Kalman_Filter
from src.estimation.moving_horizon import MovingHorizonEstimator


STATE_DIMENSION = 9
INPUT_DIMENSION = 24
SAMPLING_PERIOD = 0.001

# The first 12 inputs are contact forces and the last 12 are foot positions
# relative to the CoM. These values match the simulated sensor noise.
MEASUREMENT_NOISE_STD = np.array([
    1.1e-6, 1.1e-6, 1.4e-6,
    1.31e-1, 1.31e-1, 1.31e-1,
    2.4e-3, 6.5e-3, 7.0e-3,
])
INPUT_NOISE_STD = np.array([2.0] * 12 + [2.0e-5] * 12)


def load_matrix(file_path, expected_rows):
    """Load a data matrix and check its orientation."""
    matrix = np.loadtxt(file_path, delimiter="\t", dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{file_path} must contain {expected_rows} rows; "
            f"received shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{file_path} contains non-finite values")
    return matrix


def load_test_data(data_dir, motion, number_of_samples=None):
    """Load clean references and the noisy estimator measurements."""
    reference = load_matrix(
        data_dir / f"{motion}_test_cent_state.dat", STATE_DIMENSION
    )
    measurement = load_matrix(
        data_dir / f"{motion}_test_cent_state_noisy.dat", STATE_DIMENSION
    )
    control_input = load_matrix(
        data_dir / f"{motion}_test_input_vector_noisy.dat", INPUT_DIMENSION
    )

    available_samples = min(
        reference.shape[1], measurement.shape[1], control_input.shape[1]
    )
    if number_of_samples is None:
        number_of_samples = available_samples
    if number_of_samples < 2 or number_of_samples > available_samples:
        raise ValueError(
            f"--samples must be between 2 and {available_samples}"
        )
    return (
        reference[:, :number_of_samples],
        measurement[:, :number_of_samples],
        control_input[:, :number_of_samples],
    )


def load_koopman_model(model_path):
    """Load a fitted 9-state DMDc model."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No learned model was found at {model_path}. "
            "Run demo_dmdc.py first."
        )
    koopman_model = DMDC.load_model(model_path)
    A, B, C = koopman_model.get_linear_sys_matrices()
    expected_shapes = (
        (STATE_DIMENSION, STATE_DIMENSION),
        (STATE_DIMENSION, INPUT_DIMENSION),
        (STATE_DIMENSION, STATE_DIMENSION),
    )
    if (A.shape, B.shape, C.shape) != expected_shapes:
        raise ValueError(
            "The saved model does not describe the required 9-state/24-input "
            "centroidal system. Run demo_dmdc.py to retrain it."
        )
    return koopman_model


def get_noise_covariances(koopman_model, process_scale, measurement_scale):
    """Construct covariances from training residuals and sensor noise."""
    _, B, _ = koopman_model.get_linear_sys_matrices()

    if hasattr(koopman_model, "process_noise_covariance"):
        model_covariance = np.asarray(
            koopman_model.process_noise_covariance, dtype=float
        )
    elif all(
        getattr(koopman_model, name, None) is not None
        for name in ("X", "X_prime", "U")
    ):
        residual = (
            koopman_model.X_prime
            - koopman_model.A_matrix @ koopman_model.X
            - koopman_model.B_matrix @ koopman_model.U
        )
        model_covariance = np.cov(residual, bias=True)
    else:
        model_covariance = np.diag([
            1e-12, 1e-12, 1e-12,
            1e-4, 1e-4, 4e-4,
            1e-5, 2e-5, 2e-6,
        ])

    input_covariance = np.diag(INPUT_NOISE_STD ** 2)
    effective_process_covariance = (
        model_covariance + B @ input_covariance @ B.T
    )
    process_floor = np.array([
        1e-7, 1e-7, 1e-7,
        1e-4, 1e-4, 1e-4,
        1e-5, 1e-5, 1e-5,
    ]) ** 2
    process_variance = np.maximum(
        np.diag(effective_process_covariance), process_floor
    )
    measurement_variance = MEASUREMENT_NOISE_STD ** 2
    Q = np.diag(process_scale * process_variance)
    R = np.diag(measurement_scale * measurement_variance)
    return Q, R


def create_kalman_filter(koopman_model, Q, R):
    """Create the Koopman Kalman filter using the learned dynamics."""
    A, B, C = koopman_model.get_linear_sys_matrices()
    estimator = Kalman_Filter(A, B, C)
    estimator.set_initial_covariance(R)
    estimator.set_process_noise_cov(Q)
    estimator.set_measurement_noise_cov(R)
    return estimator


def covariance_to_weight_matrices(Q, R):
    """Convert noise covariances to MHE precision matrices."""
    Q_weight = np.diag(1.0 / np.diag(Q))
    R_weight = np.diag(1.0 / np.diag(R))
    return Q_weight, R_weight


def get_state_bounds(koopman_model):
    """Use the training envelope with a margin as physical MHE bounds."""
    if hasattr(koopman_model, "training_state_minimum"):
        lower = np.asarray(koopman_model.training_state_minimum, dtype=float)
        upper = np.asarray(koopman_model.training_state_maximum, dtype=float)
    elif getattr(koopman_model, "X", None) is not None:
        lower = np.min(koopman_model.X, axis=1)
        upper = np.max(koopman_model.X, axis=1)
    else:
        return None
    margin = 0.2 * np.maximum(upper - lower, 1e-6)
    return lower - margin, upper + margin


def create_mhe(koopman_model, horizon, Q, R):
    """Create the MHE using the same learned A, B, and C matrices."""
    A, B, C = koopman_model.get_linear_sys_matrices()
    estimator = MovingHorizonEstimator(
        A,
        B,
        C,
        N=horizon,
        state_bounds=get_state_bounds(koopman_model),
    )
    Q_weight, R_weight = covariance_to_weight_matrices(Q, R)
    estimator.set_arrival_cost_weight(R_weight)
    estimator.set_process_noise_weight(Q_weight)
    estimator.set_measurement_noise_weight(R_weight)
    return estimator


def run_estimation(estimator_name, koopman_model, reference, measurement,
                   control_input, horizon, Q, R):
    """Run the selected estimator with the common learned Koopman model."""
    if estimator_name == "kf":
        estimator = create_kalman_filter(koopman_model, Q, R)
        return koopman_model.kf_estimation(
            estimator, control_input, measurement
        )
    if estimator_name == "mhe":
        if reference.shape[1] <= horizon:
            raise ValueError("The number of samples must be larger than the MHE horizon")
        estimator = create_mhe(koopman_model, horizon, Q, R)
        return koopman_model.mhe_estimation(
            estimator, measurement[:, 0], control_input, measurement
        )
    raise ValueError(f"Unknown estimator '{estimator_name}'")


def print_estimation_errors(estimator_name, reference, measurement, estimate):
    """Print measured-state and estimated-state RMSE values."""
    state_blocks = {
        "CoM position": slice(0, 3),
        "Linear momentum": slice(3, 6),
        "Angular momentum": slice(6, 9),
    }
    print(f"Results for Koopman {estimator_name.upper()}:")
    for name, rows in state_blocks.items():
        measured_rmse = np.sqrt(
            np.mean((reference[rows] - measurement[rows]) ** 2)
        )
        estimated_rmse = np.sqrt(
            np.mean((reference[rows] - estimate[rows]) ** 2)
        )
        print(
            f"{name}: measured RMSE={measured_rmse:.6e}, "
            f"estimated RMSE={estimated_rmse:.6e}"
        )


def plot_estimation_results(time, estimator_name, reference, measurement,
                            estimate, save_path=None, show=True):
    """Plot references, noisy measurements, and Koopman estimates."""
    groups = [
        ("CoM position", "m", slice(0, 3)),
        ("Linear momentum", "kg m/s", slice(3, 6)),
        ("Angular momentum", "kg m^2/s", slice(6, 9)),
    ]
    axes_names = ["x", "y", "z"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)

    for row, (title, unit, rows) in enumerate(groups):
        for column, state_index in enumerate(range(rows.start, rows.stop)):
            axis = axes[row, column]
            axis.plot(time, reference[state_index], "b", label="Reference")
            axis.plot(
                time,
                measurement[state_index],
                color="0.65",
                linewidth=0.6,
                label="Measurement",
            )
            axis.plot(
                time,
                estimate[state_index],
                "r--",
                linewidth=0.9,
                label=f"Koopman {estimator_name.upper()}",
            )
            axis.set_title(f"{title}: {axes_names[column]}")
            axis.set_ylabel(unit)
            axis.grid(True)

    axes[0, 0].legend(loc="best")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(f"Centroidal-state estimation with Koopman {estimator_name.upper()}")
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Koopman-based centroidal-state estimation"
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="kf",
        choices=["kf", "mhe"],
        help="State estimator using the learned DMDc model (default: kf)",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="trot",
        choices=["trot", "trot_fast", "jump", "bound"],
        help="Test trajectory (default: trot)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (default: <repository>/data)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Learned model path (default: <data_dir>/koopman_dmdc_model.pkl)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to estimate (default: all)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="MHE horizon length (default: 50)",
    )
    parser.add_argument(
        "--process_noise_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the process covariance (default: 1.0)",
    )
    parser.add_argument(
        "--measurement_noise_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the measurement covariance (default: 1.0)",
    )
    parser.add_argument(
        "--save_estimate",
        type=str,
        default=None,
        help="Optional path for saving the estimated state matrix",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path for saving the result figure",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    return parser.parse_args()


def main():
    # Step 1: Parse arguments and set up paths.
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.process_noise_scale <= 0.0 or args.measurement_noise_scale <= 0.0:
        raise ValueError("Noise scale multipliers must be positive")

    data_dir = Path(args.data_dir) if args.data_dir else REPOSITORY_ROOT / "data"
    model_path = (
        Path(args.model_path)
        if args.model_path
        else data_dir / "koopman_dmdc_model.pkl"
    )

    # Step 2: Load the common learned Koopman dynamics and test trajectory.
    koopman_model = load_koopman_model(model_path)
    reference, measurement, control_input = load_test_data(
        data_dir, args.motion, args.samples
    )

    # Step 3: Build and run either estimator with the same A, B, and C.
    Q, R = get_noise_covariances(
        koopman_model,
        args.process_noise_scale,
        args.measurement_noise_scale,
    )
    estimate = run_estimation(
        args.estimator,
        koopman_model,
        reference,
        measurement,
        control_input,
        args.horizon,
        Q,
        R,
    )

    # Step 4: Report, optionally save, and plot the result.
    print_estimation_errors(
        args.estimator, reference, measurement, estimate
    )
    if args.save_estimate is not None:
        estimate_path = Path(args.save_estimate)
        estimate_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(estimate_path, estimate, delimiter="\t")
        print(f"Saved estimates to {estimate_path}")

    time = np.arange(reference.shape[1]) * SAMPLING_PERIOD
    plot_estimation_results(
        time,
        args.estimator,
        reference,
        measurement,
        estimate,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
