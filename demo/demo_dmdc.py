import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Allow this file to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from src.modeling.koopman import DMDC


STATE_DIMENSION = 9
INPUT_DIMENSION = 24
SAMPLING_PERIOD = 0.001


def load_matrix(file_path, expected_rows):
    """Load a trajectory matrix and check its orientation."""
    matrix = np.loadtxt(file_path, delimiter="\t", dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{file_path} must contain {expected_rows} rows; "
            f"received shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{file_path} contains non-finite values")
    return matrix


def combine_training_trajectories(
    data_dir,
    first_trajectory=0,
    last_trajectory=30,
    trim_start=500,
):
    """Build X, X_prime, and U from the numbered training trajectories."""
    state_snapshots = []
    next_state_snapshots = []
    input_snapshots = []

    for trajectory_number in range(first_trajectory, last_trajectory + 1):
        state_file = data_dir / f"{trajectory_number}_cent_state.dat"
        input_file = data_dir / f"{trajectory_number}_input_vector.dat"
        state = load_matrix(state_file, STATE_DIMENSION)
        control_input = load_matrix(input_file, INPUT_DIMENSION)

        if state.shape[1] != control_input.shape[1]:
            raise ValueError(
                f"Trajectory {trajectory_number} has different state and input lengths"
            )
        if trim_start < 0 or trim_start >= state.shape[1] - 1:
            raise ValueError(
                f"trim_start must be between 0 and {state.shape[1] - 2}"
            )

        # x_k and u_k predict x_{k+1}. The first 500 samples are the
        # initialization transient in the original training-data preparation.
        state_snapshots.append(state[:, trim_start:-1])
        next_state_snapshots.append(state[:, trim_start + 1:])
        input_snapshots.append(control_input[:, trim_start:-1])

    X = np.hstack(state_snapshots)
    X_prime = np.hstack(next_state_snapshots)
    U = np.hstack(input_snapshots)
    return X, X_prime, U


def load_combined_training_data(data_dir):
    """Load the previously combined snapshot matrices."""
    X = load_matrix(data_dir / "training_data_state.dat", STATE_DIMENSION)
    X_prime = load_matrix(
        data_dir / "training_data_state_prime.dat", STATE_DIMENSION
    )
    U = load_matrix(data_dir / "training_data_input.dat", INPUT_DIMENSION)
    if X.shape != X_prime.shape or X.shape[1] != U.shape[1]:
        raise ValueError("The combined training matrices have incompatible shapes")
    return X, X_prime, U


def fit_koopman_model(X, X_prime, U, svd_rank=None):
    """Learn x_{k+1} = A x_k + B u_k with DMDc."""
    koopman_model = DMDC(X, X_prime=X_prime, U=U, svd_rank=svd_rank)
    singular_values = koopman_model.fit_model()

    residual = X_prime - koopman_model.A_matrix @ X - koopman_model.B_matrix @ U
    koopman_model.process_noise_covariance = np.cov(residual, bias=True)
    koopman_model.training_residual_std = np.std(residual, axis=1)
    koopman_model.training_state_minimum = np.min(X, axis=1)
    koopman_model.training_state_maximum = np.max(X, axis=1)
    return koopman_model, singular_values


def load_test_data(data_dir, motion, number_of_samples=None):
    """Load a clean test trajectory for open-loop model validation."""
    state = load_matrix(
        data_dir / f"{motion}_test_cent_state.dat", STATE_DIMENSION
    )
    control_input = load_matrix(
        data_dir / f"{motion}_test_input_vector.dat", INPUT_DIMENSION
    )
    available_samples = min(state.shape[1], control_input.shape[1])
    if number_of_samples is None:
        number_of_samples = available_samples
    if number_of_samples < 2 or number_of_samples > available_samples:
        raise ValueError(
            f"--samples must be between 2 and {available_samples}"
        )
    return (
        state[:, :number_of_samples],
        control_input[:, :number_of_samples],
    )


def print_model_errors(reference, prediction):
    """Report RMSE for the three centroidal-state blocks."""
    state_blocks = {
        "CoM position": slice(0, 3),
        "Linear momentum": slice(3, 6),
        "Angular momentum": slice(6, 9),
    }
    for name, rows in state_blocks.items():
        rmse = np.sqrt(np.mean((reference[rows] - prediction[rows]) ** 2))
        print(f"{name} open-loop RMSE: {rmse:.6e}")


def plot_model_results(reference, prediction, save_path=None, show=True):
    """Plot the open-loop DMDc prediction against the test trajectory."""
    time = np.arange(reference.shape[1]) * SAMPLING_PERIOD
    groups = [
        ("CoM position", "m", slice(0, 3)),
        ("Linear momentum", "kg m/s", slice(3, 6)),
        ("Angular momentum", "kg m^2/s", slice(6, 9)),
    ]
    axes_names = ["x", "y", "z"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)

    for column, (title, unit, rows) in enumerate(groups):
        for row, state_index in enumerate(range(rows.start, rows.stop)):
            axis = axes[row, column]
            axis.plot(time, reference[state_index], "b", label="Reference")
            axis.plot(
                time,
                prediction[state_index],
                "r--",
                linewidth=0.9,
                label="DMDc prediction",
            )
            axis.set_title(f"{title}: {axes_names[row]}")
            axis.set_ylabel(unit)
            axis.grid(True)

    axes[0, 0].legend(loc="best")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle("Open-loop validation of the learned Koopman model")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn and validate a centroidal DMDc Koopman model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (default: <repository>/data)",
    )
    parser.add_argument(
        "--training_source",
        type=str,
        default="combined",
        choices=["combined", "trajectories"],
        help=(
            "Use the existing combined matrices or rebuild them from the "
            "numbered trajectories (default: combined)"
        ),
    )
    parser.add_argument(
        "--first_trajectory",
        type=int,
        default=0,
        help="First numbered training trajectory (default: 0)",
    )
    parser.add_argument(
        "--last_trajectory",
        type=int,
        default=30,
        help="Last numbered training trajectory, inclusive (default: 30)",
    )
    parser.add_argument(
        "--trim_start",
        type=int,
        default=500,
        help="Initial samples discarded from every trajectory (default: 500)",
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=None,
        help="Optional rank used in the DMDc SVD (default: full rank)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Saved model path (default: <data_dir>/koopman_dmdc_model.pkl)",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="trot",
        choices=["trot", "trot_fast", "jump", "bound"],
        help="Test motion used for open-loop validation (default: trot)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Test samples used for validation (default: all)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path for saving the validation figure",
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
    data_dir = Path(args.data_dir) if args.data_dir else REPOSITORY_ROOT / "data"
    model_path = (
        Path(args.model_path)
        if args.model_path
        else data_dir / "koopman_dmdc_model.pkl"
    )

    # Step 2: Load the snapshot pairs used to learn the Koopman model.
    if args.training_source == "combined":
        X, X_prime, U = load_combined_training_data(data_dir)
        training_description = "the existing combined trajectories 0 through 30"
    else:
        X, X_prime, U = combine_training_trajectories(
            data_dir,
            args.first_trajectory,
            args.last_trajectory,
            args.trim_start,
        )
        training_description = (
            f"trajectories {args.first_trajectory} through "
            f"{args.last_trajectory}"
        )
    print(f"Learning from {training_description}: {X.shape[1]} transitions")

    # Step 3: Fit and save the common DMDc dynamics model.
    koopman_model, singular_values = fit_koopman_model(X, X_prime, U, args.svd_rank)
    koopman_model.training_trajectory_range = (
        args.first_trajectory,
        args.last_trajectory,
    )
    koopman_model.training_trim_start = args.trim_start

    # The learned matrices and summary statistics are sufficient for online
    # estimation. Avoid duplicating the complete training set in the model file.
    koopman_model.X = None
    koopman_model.X_prime = None
    koopman_model.U = None
    model_path.parent.mkdir(parents=True, exist_ok=True)
    koopman_model.save_model(model_path)
    print(f"Saved the learned model to {model_path}")
    print(f"Smallest retained singular value: {singular_values[-1]:.6e}")
    print("Training residual standard deviation:")
    print(koopman_model.training_residual_std)

    # Step 4: Validate the learned model on a separate test motion.
    reference, control_input = load_test_data(
        data_dir, args.motion, args.samples
    )
    prediction = koopman_model.simulate_model(reference[:, 0], control_input)
    print_model_errors(reference, prediction)
    plot_model_results(
        reference,
        prediction,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
