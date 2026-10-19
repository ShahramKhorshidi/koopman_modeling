import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Allow this file to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from src.modeling.koopman import DMDC


DEFAULT_SAMPLING_PERIOD = 0.001
GENERIC_STATE_FILE_PATTERN = "{index}_state.dat"
GENERIC_INPUT_FILE_PATTERN = "{index}_input.dat"
LEGACY_STATE_FILE_PATTERN = "{index}_cent_state.dat"
LEGACY_INPUT_FILE_PATTERN = "{index}_input_vector.dat"


def load_matrix(file_path, expected_rows=None):
    """Load a trajectory matrix and optionally validate its row count."""
    matrix = np.loadtxt(
        file_path,
        delimiter="\t",
        dtype=np.float64,
        ndmin=2,
    )
    if matrix.ndim != 2:
        raise ValueError(
            f"{file_path} must be a two-dimensional matrix; "
            f"received shape {matrix.shape}"
        )
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{file_path} must contain {expected_rows} rows; "
            f"received shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{file_path} contains non-finite values")
    return matrix


def select_numbered_file_pattern(
    data_dir,
    first_trajectory,
    requested_pattern,
    generic_pattern,
    legacy_pattern,
    data_name,
):
    """Select an explicit pattern or detect a supported filename convention."""
    patterns = (
        [requested_pattern]
        if requested_pattern is not None
        else [generic_pattern, legacy_pattern]
    )
    attempted_paths = []
    for pattern in patterns:
        if "{index" not in pattern:
            raise ValueError(
                f"The {data_name} file pattern must contain an "
                "'{index}' placeholder"
            )
        try:
            filename = pattern.format(index=first_trajectory)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                f"The {data_name} file pattern must contain a valid "
                "'{index}' placeholder"
            ) from error
        candidate = data_dir / filename
        attempted_paths.append(candidate)
        if candidate.is_file():
            return pattern

    attempted = ", ".join(str(path) for path in attempted_paths)
    raise FileNotFoundError(
        f"Could not find the first numbered {data_name} file. Tried: {attempted}"
    )


def combine_training_trajectories(
    data_dir,
    first_trajectory=0,
    last_trajectory=30,
    trim_start=500,
    state_dimension=None,
    input_dimension=None,
    state_file_pattern=None,
    input_file_pattern=None,
):
    """Build X, X_prime, and U from the numbered training trajectories."""
    if first_trajectory > last_trajectory:
        raise ValueError("first_trajectory must not exceed last_trajectory")

    state_file_pattern = select_numbered_file_pattern(
        data_dir,
        first_trajectory,
        state_file_pattern,
        GENERIC_STATE_FILE_PATTERN,
        LEGACY_STATE_FILE_PATTERN,
        "state",
    )
    input_file_pattern = select_numbered_file_pattern(
        data_dir,
        first_trajectory,
        input_file_pattern,
        GENERIC_INPUT_FILE_PATTERN,
        LEGACY_INPUT_FILE_PATTERN,
        "input",
    )
    print(f"Using state file pattern: {state_file_pattern}")
    print(f"Using input file pattern: {input_file_pattern}")

    state_snapshots = []
    next_state_snapshots = []
    input_snapshots = []

    for trajectory_number in range(first_trajectory, last_trajectory + 1):
        state_file = data_dir / state_file_pattern.format(index=trajectory_number)
        input_file = data_dir / input_file_pattern.format(index=trajectory_number)
        state = load_matrix(state_file, state_dimension)
        control_input = load_matrix(input_file, input_dimension)

        # When dimensions are not supplied, infer them from the first
        # trajectory and use them to validate every subsequent trajectory.
        if state_dimension is None:
            state_dimension = state.shape[0]
        if input_dimension is None:
            input_dimension = control_input.shape[0]

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


def load_combined_training_data(
    data_dir,
    state_dimension=None,
    input_dimension=None,
):
    """Load snapshot matrices, inferring dimensions when they are omitted."""
    X = load_matrix(data_dir / "training_data_state.dat", state_dimension)
    X_prime = load_matrix(
        data_dir / "training_data_state_prime.dat", X.shape[0]
    )
    U = load_matrix(data_dir / "training_data_input.dat", input_dimension)
    if X.shape != X_prime.shape or X.shape[1] != U.shape[1]:
        raise ValueError("The combined training matrices have incompatible shapes")
    return X, X_prime, U


def fit_koopman_model(X, X_prime, U, svd_rank=None):
    """Learn x_{k+1} = A x_k + B u_k with DMDc."""
    koopman_model = DMDC(X, X_prime=X_prime, U=U, svd_rank=svd_rank)
    singular_values = koopman_model.fit_model()

    residual = X_prime - koopman_model.A_matrix @ X - koopman_model.B_matrix @ U
    koopman_model.process_noise_covariance = np.atleast_2d(
        np.cov(residual, bias=True)
    )
    koopman_model.training_residual_std = np.std(residual, axis=1)
    koopman_model.training_state_minimum = np.min(X, axis=1)
    koopman_model.training_state_maximum = np.max(X, axis=1)
    return koopman_model, singular_values


def load_test_data(
    data_dir,
    trajectory,
    number_of_samples=None,
    state_dimension=None,
    input_dimension=None,
    state_file=None,
    input_file=None,
):
    """Load a clean test trajectory for open-loop model validation."""
    if state_file is not None:
        state_path = Path(state_file)
    else:
        state_candidates = [
            data_dir / f"{trajectory}_test_state.dat",
            data_dir / f"{trajectory}_test_cent_state.dat",
        ]
        state_path = next(
            (path for path in state_candidates if path.is_file()),
            state_candidates[0],
        )

    if input_file is not None:
        input_path = Path(input_file)
    else:
        input_candidates = [
            data_dir / f"{trajectory}_test_input.dat",
            data_dir / f"{trajectory}_test_input_vector.dat",
        ]
        input_path = next(
            (path for path in input_candidates if path.is_file()),
            input_candidates[0],
        )
    state = load_matrix(state_path, state_dimension)
    control_input = load_matrix(input_path, input_dimension)
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
    """Report overall and per-state open-loop prediction errors."""
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction must have the same shape")

    error = reference - prediction
    overall_rmse = np.sqrt(np.mean(error ** 2))
    per_state_rmse = np.sqrt(np.mean(error ** 2, axis=1))
    print(f"Overall open-loop RMSE: {overall_rmse:.6e}")
    print("Per-state open-loop RMSE:")
    for state_index, rmse in enumerate(per_state_rmse):
        print(f"  x[{state_index}]: {rmse:.6e}")


def _state_plot_metadata(number_of_states):
    """Return descriptive labels for the original centroidal state."""
    if number_of_states == 9:
        groups = [
            ("CoM position", "m"),
            ("Linear momentum", "kg m/s"),
            ("Angular momentum", "kg m^2/s"),
        ]
        axes_names = ["x", "y", "z"]
        return [
            (f"{group}: {axis}", unit)
            for group, unit in groups
            for axis in axes_names
        ]
    return [(f"State x[{index}]", "Value") for index in range(number_of_states)]


def plot_model_results(
    reference,
    prediction,
    save_path=None,
    show=True,
    sampling_period=DEFAULT_SAMPLING_PERIOD,
):
    """Plot the open-loop DMDc prediction against the test trajectory."""
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction must have the same shape")
    if sampling_period <= 0.0:
        raise ValueError("sampling_period must be positive")

    number_of_states = reference.shape[0]
    time = np.arange(reference.shape[1]) * sampling_period
    number_of_columns = min(3, number_of_states)
    number_of_rows = int(np.ceil(number_of_states / number_of_columns))
    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(4.8 * number_of_columns, 2.8 * number_of_rows),
        sharex=True,
        squeeze=False,
    )
    labels = _state_plot_metadata(number_of_states)

    for state_index, axis in enumerate(axes.flat[:number_of_states]):
        title, unit = labels[state_index]
        axis.plot(time, reference[state_index], "b", label="Reference")
        axis.plot(
            time,
            prediction[state_index],
            "r--",
            linewidth=0.9,
            label="DMDc prediction",
        )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xlabel("Time (s)")
        axis.grid(True)

    for axis in axes.flat[number_of_states:]:
        axis.set_visible(False)
    axes.flat[0].legend(loc="best")
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
        description="Learn and validate a DMDc Koopman model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (default: <repository>/data)",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=None,
        help=(
            "Expected number of state rows; inferred from the training data "
            "when omitted"
        ),
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=None,
        help=(
            "Expected number of input rows; inferred from the training data "
            "when omitted"
        ),
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
        "--state_file_pattern",
        type=str,
        default=None,
        help=(
            "Numbered state filename pattern containing '{index}'; "
            "default: auto-detect generic or legacy names"
        ),
    )
    parser.add_argument(
        "--input_file_pattern",
        type=str,
        default=None,
        help=(
            "Numbered input filename pattern containing '{index}'; "
            "default: auto-detect generic or legacy names"
        ),
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
        "--trajectory",
        type=str,
        default=None,
        help=(
            "Validation-file prefix used with --data_dir; open-loop "
            "validation is skipped when no validation source is supplied"
        ),
    )
    parser.add_argument(
        "--validation_state_file",
        type=str,
        default=None,
        help="Optional explicit state file for open-loop validation",
    )
    parser.add_argument(
        "--validation_input_file",
        type=str,
        default=None,
        help="Optional explicit input file for open-loop validation",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Test samples used for validation (default: all)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_SAMPLING_PERIOD,
        help="Sampling period used for the time axis (default: 0.001 s)",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Fit and save the model without requiring validation files",
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
    if args.state_dim is not None and args.state_dim < 1:
        raise ValueError("--state_dim must be positive")
    if args.input_dim is not None and args.input_dim < 1:
        raise ValueError("--input_dim must be positive")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    data_dir = Path(args.data_dir) if args.data_dir else REPOSITORY_ROOT / "data"
    model_path = (
        Path(args.model_path)
        if args.model_path
        else data_dir / "koopman_dmdc_model.pkl"
    )

    # Step 2: Load the snapshot pairs used to learn the Koopman model.
    if args.training_source == "combined":
        X, X_prime, U = load_combined_training_data(
            data_dir,
            args.state_dim,
            args.input_dim,
        )
        training_description = "the supplied combined snapshot matrices"
    else:
        X, X_prime, U = combine_training_trajectories(
            data_dir,
            args.first_trajectory,
            args.last_trajectory,
            args.trim_start,
            args.state_dim,
            args.input_dim,
            args.state_file_pattern,
            args.input_file_pattern,
        )
        training_description = (
            f"trajectories {args.first_trajectory} through "
            f"{args.last_trajectory}"
        )
    print(f"Learning from {training_description}: {X.shape[1]} transitions")
    print(f"Model dimensions: {X.shape[0]} states, {U.shape[0]} inputs")

    # Step 3: Fit and save the common DMDc dynamics model.
    koopman_model, singular_values = fit_koopman_model(X, X_prime, U, args.svd_rank)
    if args.training_source == "trajectories":
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

    # Step 4: Validate the learned model on a separate test trajectory.
    validation_files_supplied = (
        args.validation_state_file is not None
        or args.validation_input_file is not None
    )
    if args.skip_validation or (
        args.trajectory is None and not validation_files_supplied
    ):
        print(
            "Open-loop validation skipped; supply --trajectory or both "
            "explicit validation files to enable it"
        )
        return
    if args.trajectory is None and not all(
        (args.validation_state_file, args.validation_input_file)
    ):
        raise ValueError(
            "Without --trajectory, both --validation_state_file and "
            "--validation_input_file are required"
        )

    reference, control_input = load_test_data(
        data_dir,
        args.trajectory,
        args.samples,
        X.shape[0],
        U.shape[0],
        args.validation_state_file,
        args.validation_input_file,
    )
    prediction = koopman_model.simulate_model(reference[:, 0], control_input)
    print_model_errors(reference, prediction)
    plot_model_results(
        reference,
        prediction,
        save_path=args.save,
        show=not args.no_show,
        sampling_period=args.dt,
    )


if __name__ == "__main__":
    main()
