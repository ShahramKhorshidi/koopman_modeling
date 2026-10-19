import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from demo.demo_dmdc import (
    combine_training_trajectories,
    load_combined_training_data,
    load_test_data,
    main,
    plot_model_results,
)
from src.modeling.koopman import DMDC


class TestDMDCTrainingDemo(unittest.TestCase):
    @staticmethod
    def save_matrix(path, matrix):
        np.savetxt(path, matrix, delimiter="\t")

    def test_combined_data_dimensions_are_inferred(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            state = np.arange(12, dtype=float).reshape(2, 6)
            next_state = state + 1.0
            control_input = np.arange(18, dtype=float).reshape(3, 6)
            self.save_matrix(data_dir / "training_data_state.dat", state)
            self.save_matrix(
                data_dir / "training_data_state_prime.dat", next_state
            )
            self.save_matrix(data_dir / "training_data_input.dat", control_input)

            X, X_prime, U = load_combined_training_data(data_dir)

            self.assertEqual(X.shape, (2, 6))
            self.assertEqual(X_prime.shape, (2, 6))
            self.assertEqual(U.shape, (3, 6))

    def test_optional_dimensions_validate_user_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self.save_matrix(
                data_dir / "training_data_state.dat", np.zeros((2, 5))
            )
            self.save_matrix(
                data_dir / "training_data_state_prime.dat", np.zeros((2, 5))
            )
            self.save_matrix(
                data_dir / "training_data_input.dat", np.zeros((1, 5))
            )

            with self.assertRaisesRegex(ValueError, "must contain 3 rows"):
                load_combined_training_data(data_dir, state_dimension=3)

    def test_validation_files_use_generic_trajectory_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            state = np.arange(20, dtype=float).reshape(4, 5)
            control_input = np.arange(10, dtype=float).reshape(2, 5)
            self.save_matrix(
                data_dir / "trajectory_01_test_state.dat", state
            )
            self.save_matrix(
                data_dir / "trajectory_01_test_input.dat", control_input
            )

            loaded_state, loaded_input = load_test_data(
                data_dir,
                "trajectory_01",
                state_dimension=4,
                input_dimension=2,
            )

            np.testing.assert_allclose(loaded_state, state)
            np.testing.assert_allclose(loaded_input, control_input)

    def test_numbered_trajectories_infer_and_enforce_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for trajectory_number in range(2):
                self.save_matrix(
                    data_dir / f"{trajectory_number}_state.dat",
                    np.arange(12, dtype=float).reshape(2, 6),
                )
                self.save_matrix(
                    data_dir / f"{trajectory_number}_input.dat",
                    np.arange(6, dtype=float).reshape(1, 6),
                )

            X, X_prime, U = combine_training_trajectories(
                data_dir,
                first_trajectory=0,
                last_trajectory=1,
                trim_start=1,
            )

            self.assertEqual(X.shape, (2, 8))
            self.assertEqual(X_prime.shape, (2, 8))
            self.assertEqual(U.shape, (1, 8))

            self.save_matrix(
                data_dir / "1_input.dat", np.zeros((2, 6))
            )
            with self.assertRaisesRegex(ValueError, "must contain 1 rows"):
                combine_training_trajectories(
                    data_dir,
                    first_trajectory=0,
                    last_trajectory=1,
                    trim_start=1,
                )

    def test_legacy_numbered_trajectory_names_are_auto_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            self.save_matrix(
                data_dir / "0_cent_state.dat", np.zeros((3, 6))
            )
            self.save_matrix(
                data_dir / "0_input_vector.dat", np.zeros((2, 6))
            )

            X, X_prime, U = combine_training_trajectories(
                data_dir,
                first_trajectory=0,
                last_trajectory=0,
                trim_start=1,
            )

            self.assertEqual(X.shape, (3, 4))
            self.assertEqual(X_prime.shape, (3, 4))
            self.assertEqual(U.shape, (2, 4))

    def test_one_dimensional_input_and_training_only_cli(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            rng = np.random.default_rng(12)
            state = rng.normal(size=(2, 20))
            control_input = rng.normal(size=(1, 20))
            next_state = (
                np.array([[0.9, 0.1], [-0.2, 0.8]]) @ state
                + np.array([[0.3], [0.1]]) @ control_input
            )
            self.save_matrix(data_dir / "training_data_state.dat", state)
            self.save_matrix(
                data_dir / "training_data_state_prime.dat", next_state
            )
            self.save_matrix(data_dir / "training_data_input.dat", control_input)
            model_path = data_dir / "custom_model.pkl"

            arguments = [
                "demo_dmdc.py",
                "--data_dir",
                str(data_dir),
                "--model_path",
                str(model_path),
            ]
            with mock.patch.object(sys, "argv", arguments):
                main()

            model = DMDC.load_model(model_path)
            self.assertEqual(model.A_matrix.shape, (2, 2))
            self.assertEqual(model.B_matrix.shape, (2, 1))
            self.assertEqual(model.process_noise_covariance.shape, (2, 2))

    def test_validation_plot_adapts_to_arbitrary_state_dimension(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "validation.png"
            reference = np.arange(35, dtype=float).reshape(5, 7)
            prediction = reference + 0.1

            plot_model_results(
                reference,
                prediction,
                save_path=output_path,
                show=False,
                sampling_period=0.02,
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
