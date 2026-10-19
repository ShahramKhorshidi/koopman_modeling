import unittest
import numpy as np
from src.estimation.kalman_filter import Kalman_Filter
from src.estimation.moving_horizon import MovingHorizonEstimator
from src.modeling.koopman import DMDC, EDMDC, EDMDCKIC
from src.modeling.observables import Observables


class TestKoopmanModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.A = np.array([[0.98, 0.05], [-0.02, 0.95]])
        self.B = np.array([[0.10], [0.04]])
        self.X = rng.normal(size=(2, 200))
        self.U = rng.normal(size=(1, 200))
        self.X_prime = self.A @ self.X + self.B @ self.U

    def test_dmdc_recovers_linear_dynamics(self):
        model = DMDC(self.X, self.X_prime, self.U)
        model.fit_model()

        np.testing.assert_allclose(model.A_matrix, self.A, atol=1e-12)
        np.testing.assert_allclose(model.B_matrix, self.B, atol=1e-12)

    def test_dmdc_simulation_uses_previous_input(self):
        model = DMDC(self.X, self.X_prime, self.U)
        model.fit_model()
        inputs = np.array([[0.4, -0.2, 0.7, 0.0]])
        prediction = model.simulate_model(np.array([0.2, -0.1]), inputs)

        for k in range(1, prediction.shape[1]):
            expected = self.A @ prediction[:, k-1] + self.B @ inputs[:, k-1]
            np.testing.assert_allclose(prediction[:, k], expected, atol=1e-12)

    def test_dmdc_optimization_solves_noisy_least_squares(self):
        rng = np.random.default_rng(11)
        noisy_next_state = self.X_prime + 1e-4 * rng.normal(size=self.X_prime.shape)
        model = DMDC(self.X, noisy_next_state, self.U)
        model.optimize_model(regularization=1e-10)

        np.testing.assert_allclose(model.A_matrix, self.A, atol=3e-5)
        np.testing.assert_allclose(model.B_matrix, self.B, atol=3e-5)

    def test_edmdc_uses_training_scaling_for_new_inputs(self):
        model = EDMDC(
            self.X,
            self.X_prime,
            self.U,
            list_of_obs=[Observables.identity()],
        )
        model.fit_model()
        constant_input = np.full((1, 5), 2.5)
        prediction = model.simulate_model(self.X[:, 0], constant_input)

        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_model_matrices_are_unavailable_before_fitting(self):
        model = DMDC(self.X, self.X_prime, self.U)
        with self.assertRaises(RuntimeError):
            model.get_linear_sys_matrices()

    def test_edmdc_with_input_observables_runs_end_to_end(self):
        model = EDMDCKIC(
            self.X,
            self.X_prime,
            self.U,
            list_of_state_obs=[Observables.identity()],
            list_of_input_obs=[Observables.identity()],
        )
        model.fit_model(lambda_val=1e-8)
        prediction = model.simulate_model(
            self.X[:, 0], self.U[:, :4], num_steps=5
        )

        self.assertEqual(prediction.shape, (2, 5))
        self.assertTrue(np.all(np.isfinite(prediction)))


class TestStateEstimation(unittest.TestCase):
    def test_dmdc_kalman_wrapper_uses_current_measurement(self):
        states = np.array([[0.0, 1.0, 3.0, 6.0]])
        inputs = np.array([[1.0, 2.0, 3.0]])
        model = DMDC(states[:, :-1], states[:, 1:], inputs)
        model.fit_model()
        estimator = Kalman_Filter(*model.get_linear_sys_matrices())
        estimator.set_process_noise_cov(np.array([[1e-9]]))
        estimator.set_measurement_noise_cov(np.array([[1e-9]]))

        estimate = model.kf_estimation(estimator, inputs, states)
        np.testing.assert_allclose(estimate, states, atol=1e-6)

    def test_kalman_filter_covariance_stays_symmetric(self):
        estimator = Kalman_Filter(
            np.array([[1.0, 0.1], [0.0, 1.0]]),
            np.array([[0.0], [0.1]]),
            np.array([[1.0, 0.0]]),
        )
        estimator.set_process_noise_cov(1e-4 * np.eye(2))
        estimator.set_measurement_noise_cov(np.array([[1e-2]]))

        for _ in range(20):
            estimator.update_filter(np.array([0.2]), np.array([0.0]))

        np.testing.assert_allclose(
            estimator.Sigma_post, estimator.Sigma_post.T, atol=1e-14
        )
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(estimator.Sigma_post)), -1e-12)

    def test_kalman_filter_handles_singular_innovation_covariance(self):
        estimator = Kalman_Filter(
            np.array([[1.0]]), np.array([[0.0]]), np.array([[0.0]])
        )
        estimator.set_initial_covariance(np.array([[0.0]]))
        estimator.set_process_noise_cov(np.array([[0.0]]))
        estimator.set_measurement_noise_cov(np.array([[0.0]]))

        estimator.update_filter(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(estimator.get_filter_output(), np.array([0.0]))

    def test_mhe_measurement_is_aligned_with_next_state(self):
        estimator = MovingHorizonEstimator(
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            N=3,
            state_bounds=(np.array([-10.0]), np.array([10.0])),
        )

        for control, measurement in zip([1.0, 2.0, 3.0], [1.0, 3.0, 6.0]):
            estimator.update_measurements(
                np.array([control]), np.array([measurement])
            )

        estimator.setup_optimization_problem(np.array([0.0]))
        state_estimate = estimator.estimate()
        np.testing.assert_allclose(
            state_estimate.reshape(-1), np.array([0.0, 1.0, 3.0, 6.0]), atol=1e-5
        )

    def test_mhe_handles_precision_weights_with_a_large_range(self):
        estimator = MovingHorizonEstimator(
            np.eye(2), np.zeros((2, 1)), np.eye(2), N=1
        )
        mixed_scale_weight = np.diag([1e12, 1.0])
        estimator.set_arrival_cost_weight(mixed_scale_weight)
        estimator.set_process_noise_weight(mixed_scale_weight)
        estimator.set_measurement_noise_weight(mixed_scale_weight)

        estimator.update_measurements(np.zeros(1), np.ones(2))
        estimator.setup_optimization_problem(np.zeros(2))
        first_problem = estimator.problem
        state_estimate = estimator.estimate()

        # For equal arrival, process, and measurement weights, the analytical
        # last-state estimate is 2/3 in both independently scaled dimensions.
        np.testing.assert_allclose(state_estimate[:, -1], 2.0 / 3.0, atol=1e-5)

        estimator.update_measurements(np.zeros(1), np.ones(2))
        estimator.setup_optimization_problem(np.zeros(2))
        self.assertIs(estimator.problem, first_problem)

    def test_dmdc_mhe_wrapper_shifts_arrival_state(self):
        inputs = np.array([[1.0, 2.0, 3.0, -1.0, 0.5]])
        states = np.zeros((1, inputs.shape[1] + 1))
        for k in range(inputs.shape[1]):
            states[:, k+1] = states[:, k] + inputs[:, k]

        model = DMDC(states[:, :-1], states[:, 1:], inputs)
        model.fit_model()
        estimator = MovingHorizonEstimator(
            *model.get_linear_sys_matrices(),
            N=3,
            state_bounds=(np.array([-10.0]), np.array([10.0])),
        )

        estimate = model.mhe_estimation(estimator, states[:, 0], inputs, states)
        np.testing.assert_allclose(estimate, states, atol=1e-5)


class TestObservables(unittest.TestCase):
    def test_full_polynomial_preserves_sample_axis(self):
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        lifted = Observables.full_polynomial(degree=2)(values)

        self.assertEqual(lifted.shape[1], values.shape[1])
        np.testing.assert_allclose(lifted[:2], values)

    def test_time_delay_uses_past_samples(self):
        values = np.array([[1.0, 2.0, 3.0, 4.0]])
        lifted = Observables.time_delay_observables(1, 3)(values)
        expected = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0, 2.0],
        ])

        np.testing.assert_allclose(lifted, expected)

    def test_tensor_observable_contains_each_order_once(self):
        values = np.array([[2.0], [3.0]])
        lifted = Observables.tensor_based(values, order=3)(values)

        self.assertEqual(lifted.shape, (14, 1))
        np.testing.assert_allclose(lifted[:2], values)


if __name__ == "__main__":
    unittest.main()
