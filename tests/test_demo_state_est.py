import unittest

import numpy as np

from demo.demo_state_est import plot_estimation_results


class TestStateEstimationDemoPlot(unittest.TestCase):
    def test_centroidal_groups_are_rows_and_xyz_are_columns(self):
        number_of_samples = 5
        time = np.arange(number_of_samples, dtype=float) * 0.001
        reference = np.zeros((9, number_of_samples))
        measurement = np.ones((9, number_of_samples))
        estimate = 0.5 * np.ones((9, number_of_samples))

        figure = plot_estimation_results(
            time,
            "kf",
            reference,
            measurement,
            estimate,
            show=False,
        )

        expected_titles = [
            "CoM position: x",
            "CoM position: y",
            "CoM position: z",
            "Linear momentum: x",
            "Linear momentum: y",
            "Linear momentum: z",
            "Angular momentum: x",
            "Angular momentum: y",
            "Angular momentum: z",
        ]
        self.assertEqual(
            [axis.get_title() for axis in figure.axes],
            expected_titles,
        )


if __name__ == "__main__":
    unittest.main()
