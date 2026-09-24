import unittest

import numpy as np

from wake import (
    actual_wind_from,
    circular_error_deg,
    jensen_ct,
    jensen_wake_bias,
    propagation_unit_vector,
)


class WindConventionTests(unittest.TestCase):
    def test_actual_direction_wraps_nacelle_plus_relative_wind(self):
        np.testing.assert_allclose(
            actual_wind_from(np.array([350.0, 20.0]), np.array([20.0, -40.0])),
            np.array([10.0, 340.0]),
        )

    def test_northerly_wind_propagates_south(self):
        east, north = propagation_unit_vector(np.array([0.0]))
        np.testing.assert_allclose(east, np.array([0.0]), atol=1e-7)
        np.testing.assert_allclose(north, np.array([-1.0]), atol=1e-7)

    def test_circular_error_handles_wraparound(self):
        error = circular_error_deg(np.array([359.0]), np.array([1.0]))
        self.assertAlmostEqual(float(error[0]), -2.0)


class JensenTests(unittest.TestCase):
    def test_ct_curve_is_bounded_and_off_outside_operating_range(self):
        ct = jensen_ct(np.array([2.9, 3.0, 5.0, 8.0, 20.0, 25.1]))
        self.assertEqual(ct[0], 0.0)
        self.assertEqual(ct[-1], 0.0)
        self.assertTrue(np.all((ct[1:-1] >= 0.0) & (ct[1:-1] < 1.0)))
        self.assertAlmostEqual(ct[2], 0.789601, places=6)

    def test_directed_wake_edge_uses_destination_source_matrix_order(self):
        # Turbine 0 is north/upstream of turbine 1 under a northerly wind.
        xy = np.array([[0.0, 500.0], [0.0, 0.0]])
        bias = jensen_wake_bias(
            xy=xy,
            wind_from_deg=np.array([[0.0, 0.0]]),
            wind_speed=np.array([[8.0, 8.0]]),
            kw=0.075,
        )
        self.assertGreater(bias[0, 1, 0], 0.0)  # Query 1 <- Key 0.
        self.assertEqual(bias[0, 0, 1], 0.0)
        self.assertTrue(np.allclose(np.diagonal(bias[0]), 0.0))

    def test_crosswind_turbine_outside_expanded_radius_has_no_edge(self):
        xy = np.array([[0.0, 500.0], [500.0, 0.0]])
        bias = jensen_wake_bias(
            xy=xy,
            wind_from_deg=np.array([[0.0, 0.0]]),
            wind_speed=np.array([[8.0, 8.0]]),
            kw=0.075,
        )
        self.assertEqual(bias[0, 1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
