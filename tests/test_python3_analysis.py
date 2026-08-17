import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from neuron import h

import analyse_funcs
import analysis
import armGraphs
import hocinterface
import neuroplot


class Python3AnalysisCompatibilityTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_hoc_vector_conversion(self):
        h("objref py3_test_vector")
        h.py3_test_vector = h.Vector([1.0, 2.0, 3.0])

        np.testing.assert_array_equal(
            hocinterface.hv2narr("py3_test_vector"),
            np.array([1.0, 2.0, 3.0]),
        )

    def test_downsample_uses_integer_block_count(self):
        times = np.arange(8, dtype=float)
        values = np.arange(8, dtype=float)

        sampled_times, sampled_values = neuroplot.downsample(times, values, 250)

        np.testing.assert_array_equal(sampled_times, np.array([0.0, 4.0]))
        np.testing.assert_array_equal(sampled_values, np.array([1.5, 5.5]))

    def test_empty_spike_vector_is_supported(self):
        neuroplot.plot_spike_times(np.array([]))

    def test_spectrogram_browser_accepts_numpy_secondary_data(self):
        freqs = np.array([0.0, 10.0, 20.0])
        times = np.array([0.0, 0.1, 0.2])
        power = np.arange(9, dtype=float).reshape(3, 3) + 1.0
        browser = neuroplot.SpecgramBrowser(freqs, times, power, power.copy())

        browser._onclick(SimpleNamespace(xdata=100.0, ydata=10.0))
        self.assertEqual(browser.tindex, 1)

    def test_standalone_spectrogram_helpers_do_not_require_instance_state(self):
        freqs = np.array([0.0, 10.0, 20.0])
        times = np.array([0.0, 0.1, 0.2])
        power = np.arange(9, dtype=float).reshape(3, 3) + 1.0

        neuroplot.plot_specgram(freqs, times, power)
        neuroplot.plot_band_specgram(["low", "high"], times, power[:2])

    def test_errorfill_uses_current_matplotlib_color_cycle(self):
        axes = plt.subplots()[1]
        analyse_funcs.errorfill(
            np.array([0.0, 1.0]),
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            ax=axes,
        )
        self.assertEqual(len(axes.lines), 1)

    def test_arm_graphs_accept_fractional_time_bounds(self):
        sample_count = 10
        armGraphs.plotGraphs(
            np.zeros((9, sample_count)),
            np.zeros((2, sample_count)),
            np.ones((18, sample_count)),
            np.ones((18, sample_count)),
            np.ones((18, sample_count)),
            np.ones((18, sample_count)),
            0.01,
            0.05,
            10.0,
            False,
            False,
            "unused.png",
        )

    def test_arm_kinematics_round_trip(self):
        expected_angles = np.array([0.62, 1.53])
        position = analysis.angles2pos(*expected_angles, 0.275, 0.2275)
        actual_angles = analysis.pos2angles(*position, 0.275, 0.2275)

        np.testing.assert_allclose(actual_angles, expected_angles)

if __name__ == "__main__":
    unittest.main()
