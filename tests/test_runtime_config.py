import os
import unittest
from unittest.mock import patch

from runtime_config import read_nonnegative_ms


class RuntimeConfigTests(unittest.TestCase):
    def test_unset_value_disables_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_nonnegative_ms("SNN_TEST_MS"), 0.0)

    def test_positive_value_is_loaded(self):
        with patch.dict(os.environ, {"SNN_TEST_MS": "2500.5"}, clear=True):
            self.assertEqual(read_nonnegative_ms("SNN_TEST_MS"), 2500.5)

    def test_negative_value_is_rejected(self):
        with patch.dict(os.environ, {"SNN_TEST_MS": "-1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SNN_TEST_MS"):
                read_nonnegative_ms("SNN_TEST_MS")

    def test_nonfinite_value_is_rejected(self):
        with patch.dict(os.environ, {"SNN_TEST_MS": "nan"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SNN_TEST_MS"):
                read_nonnegative_ms("SNN_TEST_MS")
