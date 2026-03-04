import unittest
import numpy as np
from task01 import get_my_rng, get_float_0to1, get_int_1to6

# ran with python -m unittest test_task01.py

class TestGetMyRNG(unittest.TestCase):
    def test_when_using_my_rng_then_seed_123_is_used(self):
        x = get_my_rng().random()
        self.assertEqual(x, np.random.default_rng(123).random())

    def test_when_using_my_rng_then_rng_is_consistent(self):
        x = get_my_rng().random()
        y = get_my_rng().random()
        self.assertEqual(x, y)


class TestGetFloat0to1(unittest.TestCase):
    def test_when_called_then_returns_float(self):
        rng = np.random.default_rng(1)
        x = get_float_0to1(rng)
        self.assertIsInstance(x, (float, np.floating))
        
    def test_when_called_then_value_in_0_inclusive_1_exclusive(self):
        rng = np.random.default_rng(1)
        x = get_float_0to1(rng)
        self.assertGreaterEqual(x, 0.0)
        self.assertLess(x, 1.0)



class TestGetInt1to6(unittest.TestCase):
    def test_when_called_then_returns_int(self):
        rng = np.random.default_rng(1)
        x = get_int_1to6(rng)
        self.assertIsInstance(x, (int, np.integer))

    def test_when_called_then_value_between_1_and_6_inclusive(self):
        x = get_int_1to6(np.random.default_rng(4))
        self.assertGreaterEqual(x, 1)
        self.assertLessEqual(x, 6)
