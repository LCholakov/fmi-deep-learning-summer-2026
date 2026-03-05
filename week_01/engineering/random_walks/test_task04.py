import unittest
import numpy as np
from task04 import get_my_rng, get_int_1to6, get_next_move

# absolutely copiet my task02 tests...

class TestGetMyRNG(unittest.TestCase):
    def test_when_using_my_rng_then_seed_123_is_used(self):
        x = get_my_rng().random()
        self.assertEqual(x, np.random.default_rng(123).random())

    def test_when_using_my_rng_then_rng_is_consistent(self):
        x = get_my_rng().random()
        y = get_my_rng().random()
        self.assertEqual(x, y)

class TestGetInt1to6(unittest.TestCase):
    def test_when_called_then_returns_int(self):
        rng = np.random.default_rng(1)
        x = get_int_1to6(rng)
        self.assertIsInstance(x, (int, np.integer))

    def test_when_called_then_value_between_1_and_6_inclusive(self):
        x = get_int_1to6(np.random.default_rng(4))
        self.assertGreaterEqual(x, 1)
        self.assertLessEqual(x, 6)

class TestGetNextMove(unittest.TestCase):
    def test_when_passing_1_or_2_then_return_minus_1(self):
        rng = np.random.default_rng(1)
        x = get_next_move(1, rng)
        y = get_next_move(2, rng)
        self.assertEqual(x, -1)
        self.assertEqual(y, -1)

    def test_when_passing_3_or_4_or_5_then_return_plus_1(self):
        rng = np.random.default_rng(1)
        x = get_next_move(3, rng)
        y = get_next_move(4, rng)
        z = get_next_move(5, rng)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)
        self.assertEqual(z, 1)

    def test_when_passing_6_then_return_value_between_1_and_6(self):
        rng = np.random.default_rng(1)
        x = get_next_move(6, rng)
        self.assertGreaterEqual(x, 1)
        self.assertLessEqual(x, 6)

    def test_when_passing_6_then_return_random_value(self):
        rng = np.random.default_rng(1)
        
        v = get_next_move(6, rng)
        w = get_next_move(6, rng)
        x = get_next_move(6, rng)
        y = get_next_move(6, rng)
        z = get_next_move(6, rng)
        self.assertFalse(v == w == x == y == z)
        