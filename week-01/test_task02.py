# DL W01 Task02 Tests

import unittest
import numpy as np
from task02 import baseball_dataset, create_np_baseball_array, format_initial_data_output

class TestNpBaseballNumpyArray(unittest.TestCase):
    def test_when_creating_np_baseball_array_then_return_numpy_array(self):
        arr = create_np_baseball_array(baseball_dataset)
        self.assertIsInstance(arr, np.ndarray)

class TestInitialDataOutput(unittest.TestCase):
    def test_when_formatting_initial_data_output_then_contains_correct_strings(self):
        output = format_initial_data_output()
        self.assertIn("Number of rows and columns: ", output)
        self.assertIn("Summary statistics for height:", output)
        self.assertIn("Summary statistics for weight:", output)
        self.assertIn("Summary statistics for age:", output)
