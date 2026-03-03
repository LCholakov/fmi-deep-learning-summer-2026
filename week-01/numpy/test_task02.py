# DL W01 Task02 Tests

import unittest
import numpy as np
from task02 import baseball_dataset, create_np_baseball_array, format_array_shape_output, format_mlb_player_data_summary_output

sample_baseball_dataset = [[74.0, 215.0, 34.69], [72.0, 210.0, 30.78], [72.0, 210.0, 35.43], [73.0, 188.0, 35.71], [69.0, 176.0, 29.39], [69.0, 209.0, 30.77], [71.0, 200.0, 35.07], [76.0, 231.0, 30.19], [71.0, 180.0, 27.05], [73.0, 188.0, 23.88], [73.0, 180.0, 26.96], [74.0, 185.0, 23.29]]

class TestNpBaseballNumpyArray(unittest.TestCase):
    def test_when_creating_np_baseball_array_then_return_numpy_array(self):
        arr = create_np_baseball_array(baseball_dataset)
        self.assertIsInstance(arr, np.ndarray)

class TestArrayShapeOutput(unittest.TestCase):
    def test_when_formatting_initial_data_output_then_contains_correct_strings(self):
        output = format_array_shape_output(sample_baseball_dataset)
        self.assertIn("Number of rows and columns: (", output)


class TestMLBPlayerDataSummaryOutput(unittest.TestCase):
    def test_when_formatting_mlb_player_data_summary_output_then_contains_correct_strings(self):
        output = format_mlb_player_data_summary_output(sample_baseball_dataset)
        self.assertIn("Summary statistics for height:", output)
        self.assertIn("Summary statistics for weight:", output)
        self.assertIn("Summary statistics for age:", output)
