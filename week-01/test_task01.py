import unittest
import numpy as np
from task01 import baseball, height_in, create_baseball_np_array, format_sunday_output, convert_np_array_in_to_cm

class TestBaseballNumpyArray(unittest.TestCase):
    def test_when_creating_baseball_np_array_then_return_numpy_array(self):
        arr = create_baseball_np_array(baseball)
        self.assertIsInstance(arr, np.ndarray)

class TestSundayOutput(unittest.TestCase):
    def test_when_formatting_sunday_output_then_contains_correct_strings(self):
        output = format_sunday_output()
        self.assertIn("Sunday analysis:", output)
        self.assertIn("Baseball array: [", output)
        self.assertIn("Type of baseball array: <class 'numpy.ndarray'>", output)

class TestInCmConversion(unittest.TestCase):
    def test_when_convert_np_array_in_to_cm_then_result_is_accurate(self):
        arr = create_baseball_np_array(height_in)
        arr_cm = convert_np_array_in_to_cm(arr)
        self.assertEqual(arr * 0.0254, arr_cm)


class TestMondayOutput(unittest.TestCase):
    def test_when_formatting_monday_output_then_contains_correct_strings(self):
        output = format_sunday_output()
        self.assertIn("Monday analysis:", output)
        self.assertIn("np_height_in=array([", output)
        self.assertIn("np_height_metres=array(", output)

