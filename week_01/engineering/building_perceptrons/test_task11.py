import unittest
import numpy as np

from task11 import (
    ModelPowerOfTwo,
    create_dataset_power_of_two,
    sigmoid,
    calculate_loss,
    finite_diff_grad,
    train,
    predict_all,
)


class TestCreateDatasetPowerOfTwo(unittest.TestCase):

    def test_when_start_2_and_count_3_then_data_is_correct(self):
        data = create_dataset_power_of_two(2, 3)
        self.assertEqual(data, [[2, 4], [3, 9], [4, 16]])

    def test_when_count_is_zero_then_returns_empty_list(self):
        data = create_dataset_power_of_two(5, 0)
        self.assertEqual(data, [])


class TestSigmoid(unittest.TestCase):

    def test_when_input_zero_then_returns_half(self):
        self.assertTrue(np.isclose(sigmoid(0.0), 0.5, atol=1e-12))

    def test_when_large_positive_then_approximately_one(self):
        self.assertTrue(np.isclose(sigmoid(50.0), 1.0, atol=1e-12))

    def test_when_large_negative_then_approximately_zero(self):
        self.assertTrue(np.isclose(sigmoid(-50.0), 0.0, atol=1e-12))


class TestModelParams(unittest.TestCase):

    def test_when_get_params_as_vector_then_returns_copy(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        params = model.get_params_as_vector()
        params_before = params.copy()
        params[0] += 999.0
        self.assertTrue(
            np.array_equal(model.get_params_as_vector(), params_before))

    def test_when_load_vector_then_params_updated_correctly(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        new_params = np.arange(10, dtype=float)
        model.load_vector(new_params)
        self.assertTrue(
            np.array_equal(model.get_params_as_vector(), new_params))


class TestForward(unittest.TestCase):

    def test_when_all_params_zero_then_output_zero(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        model.load_vector(np.zeros(10, dtype=float))
        y = model.forward(3.0)
        self.assertTrue(np.isclose(y, 0.0, atol=1e-12))


class TestCalculateLoss(unittest.TestCase):

    def test_when_dataset_empty_then_loss_is_zero(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        params = model.get_params_as_vector()
        loss = calculate_loss(model, params, dataset=[])
        self.assertTrue(np.isclose(loss, 0.0, atol=1e-12))


class TestFiniteDiffGrad(unittest.TestCase):

    def test_when_called_then_returns_gradient_same_length_as_params(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        params = model.get_params_as_vector()
        dataset = create_dataset_power_of_two(0, 3)
        g = finite_diff_grad(model, params, dataset, eps=1e-3)
        self.assertEqual(g.shape, params.shape)
        self.assertTrue(np.all(np.isfinite(g)))


class TestTrain(unittest.TestCase):

    def test_when_called_then_returns_model_and_losses_with_expected_length(
            self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        dataset = create_dataset_power_of_two(2, 5)
        epochs = 10
        learning_rate = 0.01
        eps = 1e-3

        trained_model, losses = train(model, dataset, learning_rate, eps,
                                      epochs)

        self.assertIsInstance(trained_model, ModelPowerOfTwo)
        self.assertEqual(len(losses), epochs)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))


class TestPredictAll(unittest.TestCase):

    def test_when_called_then_returns_list_with_same_length_as_dataset(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        dataset = create_dataset_power_of_two(0, 7)
        preds = predict_all(model, dataset)
        self.assertEqual(len(preds), len(dataset))

    def test_when_forward_is_constant_then_predictions_match_expected(self):
        rng = np.random.default_rng(1)
        model = ModelPowerOfTwo(rng)
        model.load_vector(np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
                                   dtype=float))  # y == 5.5
        dataset = create_dataset_power_of_two(2, 5)
        preds = predict_all(model, dataset)
        for p in preds:
            self.assertTrue(np.isclose(p, 5.5, atol=1e-12))
