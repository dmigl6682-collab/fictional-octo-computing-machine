"""
Unit Test Suite for On-Device AI Edge Suite (Python / NumPy)
Run: pytest test_models.py
"""

import unittest
import numpy as np
from main import (
    FeatureStandardizer,
    OnDeviceKNNClassifier,
    OnDeviceNeuralNet,
    OnDeviceAnomalyDetector,
    OnDeviceLinearRegressor
)

class TestOnDeviceAIModels(unittest.TestCase):

    def setUp(self):
        self.sample_vec = np.array([0.85, 0.12, 0.90, 0.01], dtype=np.float32)
        self.anomaly_vec = np.array([0.05, 0.99, 0.02, 0.95], dtype=np.float32)

    def test_feature_standardizer(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        scaler = FeatureStandardizer(X)
        norm = scaler.transform(X)
        self.assertAlmostEqual(float(np.mean(norm)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(norm)), 1.0, places=4)

    def test_knn_classifier(self):
        knn = OnDeviceKNNClassifier(k=3)
        res = knn.predict(self.sample_vec)
        self.assertIn("prediction", res)
        self.assertEqual(res["class_id"], 0)
        self.assertGreater(res["confidence"], 60.0)
        self.assertLess(res["latency_ms"], 15.0)  # Must be fast for mobile

    def test_neural_net_forward_pass(self):
        mlp = OnDeviceNeuralNet()
        res = mlp.forward(self.sample_vec)
        self.assertIn("prediction", res)
        self.assertEqual(len(res["probabilities"]), 4)
        # Probabilities sum to 100%
        self.assertAlmostEqual(sum(res["probabilities"]), 100.0, places=1)
        self.assertLess(res["latency_ms"], 10.0)

    def test_anomaly_detector(self):
        knn = OnDeviceKNNClassifier(k=3)
        detector = OnDeviceAnomalyDetector(knn, threshold_sigma=2.2)
        normal_res = detector.detect(self.sample_vec)
        self.assertFalse(normal_res["is_anomaly"])

        outlier_res = detector.detect(self.anomaly_vec)
        self.assertTrue(outlier_res["is_anomaly"])

    def test_linear_regressor(self):
        reg = OnDeviceLinearRegressor()
        res = reg.predict(self.sample_vec)
        self.assertGreater(res["predicted_latency_ms"], 0.0)
        self.assertLess(res["latency_calc_ms"], 5.0)

if __name__ == '__main__':
    unittest.main()
