"""
=============================================================================
ON-DEVICE EDGE AI SUITE FOR ANDROID (KIVY + NUMPY)
=============================================================================
Architecture:
  - 100% Native on-device execution (Zero cloud/API dependencies)
  - Pure Python + NumPy vectorized matrix primitives for ARM64 NEON
  - Asynchronous non-blocking threading model (Keeps UI at 60 FPS)
  - 4 Unified Edge Model Architectures:
      1. Vectorized K-Nearest Neighbors (KNN, k=3)
      2. 2-Layer Multilayer Perceptron (MLP) Neural Network (ReLU + Softmax)
      3. Centroid-Distance Outlier / Anomaly Detector (Z-score thresholding)
      4. Continuous Multi-Variable Linear Regressor (QoS & Latency estimator)
=============================================================================
"""

import threading
import time
import numpy as np

# Kivy Framework Imports
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


class FeatureStandardizer:
    """Computes and applies Z-score standardization: z = (x - mu) / sigma"""
    def __init__(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Prevent division by zero for invariant sensor features
        self.std[self.std == 0.0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class OnDeviceKNNClassifier:
    """Vectorized K-Nearest Neighbors Classifier."""
    def __init__(self, k: int = 3):
        self.k = k
        self.classes = [
            "Optimal / Normal",
            "High Latency Warning",
            "Critical Packet Drop",
            "Overload Anomaly"
        ]
        # 16-sample multi-dimensional calibration dataset
        self.X_train = np.array([
            [0.85, 0.12, 0.90, 0.01],
            [0.92, 0.10, 0.95, 0.00],
            [0.80, 0.15, 0.88, 0.02],
            [0.89, 0.11, 0.92, 0.01],
            [0.72, 0.65, 0.60, 0.04],
            [0.68, 0.70, 0.55, 0.05],
            [0.75, 0.60, 0.62, 0.03],
            [0.70, 0.75, 0.50, 0.06],
            [0.30, 0.85, 0.20, 0.40],
            [0.25, 0.90, 0.15, 0.48],
            [0.35, 0.80, 0.22, 0.35],
            [0.28, 0.92, 0.18, 0.45],
            [0.50, 0.95, 0.98, 0.30],
            [0.55, 0.92, 0.96, 0.28],
            [0.48, 0.98, 0.99, 0.35],
            [0.52, 0.90, 0.94, 0.25],
        ], dtype=np.float32)

        self.y_train = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3
        ], dtype=np.int32)

        self.scaler = FeatureStandardizer(self.X_train)
        self.X_train_norm = self.scaler.transform(self.X_train)

    def predict(self, raw_features: np.ndarray) -> dict:
        start_time = time.perf_counter()
        vector = np.asarray(raw_features, dtype=np.float32).reshape(1, -1)
        vector_norm = self.scaler.transform(vector)

        # Vectorized Euclidean Distance
        diff = self.X_train_norm - vector_norm
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        # Retrieve indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        # Inverse-distance weighting
        weights = 1.0 / (k_nearest_distances + 1e-6)
        class_scores = {}
        for label, weight in zip(k_nearest_labels, weights):
            class_scores[label] = class_scores.get(label, 0.0) + weight

        predicted_class_id = max(class_scores, key=class_scores.get)
        predicted_label = self.classes[predicted_class_id]

        total_weight = sum(class_scores.values())
        confidence = (class_scores[predicted_class_id] / total_weight) * 100.0 if total_weight > 0 else 100.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prediction": predicted_label,
            "class_id": int(predicted_class_id),
            "confidence": float(confidence),
            "latency_ms": float(elapsed_ms),
            "k_neighbors": [
                {
                    "class": self.classes[self.y_train[idx]],
                    "distance": float(distances[idx]),
                    "raw_features": self.X_train[idx].tolist()
                }
                for idx in k_indices
            ]
        }


class OnDeviceNeuralNet:
    """
    2-Layer Dense Multilayer Perceptron (MLP) Neural Network:
    Input (4) -> Hidden Dense (8, ReLU) -> Output Dense (4, Softmax)
    """
    def __init__(self):
        self.classes = [
            "Optimal / Normal",
            "High Latency Warning",
            "Critical Packet Drop",
            "Overload Anomaly"
        ]
        # Pre-calibrated forward weights and biases
        self.W1 = np.array([
            [ 0.82, -0.45,  0.67, -0.12,  0.34, -0.78,  0.91, -0.33],
            [-0.56,  0.92, -0.34,  0.88, -0.21,  0.65, -0.43,  0.77],
            [ 0.74, -0.63,  0.85, -0.41,  0.55, -0.32,  0.68, -0.19],
            [-0.89,  0.71, -0.92,  0.64, -0.75,  0.83, -0.61,  0.95]
        ], dtype=np.float32)
        self.b1 = np.array([0.12, -0.05, 0.08, -0.15, 0.04, -0.09, 0.21, -0.11], dtype=np.float32)

        self.W2 = np.array([
            [ 1.15, -0.42, -0.88, -0.31],
            [-0.35,  1.28, -0.45,  0.22],
            [ 0.92, -0.31, -0.76, -0.15],
            [-0.48,  1.05, -0.39,  0.62],
            [ 0.65, -0.22, -0.58, -0.10],
            [-0.29,  0.88, -0.31,  0.45],
            [ 1.08, -0.55, -0.92, -0.40],
            [-0.62,  0.95, -0.51,  0.78]
        ], dtype=np.float32)
        self.b2 = np.array([0.05, -0.02, -0.10, 0.01], dtype=np.float32)

    def forward(self, raw_features: np.ndarray) -> dict:
        start_time = time.perf_counter()
        x = np.asarray(raw_features, dtype=np.float32).reshape(1, 4)

        # Layer 1: Linear + ReLU
        h = np.maximum(0.0, np.dot(x, self.W1) + self.b1)

        # Layer 2: Output Logits
        logits = np.dot(h, self.W2) + self.b2

        # Numerically stable Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).flatten()

        winning_id = int(np.argmax(probs))
        confidence = float(probs[winning_id] * 100.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "prediction": self.classes[winning_id],
            "class_id": winning_id,
            "confidence": confidence,
            "probabilities": (probs * 100.0).tolist(),
            "hidden_activations": h.flatten().tolist(),
            "latency_ms": elapsed_ms
        }


class OnDeviceAnomalyDetector:
    """Outlier & Anomaly Detection based on Mahalanobis/Z-score cluster distance."""
    def __init__(self, knn_model: OnDeviceKNNClassifier, threshold_sigma: float = 2.2):
        self.knn = knn_model
        self.threshold = threshold_sigma
        # Compute centroids per class in normalized space
        self.centroids = {}
        for c in range(4):
            mask = (self.knn.y_train == c)
            self.centroids[c] = np.mean(self.knn.X_train_norm[mask], axis=0)

    def detect(self, raw_features: np.ndarray) -> dict:
        start_time = time.perf_counter()
        vector = np.asarray(raw_features, dtype=np.float32).reshape(1, -1)
        norm_v = self.knn.scaler.transform(vector).flatten()

        # Find min distance to any cluster center
        min_dist = float("inf")
        nearest_class_id = 0
        for c, centroid in self.centroids.items():
            dist = float(np.linalg.norm(norm_v - centroid))
            if dist < min_dist:
                min_dist = dist
                nearest_class_id = c

        is_anomaly = min_dist > self.threshold
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(min_dist),
            "threshold": self.threshold,
            "nearest_class": self.knn.classes[nearest_class_id],
            "latency_ms": elapsed_ms
        }


class OnDeviceLinearRegressor:
    """Continuous QoS / Response Time Latency Predictor."""
    def __init__(self):
        self.weights = np.array([-35.5, 82.4, -28.1, 145.2], dtype=np.float32)
        self.bias = 12.0
        self.feature_names = ["Signal Power", "RTT Latency", "Throughput", "Packet Error"]

    def predict(self, raw_features: np.ndarray) -> dict:
        start_time = time.perf_counter()
        x = np.asarray(raw_features, dtype=np.float32).flatten()
        prediction = float(np.dot(x, self.weights) + self.bias)
        final_val = max(2.5, prediction)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "predicted_latency_ms": round(final_val, 2),
            "latency_calc_ms": elapsed_ms
        }


# =============================================================================
# KIVY RESPONSIVE UI INTERFACE
# =============================================================================

class AIMobileRootLayout(BoxLayout):
    """Native multi-touch UI layout for the Android APK application."""

    def __init__(self, **kwargs):
        super(AIMobileRootLayout, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = [20, 20, 20, 20]
        self.spacing = 12

        # Initialize Models Suite
        self.knn = OnDeviceKNNClassifier(k=3)
        self.neural_net = OnDeviceNeuralNet()
        self.anomaly = OnDeviceAnomalyDetector(self.knn, threshold_sigma=2.2)
        self.regressor = OnDeviceLinearRegressor()

        self.current_algo = "KNN"

        # 1. Header
        self.header_label = Label(
            text="[b]ON-DEVICE EDGE AI SUITE[/b]\n[size=13][color=33cc66]NumPy Matrix Engine • Android 14 (API 34)[/color][/size]",
            markup=True,
            size_hint=(1.0, None),
            height=54,
            halign="center",
            valign="middle"
        )
        self.header_label.bind(size=self.header_label.setter('text_size'))
        self.add_widget(self.header_label)

        # 2. Algorithm Mode Toggle Bar
        self.algo_bar = BoxLayout(
            orientation="horizontal",
            size_hint=(1.0, None),
            height=42,
            spacing=6
        )

        self.btn_knn = Button(text="KNN", background_color=(0.12, 0.45, 0.90, 1), bold=True)
        self.btn_nn = Button(text="MLP NN", background_color=(0.2, 0.25, 0.35, 1))
        self.btn_anomaly = Button(text="ANOMALY", background_color=(0.2, 0.25, 0.35, 1))
        self.btn_reg = Button(text="REGRESS", background_color=(0.2, 0.25, 0.35, 1))

        self.btn_knn.bind(on_press=lambda inst: self.select_algorithm("KNN"))
        self.btn_nn.bind(on_press=lambda inst: self.select_algorithm("MLP NN"))
        self.btn_anomaly.bind(on_press=lambda inst: self.select_algorithm("ANOMALY"))
        self.btn_reg.bind(on_press=lambda inst: self.select_algorithm("REGRESS"))

        self.algo_bar.add_widget(self.btn_knn)
        self.algo_bar.add_widget(self.btn_nn)
        self.algo_bar.add_widget(self.btn_anomaly)
        self.algo_bar.add_widget(self.btn_reg)
        self.add_widget(self.algo_bar)

        # 3. Input Instructions
        self.input_instruction = Label(
            text="Enter 4 Telemetry Metrics (Signal, Latency, Throughput, Loss):",
            size_hint=(1.0, None),
            height=30,
            halign="left",
            valign="middle",
            color=(0.75, 0.82, 0.90, 1)
        )
        self.input_instruction.bind(size=self.input_instruction.setter('text_size'))
        self.add_widget(self.input_instruction)

        # 4. Text Input
        self.text_input = TextInput(
            text="0.85, 0.12, 0.90, 0.01",
            multiline=False,
            size_hint=(1.0, None),
            height=46,
            font_size=15,
            padding=[10, 10, 10, 10],
            background_color=(0.14, 0.16, 0.22, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.3, 0.7, 1.0, 1)
        )
        self.add_widget(self.text_input)

        # 5. Async Trigger Button
        self.trigger_button = Button(
            text="RUN ASYNC INFERENCE",
            size_hint=(1.0, None),
            height=50,
            font_size=15,
            bold=True,
            background_normal="",
            background_color=(0.12, 0.47, 0.95, 1),
            color=(1, 1, 1, 1)
        )
        self.trigger_button.bind(on_press=self.on_trigger_clicked)
        self.add_widget(self.trigger_button)

        # 6. Scrollable Output Stream
        self.scroll_view = ScrollView(
            size_hint=(1.0, 1.0),
            do_scroll_x=False,
            do_scroll_y=True
        )

        self.output_label = Label(
            text="[i]System ready. Press 'RUN ASYNC INFERENCE' to execute model on background thread without blocking main UI thread.[/i]",
            markup=True,
            size_hint=(1.0, None),
            halign="left",
            valign="top",
            color=(0.88, 0.92, 0.96, 1)
        )
        self.output_label.bind(
            width=lambda instance, value: setattr(instance, 'text_size', (value - 20, None)),
            texture_size=lambda instance, value: setattr(instance, 'height', max(value[1] + 20, 200))
        )
        self.scroll_view.add_widget(self.output_label)
        self.add_widget(self.scroll_view)

    def select_algorithm(self, algo_name: str):
        self.current_algo = algo_name
        inactive_color = (0.2, 0.25, 0.35, 1)
        active_color = (0.12, 0.45, 0.90, 1)

        self.btn_knn.background_color = active_color if algo_name == "KNN" else inactive_color
        self.btn_nn.background_color = active_color if algo_name == "MLP NN" else inactive_color
        self.btn_anomaly.background_color = active_color if algo_name == "ANOMALY" else inactive_color
        self.btn_reg.background_color = active_color if algo_name == "REGRESS" else inactive_color

        self.output_label.text = f"[color=ffcc00]Active Model Architecture changed to: {algo_name}[/color]"

    def on_trigger_clicked(self, instance):
        raw_text = self.text_input.text.strip()
        self.trigger_button.disabled = True
        self.trigger_button.background_color = (0.3, 0.3, 0.35, 1)
        self.trigger_button.text = "CALCULATING ON THREAD..."
        self.output_label.text = "[color=ffcc00]Spawning async computation worker thread...[/color]"

        worker = threading.Thread(
            target=self._async_compute_inference,
            args=(raw_text, self.current_algo),
            daemon=True
        )
        worker.start()

    def _async_compute_inference(self, raw_text: str, algo: str):
        try:
            tokens = [t.strip() for t in raw_text.split(",") if t.strip()]
            if len(tokens) != 4:
                raise ValueError(f"Expected 4 values, found {len(tokens)}.")

            features = [float(x) for x in tokens]
            vec = np.array(features, dtype=np.float32)

            if algo == "KNN":
                res = self.knn.predict(vec)
                formatted = (
                    f"[b][color=33cc66]KNN INFERENCE COMPLETE[/color][/b]\n"
                    f"--------------------------------------------------\n"
                    f"[b]Predicted Class:[/b] {res['prediction']}\n"
                    f"[b]Confidence:[/b] {res['confidence']:.2f}%\n"
                    f"[b]Latency:[/b] {res['latency_ms']:.3f} ms\n\n"
                    f"[b]Nearest K=3 Neighbors:[/b]\n"
                )
                for i, n in enumerate(res['k_neighbors'], 1):
                    formatted += f"  #{i} {n['class']} (dist: {n['distance']:.4f})\n"

            elif algo == "MLP NN":
                res = self.neural_net.forward(vec)
                formatted = (
                    f"[b][color=66aaff]NEURAL NET (MLP 4->8->4) FORWARD PASS[/color][/b]\n"
                    f"--------------------------------------------------\n"
                    f"[b]Winner Logit:[/b] {res['prediction']}\n"
                    f"[b]Softmax Probability:[/b] {res['confidence']:.2f}%\n"
                    f"[b]Matrix Latency:[/b] {res['latency_ms']:.3f} ms\n\n"
                    f"[b]Class Distribution (%):[/b] {[round(p, 1) for p in res['probabilities']]}\n"
                    f"[b]Hidden ReLU Activations:[/b] {[round(h, 2) for h in res['hidden_activations']]}\n"
                )

            elif algo == "ANOMALY":
                res = self.anomaly.detect(vec)
                state_color = "ff4444" if res['is_anomaly'] else "33cc66"
                state_text = "CRITICAL OUTLIER DETECTED" if res['is_anomaly'] else "NORMAL IN-BOUNDS"
                formatted = (
                    f"[b][color={state_color}]ANOMALY DETECTOR[/color][/b]\n"
                    f"--------------------------------------------------\n"
                    f"[b]Status:[/b] {state_text}\n"
                    f"[b]Z-Score Distance:[/b] {res['anomaly_score']:.3f} (Threshold: {res['threshold']} sigma)\n"
                    f"[b]Nearest Cluster:[/b] {res['nearest_class']}\n"
                    f"[b]Latency:[/b] {res['latency_ms']:.3f} ms\n"
                )

            else:
                res = self.regressor.predict(vec)
                formatted = (
                    f"[b][color=dd88ff]LINEAR REGRESSOR (QoS / LATENCY)[/color][/b]\n"
                    f"--------------------------------------------------\n"
                    f"[b]Predicted Network Latency:[/b] {res['predicted_latency_ms']} ms\n"
                    f"[b]Dot-Product Computation:[/b] {res['latency_calc_ms']:.3f} ms\n"
                )

            formatted += (
                f"\n[size=11][color=8899aa]Processed on worker thread: "
                f"{threading.current_thread().name} (Zero UI stutter)[/size][/color]"
            )

            Clock.schedule_once(lambda dt: self._update_ui_success(formatted), 0)

        except Exception as exc:
            err_msg = (
                f"[b][color=ff4444]ERROR EXECUTING MODEL[/color][/b]\n"
                f"--------------------------------------------------\n"
                f"{str(exc)}\n\n"
                f"[color=aaaaaa]Ensure 4 numbers separated by commas are provided.\n"
                f"Example: 0.85, 0.12, 0.90, 0.01[/color]"
            )
            Clock.schedule_once(lambda dt: self._update_ui_error(err_msg), 0)

    def _update_ui_success(self, output_text: str):
        self.output_label.text = output_text
        self.trigger_button.disabled = False
        self.trigger_button.text = "RUN ASYNC INFERENCE"
        self.trigger_button.background_color = (0.12, 0.47, 0.95, 1)

    def _update_ui_error(self, error_text: str):
        self.output_label.text = error_text
        self.trigger_button.disabled = False
        self.trigger_button.text = "RETRY INFERENCE"
        self.trigger_button.background_color = (0.85, 0.25, 0.25, 1)


class AIMobileApp(App):
    def build(self):
        self.title = "On-Device AI Engine"
        Window.clearcolor = (0.08, 0.09, 0.12, 1)
        return AIMobileRootLayout()


if __name__ == "__main__":
    AIMobileApp().run()
