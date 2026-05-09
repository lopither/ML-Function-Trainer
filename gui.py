from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.animator import SnapshotAnimator
from app.dataset import (
    DatasetBundle,
    DatasetGenerationError,
    NormalizationStats,
    generate_dataset,
)
from app.model import FunctionApproximator, activation_names
from app.parser import FunctionParseError, ParsedFunction, SafeFunctionParser
from app.trainer import PredictionSnapshot, Trainer, TrainingResult
from app.utils import format_loss, load_json, resolve_device, save_json, set_random_seed


class PlotCanvas(FigureCanvas):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(8, 7.5), dpi=100)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax_function = self.figure.add_subplot(3, 1, 1)
        self.ax_residual = self.figure.add_subplot(3, 1, 2)
        self.ax_loss = self.figure.add_subplot(3, 1, 3)
        self.figure.subplots_adjust(hspace=0.5, left=0.08, right=0.98, top=0.94)


class TrainingWorker(QObject):
    progress = pyqtSignal(int, int, float, object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, dataset: DatasetBundle, params: dict[str, object]) -> None:
        super().__init__()
        self.dataset = dataset
        self.params = params
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            seed = int(self.params["seed"])
            set_random_seed(seed)
            train_tensors, test_tensors, prediction_x, normalization_stats = (
                self._prepare_training_inputs()
            )

            model = FunctionApproximator(
                input_dim=1,
                output_dim=1,
                hidden_layers=int(self.params["hidden_layers"]),
                neurons_per_layer=int(self.params["neurons"]),
                activation_name=str(self.params["activation"]),
            )

            device = resolve_device(str(self.params["device"]))
            trainer = Trainer(
                model=model,
                train_tensors=train_tensors,
                test_tensors=test_tensors,
                prediction_x=prediction_x,
                learning_rate=float(self.params["learning_rate"]),
                optimizer_name=str(self.params["optimizer"]),
                epochs=int(self.params["epochs"]),
                batch_size=int(self.params["batch_size"]),
                snapshot_interval=int(self.params["snapshot_interval"]),
                device=device,
            )

            emit_every = max(1, int(self.params["epochs"]) // 200)

            def progress_callback(
                epoch: int,
                total_epochs: int,
                train_loss: float,
                test_loss: float | None,
            ) -> None:
                if (
                    epoch == 1
                    or epoch == total_epochs
                    or epoch % emit_every == 0
                ):
                    self.progress.emit(
                        epoch,
                        total_epochs,
                        self._display_loss(train_loss, normalization_stats),
                        self._display_loss(test_loss, normalization_stats),
                    )

            result = trainer.train(
                progress_callback=progress_callback,
                stop_checker=lambda: self._stop_requested,
            )
            if normalization_stats is not None:
                result = self._denormalize_result(result, normalization_stats)
            result.normalization_stats = normalization_stats
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _prepare_training_inputs(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        np.ndarray,
        NormalizationStats | None,
    ]:
        if not bool(self.params.get("normalize_training", False)):
            return (
                (self.dataset.x_train, self.dataset.y_train),
                (self.dataset.x_test, self.dataset.y_test),
                self.dataset.x_plot,
                None,
            )

        stats = NormalizationStats.from_training_arrays(
            self.dataset.x_train_np,
            self.dataset.y_train_np,
        )
        x_train = torch.tensor(
            stats.normalize_x_np(self.dataset.x_train_np).reshape(-1, 1),
            dtype=torch.float32,
        )
        y_train = torch.tensor(
            stats.normalize_y_np(self.dataset.y_train_np).reshape(-1, 1),
            dtype=torch.float32,
        )
        x_test = torch.tensor(
            stats.normalize_x_np(self.dataset.x_test_np).reshape(-1, 1),
            dtype=torch.float32,
        )
        y_test = torch.tensor(
            stats.normalize_y_np(self.dataset.y_test_np).reshape(-1, 1),
            dtype=torch.float32,
        )
        prediction_x = stats.normalize_x_np(self.dataset.x_plot)
        return (x_train, y_train), (x_test, y_test), prediction_x, stats

    def _display_loss(
        self,
        loss: float | None,
        normalization_stats: NormalizationStats | None,
    ) -> float | None:
        if normalization_stats is None:
            return loss
        return normalization_stats.mse_to_original_scale(loss)

    def _denormalize_result(
        self,
        result: TrainingResult,
        normalization_stats: NormalizationStats,
    ) -> TrainingResult:
        result.loss_history = [
            float(normalization_stats.mse_to_original_scale(loss))
            for loss in result.loss_history
        ]
        result.test_loss_history = [
            normalization_stats.mse_to_original_scale(loss)
            for loss in result.test_loss_history
        ]
        result.snapshots = [
            PredictionSnapshot(
                epoch=snapshot.epoch,
                predictions=normalization_stats.denormalize_y_np(snapshot.predictions),
                train_loss=float(
                    normalization_stats.mse_to_original_scale(snapshot.train_loss)
                ),
                test_loss=normalization_stats.mse_to_original_scale(snapshot.test_loss),
            )
            for snapshot in result.snapshots
        ]
        return result


class MainWindow(QMainWindow):
    PRESETS: dict[str, dict[str, object]] = {
        "Good Fit Demo": {
            "function": "sin(x)",
            "x_min": -5.0,
            "x_max": 5.0,
            "training_points": 250,
            "noise_level": 0.0,
            "hidden_layers": 2,
            "neurons_per_layer": 64,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 1000,
            "batch_size": 32,
            "snapshot_interval": 25,
            "normalize_training": True,
        },
        "Underfitting Demo": {
            "function": "sin(5*x)",
            "x_min": -5.0,
            "x_max": 5.0,
            "training_points": 220,
            "noise_level": 0.0,
            "hidden_layers": 1,
            "neurons_per_layer": 4,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 800,
            "batch_size": 32,
            "snapshot_interval": 20,
            "normalize_training": True,
        },
        "Overfitting Demo": {
            "function": "sin(x) + 0.2*x",
            "x_min": -5.0,
            "x_max": 5.0,
            "training_points": 45,
            "noise_level": 0.25,
            "hidden_layers": 4,
            "neurons_per_layer": 128,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 2500,
            "batch_size": 16,
            "snapshot_interval": 50,
            "normalize_training": True,
        },
        "Noisy Data Demo": {
            "function": "exp(-x^2) * cos(3*x)",
            "x_min": -4.0,
            "x_max": 4.0,
            "training_points": 300,
            "noise_level": 0.08,
            "hidden_layers": 3,
            "neurons_per_layer": 96,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 1500,
            "batch_size": 32,
            "snapshot_interval": 30,
            "normalize_training": True,
        },
        "Extrapolation Demo": {
            "function": "x^2",
            "x_min": -2.0,
            "x_max": 2.0,
            "training_points": 180,
            "noise_level": 0.0,
            "hidden_layers": 2,
            "neurons_per_layer": 64,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 1200,
            "batch_size": 32,
            "snapshot_interval": 30,
            "normalize_training": True,
        },
        "High Frequency Demo": {
            "function": "sin(8*x)",
            "x_min": -4.0,
            "x_max": 4.0,
            "training_points": 400,
            "noise_level": 0.0,
            "hidden_layers": 3,
            "neurons_per_layer": 128,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 2500,
            "batch_size": 32,
            "snapshot_interval": 50,
            "normalize_training": True,
        },
        "Large Scale Normalization Demo": {
            "function": "x^2 + 3*x - 1",
            "x_min": -50.0,
            "x_max": 50.0,
            "training_points": 350,
            "noise_level": 0.0,
            "hidden_layers": 3,
            "neurons_per_layer": 128,
            "activation": "Tanh",
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epochs": 2000,
            "batch_size": 32,
            "snapshot_interval": 40,
            "normalize_training": True,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Function Visualizer")
        self.resize(1320, 840)

        self.parser = SafeFunctionParser()
        self.parsed_function: ParsedFunction | None = None
        self.dataset: DatasetBundle | None = None
        self.training_result: TrainingResult | None = None
        self.current_snapshot: PredictionSnapshot | None = None
        self.comparison_result: dict[str, float] | None = None
        self.model_config: dict[str, object] = {}

        self.training_thread: QThread | None = None
        self.training_worker: TrainingWorker | None = None
        self.reset_pending = False

        self.animator = SnapshotAnimator(self)
        self.animator.frame_changed.connect(self._show_snapshot)
        self.animator.playback_started.connect(
            lambda: self.status_label.setText("Animation playing.")
        )
        self.animator.playback_stopped.connect(
            lambda: self.status_label.setText("Animation paused.")
        )

        self._build_ui()
        self._connect_signals()
        self._draw_plots()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 940])

        root_layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_left_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self._build_function_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_training_group())
        layout.addWidget(self._build_animation_group())
        layout.addWidget(self._build_compare_group())
        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_utility_group())

        scroll_area.setWidget(container)
        return scroll_area

    def _build_function_group(self) -> QGroupBox:
        group = QGroupBox("Function and Data")
        form = QFormLayout(group)

        self.function_input = QLineEdit("sin(x)")
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.x_min_spin.setDecimals(4)
        self.x_min_spin.setValue(-5.0)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.x_max_spin.setDecimals(4)
        self.x_max_spin.setValue(5.0)

        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 100_000)
        self.points_spin.setValue(250)

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 1_000_000.0)
        self.noise_spin.setDecimals(5)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.0)

        self.generate_button = QPushButton("Generate Data")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.apply_preset_button = QPushButton("Apply Preset")

        form.addRow("Preset", self.preset_combo)
        form.addRow(self.apply_preset_button)
        form.addRow("Mathematical function", self.function_input)
        form.addRow("x minimum", self.x_min_spin)
        form.addRow("x maximum", self.x_max_spin)
        form.addRow("Training points", self.points_spin)
        form.addRow("Noise level", self.noise_spin)
        form.addRow(self.generate_button)
        return group

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        form = QFormLayout(group)

        self.hidden_layers_spin = QSpinBox()
        self.hidden_layers_spin.setRange(0, 20)
        self.hidden_layers_spin.setValue(2)

        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(1, 4096)
        self.neurons_spin.setValue(64)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(activation_names())
        self.activation_combo.setCurrentText("Tanh")

        form.addRow("Hidden layers", self.hidden_layers_spin)
        form.addRow("Neurons per layer", self.neurons_spin)
        form.addRow("Activation function", self.activation_combo)
        return group

    def _build_training_group(self) -> QGroupBox:
        group = QGroupBox("Training")
        form = QFormLayout(group)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.000001, 10.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(0.0005)
        self.learning_rate_spin.setValue(0.001)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500_000)
        self.epochs_spin.setValue(1000)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100_000)
        self.batch_size_spin.setValue(32)

        self.snapshot_interval_spin = QSpinBox()
        self.snapshot_interval_spin.setRange(1, 100_000)
        self.snapshot_interval_spin.setValue(25)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA")

        self.normalize_checkbox = QCheckBox("Normalize x/y during training")
        self.normalize_checkbox.setChecked(True)

        self.train_button = QPushButton("Train Model")
        self.reset_button = QPushButton("Reset")

        button_row = QHBoxLayout()
        button_row.addWidget(self.train_button)
        button_row.addWidget(self.reset_button)

        form.addRow("Learning rate", self.learning_rate_spin)
        form.addRow("Optimizer", self.optimizer_combo)
        form.addRow("Epochs", self.epochs_spin)
        form.addRow("Batch size", self.batch_size_spin)
        form.addRow("Snapshot interval", self.snapshot_interval_spin)
        form.addRow("Random seed", self.seed_spin)
        form.addRow("Device", self.device_combo)
        form.addRow("Normalization", self.normalize_checkbox)
        form.addRow(button_row)
        return group

    def _build_animation_group(self) -> QGroupBox:
        group = QGroupBox("Animation")
        layout = QVBoxLayout(group)

        self.play_button = QPushButton("Play Animation")
        self.pause_button = QPushButton("Pause")
        self.step_back_button = QPushButton("Step Back")
        self.step_forward_button = QPushButton("Step Forward")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(5)

        button_grid = QGridLayout()
        button_grid.addWidget(self.play_button, 0, 0, 1, 2)
        button_grid.addWidget(self.pause_button, 1, 0)
        button_grid.addWidget(self.step_forward_button, 1, 1)
        button_grid.addWidget(self.step_back_button, 2, 0, 1, 2)

        layout.addLayout(button_grid)
        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self.speed_slider)
        return group

    def _build_compare_group(self) -> QGroupBox:
        group = QGroupBox("Compare")
        form = QFormLayout(group)

        self.compare_x_spin = QDoubleSpinBox()
        self.compare_x_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.compare_x_spin.setDecimals(6)
        self.compare_x_spin.setSingleStep(0.1)
        self.compare_x_spin.setValue(0.0)

        self.compare_button = QPushButton("Compare at x")
        self.compare_button.setEnabled(False)
        self.compare_true_label = QLabel("n/a")
        self.compare_model_label = QLabel("n/a")
        self.compare_signed_error_label = QLabel("n/a")
        self.compare_abs_error_label = QLabel("n/a")
        self.compare_relative_error_label = QLabel("n/a")
        self.compare_analysis_label = QLabel("Generate data and train a model first.")
        self.compare_analysis_label.setWordWrap(True)

        form.addRow("x value", self.compare_x_spin)
        form.addRow(self.compare_button)
        form.addRow("True f(x)", self.compare_true_label)
        form.addRow("Model output", self.compare_model_label)
        form.addRow("Signed error", self.compare_signed_error_label)
        form.addRow("Absolute error", self.compare_abs_error_label)
        form.addRow("Relative error", self.compare_relative_error_label)
        form.addRow("Analysis", self.compare_analysis_label)
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        form = QFormLayout(group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.train_loss_label = QLabel("n/a")
        self.test_loss_label = QLabel("n/a")

        form.addRow("Progress", self.progress_bar)
        form.addRow("Status", self.status_label)
        form.addRow("Train loss", self.train_loss_label)
        form.addRow("Test loss", self.test_loss_label)
        return group

    def _build_utility_group(self) -> QGroupBox:
        group = QGroupBox("Utilities")
        layout = QVBoxLayout(group)

        self.save_model_button = QPushButton("Export Trained Model")
        self.save_model_button.setEnabled(False)
        self.export_settings_button = QPushButton("Export Settings")
        self.import_settings_button = QPushButton("Import Settings")

        layout.addWidget(self.save_model_button)
        layout.addWidget(self.export_settings_button)
        layout.addWidget(self.import_settings_button)
        return group

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.canvas = PlotCanvas(panel)
        layout.addWidget(self.canvas)
        return panel

    def _connect_signals(self) -> None:
        self.generate_button.clicked.connect(self.generate_data)
        self.train_button.clicked.connect(self.train_model)
        self.reset_button.clicked.connect(self.reset)
        self.play_button.clicked.connect(self.play_animation)
        self.pause_button.clicked.connect(self.animator.pause)
        self.step_back_button.clicked.connect(self.animator.step_backward)
        self.step_forward_button.clicked.connect(self.animator.step_forward)
        self.speed_slider.valueChanged.connect(self.animator.set_speed)
        self.compare_button.clicked.connect(self.compare_at_x)
        self.apply_preset_button.clicked.connect(self.apply_selected_preset)
        self.save_model_button.clicked.connect(self.export_model)
        self.export_settings_button.clicked.connect(self.export_settings)
        self.import_settings_button.clicked.connect(self.import_settings)

    def _collect_settings(self) -> dict[str, object]:
        return {
            "function": self.function_input.text(),
            "x_min": self.x_min_spin.value(),
            "x_max": self.x_max_spin.value(),
            "training_points": self.points_spin.value(),
            "noise_level": self.noise_spin.value(),
            "hidden_layers": self.hidden_layers_spin.value(),
            "neurons_per_layer": self.neurons_spin.value(),
            "activation": self.activation_combo.currentText(),
            "learning_rate": self.learning_rate_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "snapshot_interval": self.snapshot_interval_spin.value(),
            "seed": self.seed_spin.value(),
            "device": self.device_combo.currentText(),
            "normalize_training": self.normalize_checkbox.isChecked(),
        }

    def _apply_settings(self, settings: dict[str, object]) -> None:
        self.function_input.setText(str(settings.get("function", self.function_input.text())))
        self.x_min_spin.setValue(float(settings.get("x_min", self.x_min_spin.value())))
        self.x_max_spin.setValue(float(settings.get("x_max", self.x_max_spin.value())))
        self.points_spin.setValue(int(settings.get("training_points", self.points_spin.value())))
        self.noise_spin.setValue(float(settings.get("noise_level", self.noise_spin.value())))
        self.hidden_layers_spin.setValue(
            int(settings.get("hidden_layers", self.hidden_layers_spin.value()))
        )
        self.neurons_spin.setValue(
            int(settings.get("neurons_per_layer", self.neurons_spin.value()))
        )
        self.activation_combo.setCurrentText(
            str(settings.get("activation", self.activation_combo.currentText()))
        )
        self.learning_rate_spin.setValue(
            float(settings.get("learning_rate", self.learning_rate_spin.value()))
        )
        self.optimizer_combo.setCurrentText(
            str(settings.get("optimizer", self.optimizer_combo.currentText()))
        )
        self.epochs_spin.setValue(int(settings.get("epochs", self.epochs_spin.value())))
        self.batch_size_spin.setValue(
            int(settings.get("batch_size", self.batch_size_spin.value()))
        )
        self.snapshot_interval_spin.setValue(
            int(settings.get("snapshot_interval", self.snapshot_interval_spin.value()))
        )
        self.seed_spin.setValue(int(settings.get("seed", self.seed_spin.value())))
        device = str(settings.get("device", self.device_combo.currentText()))
        if self.device_combo.findText(device) >= 0:
            self.device_combo.setCurrentText(device)
        self.normalize_checkbox.setChecked(
            bool(settings.get("normalize_training", self.normalize_checkbox.isChecked()))
        )

    def apply_selected_preset(self) -> None:
        if self._training_running():
            self._show_error("Training Running", "Wait for training to finish before applying a preset.")
            return

        preset_name = self.preset_combo.currentText()
        preset = self.PRESETS.get(preset_name)
        if preset is None:
            return

        self._apply_settings(preset)
        self._clear_state()
        self.compare_x_spin.setValue((self.x_min_spin.value() + self.x_max_spin.value()) / 2.0)
        self.status_label.setText(f"Preset applied: {preset_name}. Generate data to begin.")

    def generate_data(self) -> None:
        if self._training_running():
            self._show_error("Training Running", "Wait for training to finish before regenerating data.")
            return

        try:
            parsed = self.parser.parse(self.function_input.text())
            self.dataset = generate_dataset(
                parsed.numpy_function,
                x_min=self.x_min_spin.value(),
                x_max=self.x_max_spin.value(),
                num_points=self.points_spin.value(),
                noise_level=self.noise_spin.value(),
                seed=self.seed_spin.value(),
            )
            self.parsed_function = parsed
        except (FunctionParseError, DatasetGenerationError) as exc:
            self._show_error("Data Generation Error", str(exc))
            return
        except Exception as exc:
            self._show_error("Data Generation Error", f"Unexpected error: {exc}")
            return

        self.training_result = None
        self.current_snapshot = None
        self.comparison_result = None
        self.animator.set_snapshots([])
        self.progress_bar.setValue(0)
        self.train_loss_label.setText("n/a")
        self.test_loss_label.setText("n/a")
        self.compare_true_label.setText("n/a")
        self.compare_model_label.setText("n/a")
        self.compare_signed_error_label.setText("n/a")
        self.compare_abs_error_label.setText("n/a")
        self.compare_relative_error_label.setText("n/a")
        self.compare_analysis_label.setText("Train a model, then compare at a selected x value.")
        self.compare_x_spin.setValue((self.x_min_spin.value() + self.x_max_spin.value()) / 2.0)
        self.compare_button.setEnabled(False)
        self.save_model_button.setEnabled(False)

        message = f"Generated {self.dataset.x_clean.size} finite data points."
        if self.dataset.removed_points:
            message += f" Removed {self.dataset.removed_points} invalid point(s)."
        self.status_label.setText(message)
        self._draw_plots()

    def train_model(self) -> None:
        if self.dataset is None:
            self._show_error("No Data", "Generate data before training a model.")
            return
        if self._training_running():
            self._show_error("Training Running", "Training is already in progress.")
            return

        epochs = self.epochs_spin.value()
        if epochs > 20_000:
            reply = QMessageBox.question(
                self,
                "Long Training Run",
                "This epoch count may take a long time. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        params = self._collect_training_params()
        self.model_config = {
            "hidden_layers": params["hidden_layers"],
            "neurons": params["neurons"],
            "activation": params["activation"],
            "input_dim": 1,
            "output_dim": 1,
            "normalize_training": params["normalize_training"],
        }

        self.progress_bar.setValue(0)
        self.status_label.setText("Training model...")
        self.train_loss_label.setText("n/a")
        self.test_loss_label.setText("n/a")
        self._set_training_ui_enabled(False)

        self.training_thread = QThread(self)
        self.training_worker = TrainingWorker(self.dataset, params)
        self.training_worker.moveToThread(self.training_thread)

        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self._on_training_progress)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self._on_training_thread_finished)
        self.training_thread.start()

    def _collect_training_params(self) -> dict[str, object]:
        return {
            "hidden_layers": self.hidden_layers_spin.value(),
            "neurons": self.neurons_spin.value(),
            "activation": self.activation_combo.currentText(),
            "learning_rate": self.learning_rate_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "snapshot_interval": self.snapshot_interval_spin.value(),
            "seed": self.seed_spin.value(),
            "device": self.device_combo.currentText(),
            "normalize_training": self.normalize_checkbox.isChecked(),
        }

    def _on_training_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        test_loss: float | None,
    ) -> None:
        percent = int(round((epoch / max(total_epochs, 1)) * 100))
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Training epoch {epoch} of {total_epochs}.")
        self.train_loss_label.setText(format_loss(train_loss))
        self.test_loss_label.setText(format_loss(test_loss))

    def _on_training_finished(self, result: TrainingResult) -> None:
        if self.reset_pending:
            self.reset_pending = False
            self._clear_state()
            return

        self.training_result = result
        if isinstance(result.normalization_stats, NormalizationStats):
            self.model_config["normalization"] = result.normalization_stats.to_dict()
        self.animator.set_snapshots(result.snapshots)
        self.progress_bar.setValue(100)

        if result.snapshots:
            self._show_snapshot(len(result.snapshots) - 1, result.snapshots[-1])
        else:
            self._draw_plots()

        final_train = result.loss_history[-1] if result.loss_history else None
        final_test = result.test_loss_history[-1] if result.test_loss_history else None
        self.train_loss_label.setText(format_loss(final_train))
        self.test_loss_label.setText(format_loss(final_test))

        status = "Training finished."
        if result.stopped_early:
            status = "Training stopped early."
        if final_train is not None and final_train > 1.0:
            status += " Loss is still high; try more epochs, more neurons, or a lower learning rate."
        if isinstance(result.normalization_stats, NormalizationStats):
            status += " Training used x/y normalization."
        self.status_label.setText(status)

    def _on_training_failed(self, message: str) -> None:
        self._show_error("Training Error", message)
        self.status_label.setText("Training failed.")
        self.progress_bar.setValue(0)

    def _on_training_thread_finished(self) -> None:
        self.training_thread = None
        self.training_worker = None
        self._set_training_ui_enabled(True)

    def play_animation(self) -> None:
        if not self.animator.has_snapshots():
            self._show_error("No Animation", "Train a model before playing the animation.")
            return
        self.animator.play()

    def compare_at_x(self) -> None:
        if self.dataset is None or self.parsed_function is None:
            self._show_error("No Data", "Generate data before comparing function values.")
            return
        if self.training_result is None:
            self._show_error("No Trained Model", "Train a model before comparing predictions.")
            return
        if self._training_running():
            self._show_error("Training Running", "Wait for training to finish before comparing outputs.")
            return

        x_value = float(self.compare_x_spin.value())
        try:
            true_value = self._evaluate_true_value(x_value)
            model_value = self._predict_model_value(x_value)
        except ValueError as exc:
            self._show_error("Compare Error", str(exc))
            return
        except Exception as exc:
            self._show_error("Compare Error", f"Unexpected error: {exc}")
            return

        signed_error = model_value - true_value
        absolute_error = abs(signed_error)
        relative_error = None
        if abs(true_value) > 1e-12:
            relative_error = absolute_error / abs(true_value)

        self.comparison_result = {
            "x": x_value,
            "true": true_value,
            "model": model_value,
            "signed_error": signed_error,
            "absolute_error": absolute_error,
            "relative_error": float("nan") if relative_error is None else relative_error,
        }

        self.compare_true_label.setText(self._format_number(true_value))
        self.compare_model_label.setText(self._format_number(model_value))
        self.compare_signed_error_label.setText(self._format_number(signed_error))
        self.compare_abs_error_label.setText(self._format_number(absolute_error))
        if relative_error is None:
            self.compare_relative_error_label.setText("n/a near zero true value")
        else:
            self.compare_relative_error_label.setText(f"{relative_error * 100:.4f}%")
        self.compare_analysis_label.setText(
            self._comparison_analysis(x_value, true_value, model_value, absolute_error, relative_error)
        )
        self.status_label.setText("Comparison completed.")
        self._draw_plots()

    def _evaluate_true_value(self, x_value: float) -> float:
        if self.parsed_function is None:
            raise ValueError("No parsed function is available.")

        with np.errstate(all="ignore"):
            values = np.asarray(
                self.parsed_function.numpy_function(np.array([x_value], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)

        if values.size == 0 or not np.isfinite(values[0]):
            raise ValueError(
                "The true function is not finite at this x value. Try a value inside the function domain."
            )
        return float(values[0])

    def _predict_model_value(self, x_value: float) -> float:
        if self.training_result is None:
            raise ValueError("No trained model is available.")

        model = self.training_result.model
        model.eval()
        model_input = x_value
        if isinstance(self.training_result.normalization_stats, NormalizationStats):
            model_input = self.training_result.normalization_stats.normalize_x_value(x_value)
        with torch.no_grad():
            prediction = model(torch.tensor([[model_input]], dtype=torch.float32))
        value = float(prediction.detach().cpu().numpy().reshape(-1)[0])
        if isinstance(self.training_result.normalization_stats, NormalizationStats):
            value = self.training_result.normalization_stats.denormalize_y_value(value)
        if not np.isfinite(value):
            raise ValueError("The trained model returned a non-finite prediction.")
        return value

    def _comparison_analysis(
        self,
        x_value: float,
        true_value: float,
        model_value: float,
        absolute_error: float,
        relative_error: float | None,
    ) -> str:
        if self.dataset is None:
            return "No dataset is available."

        x_min = float(np.min(self.dataset.x_clean))
        x_max = float(np.max(self.dataset.x_clean))
        in_range = x_min <= x_value <= x_max
        location = "inside the training range" if in_range else "outside the training range"
        direction = "overestimates" if model_value > true_value else "underestimates"
        if abs(model_value - true_value) <= 1e-12:
            direction = "matches"

        if relative_error is None:
            error_text = f"absolute error is {self._format_number(absolute_error)}"
        else:
            error_text = (
                f"absolute error is {self._format_number(absolute_error)} "
                f"({relative_error * 100:.3f}% relative)"
            )

        caution = ""
        if not in_range:
            caution = " This is extrapolation, so larger errors are expected."
        elif relative_error is not None and relative_error > 0.10:
            caution = " This point has a relatively large error; try more epochs, lower learning rate, or a wider model."
        elif absolute_error < 0.01:
            caution = " The approximation is tight at this point."

        return (
            f"At x = {self._format_number(x_value)}, the model {direction} f(x); "
            f"{error_text}. The point is {location}.{caution}"
        )

    def _format_number(self, value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        if abs(value) >= 1000 or (0 < abs(value) < 0.0001):
            return f"{value:.6e}"
        return f"{value:.6f}"

    def reset(self) -> None:
        if self._training_running():
            reply = QMessageBox.question(
                self,
                "Stop Training",
                "Stop the current training run and reset the app?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes and self.training_worker is not None:
                self.reset_pending = True
                self.training_worker.stop()
                self.status_label.setText("Stopping training...")
            return

        self._clear_state()

    def _clear_state(self) -> None:
        self.animator.set_snapshots([])
        self.parsed_function = None
        self.dataset = None
        self.training_result = None
        self.current_snapshot = None
        self.comparison_result = None
        self.model_config = {}
        self.progress_bar.setValue(0)
        self.train_loss_label.setText("n/a")
        self.test_loss_label.setText("n/a")
        self.compare_true_label.setText("n/a")
        self.compare_model_label.setText("n/a")
        self.compare_signed_error_label.setText("n/a")
        self.compare_abs_error_label.setText("n/a")
        self.compare_relative_error_label.setText("n/a")
        self.compare_analysis_label.setText("Generate data and train a model first.")
        self.compare_button.setEnabled(False)
        self.save_model_button.setEnabled(False)
        self.status_label.setText("Ready.")
        self._draw_plots()

    def _show_snapshot(self, index: int, snapshot: object) -> None:
        if not isinstance(snapshot, PredictionSnapshot):
            return
        self.current_snapshot = snapshot
        self.status_label.setText(
            f"Showing snapshot {index + 1}: epoch {snapshot.epoch}."
        )
        self.train_loss_label.setText(format_loss(snapshot.train_loss))
        self.test_loss_label.setText(format_loss(snapshot.test_loss))
        self._draw_plots(snapshot)

    def _draw_plots(self, snapshot: PredictionSnapshot | None = None) -> None:
        ax_function = self.canvas.ax_function
        ax_residual = self.canvas.ax_residual
        ax_loss = self.canvas.ax_loss
        ax_function.clear()
        ax_residual.clear()
        ax_loss.clear()

        if self.dataset is None:
            ax_function.set_title("Neural Network Function Approximation")
            ax_function.set_xlabel("x")
            ax_function.set_ylabel("y")
            ax_function.grid(True, alpha=0.25)
            ax_function.text(
                0.5,
                0.5,
                "Generate data to begin",
                ha="center",
                va="center",
                transform=ax_function.transAxes,
            )
            self._draw_empty_residual_plot(ax_residual)
        else:
            ax_function.plot(
                self.dataset.x_plot,
                self.dataset.y_plot,
                color="#1f77b4",
                linewidth=2.0,
                label="True Function",
            )
            ax_function.scatter(
                self.dataset.x_train_np,
                self.dataset.y_train_np,
                color="#222222",
                s=20,
                alpha=0.68,
                label="Training Data",
            )

            active_snapshot = snapshot or self.current_snapshot
            if active_snapshot is not None:
                residuals = active_snapshot.predictions - self.dataset.y_plot
                ax_function.plot(
                    self.dataset.x_plot,
                    active_snapshot.predictions,
                    color="#d62728",
                    linewidth=2.0,
                    label="Model Prediction",
                )
                title = (
                    "Neural Network Function Approximation "
                    f"(epoch {active_snapshot.epoch}, loss {format_loss(active_snapshot.train_loss)})"
                )
                self._draw_error_heatmap(
                    ax_function,
                    self.dataset.x_plot,
                    np.abs(residuals),
                )
                self._draw_residual_plot(
                    ax_residual,
                    self.dataset.x_plot,
                    residuals,
                    active_snapshot.epoch,
                )
            else:
                title = "Neural Network Function Approximation"
                self._draw_empty_residual_plot(ax_residual)

            if self.comparison_result is not None:
                ax_function.scatter(
                    [self.comparison_result["x"]],
                    [self.comparison_result["true"]],
                    color="#1f77b4",
                    edgecolors="#ffffff",
                    linewidths=1.0,
                    marker="o",
                    s=90,
                    zorder=5,
                    label="Compare True",
                )
                ax_function.scatter(
                    [self.comparison_result["x"]],
                    [self.comparison_result["model"]],
                    color="#d62728",
                    edgecolors="#ffffff",
                    linewidths=1.0,
                    marker="X",
                    s=90,
                    zorder=6,
                    label="Compare Model",
                )
                y_low = min(self.comparison_result["true"], self.comparison_result["model"])
                y_high = max(self.comparison_result["true"], self.comparison_result["model"])
                ax_function.vlines(
                    self.comparison_result["x"],
                    y_low,
                    y_high,
                    colors="#ff7f0e",
                    linestyles="--",
                    linewidth=1.4,
                    label="Compare Error",
                )

            ax_function.set_title(title)
            ax_function.set_xlabel("x")
            ax_function.set_ylabel("y")
            ax_function.grid(True, alpha=0.25)
            ax_function.legend(loc="best")

        ax_loss.set_title("Loss vs Epoch")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.grid(True, alpha=0.25)

        if self.training_result is not None and self.training_result.epochs:
            ax_loss.plot(
                self.training_result.epochs,
                self.training_result.loss_history,
                color="#2ca02c",
                linewidth=1.8,
                label="Train Loss",
            )
            test_points = [
                (epoch, loss)
                for epoch, loss in zip(
                    self.training_result.epochs,
                    self.training_result.test_loss_history,
                    strict=False,
                )
                if loss is not None
            ]
            if test_points:
                test_epochs, test_losses = zip(*test_points, strict=False)
                ax_loss.plot(
                    test_epochs,
                    test_losses,
                    color="#9467bd",
                    linewidth=1.6,
                    label="Test Loss",
                )
            active_snapshot = snapshot or self.current_snapshot
            if active_snapshot is not None and active_snapshot.epoch > 0:
                ax_loss.axvline(
                    active_snapshot.epoch,
                    color="#777777",
                    linestyle="--",
                    linewidth=1.0,
                )
            ax_loss.legend(loc="best")

        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_error_heatmap(
        self,
        ax_function,
        x_values: np.ndarray,
        absolute_errors: np.ndarray,
    ) -> None:
        finite_mask = np.isfinite(x_values) & np.isfinite(absolute_errors)
        if finite_mask.sum() < 2:
            return

        x_finite = x_values[finite_mask]
        errors = absolute_errors[finite_mask]
        max_error = float(np.max(errors))
        if max_error <= 0:
            normalized_errors = np.zeros_like(errors)
        else:
            normalized_errors = errors / max_error

        y_min, y_max = ax_function.get_ylim()
        y_span = y_max - y_min
        if y_span <= 0:
            return

        band_bottom = y_min
        band_top = y_min + 0.08 * y_span
        ax_function.imshow(
            normalized_errors.reshape(1, -1),
            aspect="auto",
            cmap="RdYlGn_r",
            interpolation="nearest",
            alpha=0.45,
            extent=[float(np.min(x_finite)), float(np.max(x_finite)), band_bottom, band_top],
            zorder=0,
        )
        ax_function.text(
            float(np.min(x_finite)),
            band_top,
            "Error heatmap",
            fontsize=8,
            color="#444444",
            va="bottom",
        )
        ax_function.set_ylim(y_min, y_max)

    def _draw_empty_residual_plot(self, ax_residual) -> None:
        ax_residual.set_title("Residuals: Model Prediction - True Function")
        ax_residual.set_xlabel("x")
        ax_residual.set_ylabel("Residual")
        ax_residual.grid(True, alpha=0.25)
        ax_residual.axhline(0.0, color="#777777", linewidth=1.0)
        self._configure_residual_x_axis(ax_residual)
        ax_residual.text(
            0.5,
            0.5,
            "Train a model to see residuals",
            ha="center",
            va="center",
            transform=ax_residual.transAxes,
        )

    def _draw_residual_plot(
        self,
        ax_residual,
        x_values: np.ndarray,
        residuals: np.ndarray,
        epoch: int,
    ) -> None:
        finite_mask = np.isfinite(x_values) & np.isfinite(residuals)
        if finite_mask.sum() < 2:
            self._draw_empty_residual_plot(ax_residual)
            return

        x_finite = x_values[finite_mask]
        residuals_finite = residuals[finite_mask]
        absolute_errors = np.abs(residuals_finite)
        max_abs_error = float(np.max(absolute_errors))
        mean_abs_error = float(np.mean(absolute_errors))

        ax_residual.axhline(0.0, color="#777777", linewidth=1.0)
        ax_residual.plot(
            x_finite,
            residuals_finite,
            color="#ff7f0e",
            linewidth=1.4,
            label="Residual",
        )
        ax_residual.fill_between(
            x_finite,
            0.0,
            residuals_finite,
            color="#ff7f0e",
            alpha=0.18,
        )
        ax_residual.set_title(
            "Residuals: Model Prediction - True Function "
            f"(epoch {epoch}, MAE {self._format_number(mean_abs_error)}, "
            f"max |error| {self._format_number(max_abs_error)})"
        )
        ax_residual.set_xlabel("x")
        ax_residual.set_ylabel("Residual")
        ax_residual.grid(True, alpha=0.25)
        self._configure_residual_x_axis(ax_residual)
        ax_residual.legend(loc="best")

    def _configure_residual_x_axis(self, ax_residual) -> None:
        ax_residual.xaxis.set_major_locator(MultipleLocator(0.5))
        ax_residual.xaxis.set_major_formatter(FuncFormatter(self._format_half_step_tick))
        ax_residual.tick_params(axis="x", labelrotation=45, labelsize=8)

    def _format_half_step_tick(self, value: float, _position: int) -> str:
        rounded = round(value * 2) / 2
        if abs(rounded) < 1e-12:
            rounded = 0.0
        if float(rounded).is_integer():
            return str(int(rounded))
        return f"{rounded:.1f}"

    def _set_training_ui_enabled(self, enabled: bool) -> None:
        self.generate_button.setEnabled(enabled)
        self.apply_preset_button.setEnabled(enabled)
        self.train_button.setEnabled(enabled)
        self.train_button.setText("Train Model" if enabled else "Training...")
        self.save_model_button.setEnabled(enabled and self.training_result is not None)
        self.compare_button.setEnabled(enabled and self.training_result is not None)

    def _training_running(self) -> bool:
        return self.training_thread is not None and self.training_thread.isRunning()

    def export_model(self) -> None:
        if self.training_result is None:
            self._show_error("No Model", "Train a model before exporting it.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trained Model",
            "ml_function_model.pt",
            "PyTorch Model (*.pt);;All Files (*)",
        )
        if not path:
            return

        payload = {
            "state_dict": self.training_result.model.state_dict(),
            "model_config": self.model_config,
            "normalization": (
                self.training_result.normalization_stats.to_dict()
                if isinstance(self.training_result.normalization_stats, NormalizationStats)
                else None
            ),
            "settings": self._collect_settings(),
            "loss_history": self.training_result.loss_history,
            "test_loss_history": self.training_result.test_loss_history,
        }
        try:
            torch.save(payload, path)
            self.status_label.setText(f"Model exported to {Path(path).name}.")
        except Exception as exc:
            self._show_error("Export Error", str(exc))

    def export_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Settings",
            "ml_visualizer_settings.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_json(path, self._collect_settings())
            self.status_label.setText(f"Settings exported to {Path(path).name}.")
        except Exception as exc:
            self._show_error("Export Error", str(exc))

    def import_settings(self) -> None:
        if self._training_running():
            self._show_error("Training Running", "Wait for training to finish before importing settings.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            self._apply_settings(load_json(path))
            self.status_label.setText(f"Settings imported from {Path(path).name}.")
        except Exception as exc:
            self._show_error("Import Error", str(exc))

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._training_running() and self.training_worker is not None:
            self.training_worker.stop()
            self.training_thread.quit()
            self.training_thread.wait(2000)
        super().closeEvent(event)
