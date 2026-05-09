# ML Function Visualizer

ML Function Visualizer is a Python desktop application for learning how neural networks approximate mathematical functions. A user enters a function of `x`, generates synthetic training data, trains a PyTorch neural network, and watches the prediction curve improve over training epochs.

The project is designed as an educational machine-learning visualizer. It does not hide the training process behind a single final number. Instead, it shows the true function, sampled training points, neural-network predictions, loss curves, residual errors, an error heatmap, and point-by-point comparison between the exact function and the trained model.

## What The App Demonstrates

This application is useful for exploring questions like:

- How does a neural network learn a smooth function such as `sin(x)`?
- Why do some functions need more neurons or more layers?
- What does underfitting look like?
- What does overfitting look like when data contains noise?
- Why is extrapolation outside the training range risky?
- How does normalization change training stability?
- Where does the model make its largest errors?
- How do training loss and test loss differ?

The app is intentionally visual. It is meant for students, teachers, hobbyists, and developers who want to inspect neural-network approximation behavior without writing training scripts manually.

## Features

- Safe mathematical function parsing with SymPy
- Natural input syntax such as `sin(x)`, `x^2`, `exp(-x^2) * cos(3*x)`, and `sqrt(abs(x))`
- Synthetic dataset generation with optional Gaussian noise
- Automatic removal of invalid, NaN, and infinite data points
- Train/test split using scikit-learn
- Dynamic PyTorch model creation with configurable depth, width, activation, optimizer, learning rate, epochs, and batch size
- Background training with PyQt6 `QThread`, keeping the GUI responsive
- Embedded Matplotlib plots inside a PyQt6 desktop interface
- Animation of prediction snapshots across epochs
- Loss curve for train and test loss
- Residual plot showing `model prediction - true function`
- Error heatmap showing where the model is most wrong across the x-axis
- Point-wise Compare tool for checking exact `f(x)` against model output
- Educational presets for common ML behaviors
- Optional x/y normalization during training
- CPU/CUDA device selection when CUDA is available
- Random seed control for reproducible experiments
- Import/export settings as JSON
- Export trained model as a PyTorch `.pt` file

## Screenshots

Add screenshots here before publishing the repository publicly. Recommended screenshots:

- Main application after generating data
- Training animation in progress
- Residual plot and error heatmap after training
- Compare panel with a selected x-value
- Preset dropdown showing the educational demos

Example Markdown:

```md
![Main window](docs/screenshots/main-window.png)
![Residual diagnostics](docs/screenshots/residuals.png)
```

## Project Structure

```text
ml-function-visualizer/
|-- main.py
|-- requirements.txt
|-- README.md
|
|-- app/
|   |-- __init__.py
|   |-- gui.py
|   |-- parser.py
|   |-- dataset.py
|   |-- model.py
|   |-- trainer.py
|   |-- animator.py
|   `-- utils.py
|
`-- examples/
    `-- example_functions.txt
```

### File Responsibilities

`main.py`

Application entry point. Creates the PyQt6 application and opens the main window.

`app/gui.py`

Main PyQt6 interface. It owns the controls, Matplotlib canvas, training thread, presets, Compare panel, plotting logic, settings import/export, and model export.

`app/parser.py`

Safe SymPy-based parser. It accepts a restricted mathematical language and converts valid expressions into NumPy-compatible functions using `sympy.lambdify`.

`app/dataset.py`

Dataset generation utilities. It samples x-values, evaluates the parsed function, adds optional noise, removes invalid values, creates train/test splits, converts data into PyTorch tensors, and stores normalization statistics.

`app/model.py`

Dynamic PyTorch neural-network model. It builds an `nn.Sequential` network from user-selected layer count, neuron count, and activation function.

`app/trainer.py`

Training engine. It handles the optimizer, loss function, batching, train/test loss tracking, and prediction snapshot storage.

`app/animator.py`

Timer-based animation controller for replaying stored prediction snapshots inside the PyQt6 Matplotlib canvas.

`app/utils.py`

Small shared helpers for random seeds, device resolution, loss formatting, and JSON loading/saving.

`examples/example_functions.txt`

List of example functions that users can try.

## Installation

### 1. Clone The Repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-function-visualizer.git
cd ml-function-visualizer
```

### 2. Create A Virtual Environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses:

- PyTorch
- NumPy
- SymPy
- Matplotlib
- PyQt6
- scikit-learn

### PyTorch Note

PyTorch installation can vary depending on whether you want CPU-only or CUDA support. The plain requirement:

```text
torch
```

works for many users, but CUDA users may prefer the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

## Running The App

From the project root:

```bash
python main.py
```

The main window opens with controls on the left and plots on the right.

## Basic Workflow

1. Enter a mathematical function of `x`.
2. Choose an x-range.
3. Choose the number of training points.
4. Optionally add Gaussian noise.
5. Click **Generate Data**.
6. Choose neural-network settings.
7. Click **Train Model**.
8. Use **Play Animation** to replay prediction snapshots.
9. Use **Compare** to inspect a specific x-value.

## Example Functions

Simple functions:

```text
sin(x)
cos(2*x)
x^2
x^3 - 2*x
```

Smooth nonlinear functions:

```text
exp(-x^2)
exp(-x^2) * cos(3*x)
1 / (1 + x^2)
```

Noisy or trend-like functions:

```text
sin(x) + 0.2*x
sin(5*x) + 0.5*x
```

Domain-sensitive functions:

```text
log(x + 6)
sqrt(abs(x))
```

High-frequency functions:

```text
sin(8*x)
sin(5*x) * exp(-0.1*x^2)
```

## Supported Function Syntax

The parser supports:

- `sin`
- `cos`
- `tan`
- `exp`
- `log`
- `sqrt`
- `abs`
- `pi`
- `e`
- `x`

The app accepts `^` for exponentiation, so these are equivalent:

```text
x^2
x**2
```

Examples:

```text
sin(x) + x^2
exp(-x^2)
sqrt(abs(x))
log(x + 6)
cos(3*x)
```

## Parser Safety

The app does not use unrestricted Python `eval`.

User input is parsed with SymPy using a restricted dictionary of allowed symbols and functions. Only the supported mathematical names are accepted. Unknown variables, unsupported functions, malformed expressions, and expressions that immediately evaluate to undefined or infinite values are rejected with a GUI error message.

This keeps the function input focused on mathematical expressions rather than arbitrary Python execution.

## Data Generation

When the user clicks **Generate Data**, the app:

1. Parses the function.
2. Samples x-values with `numpy.linspace`.
3. Evaluates the function.
4. Adds optional Gaussian noise.
5. Removes NaN and infinite values.
6. Splits the remaining data into training and test sets.
7. Converts train/test arrays into PyTorch tensors.
8. Creates a dense plotting grid for the true function curve.

If too few valid points remain, the app shows an error. This can happen when the x-range includes invalid regions, such as:

```text
log(x)
```

with negative x-values.

## Neural Network Model

The model is a fully connected feed-forward network:

```text
input x -> hidden layers -> output y
```

The architecture is created dynamically from the GUI settings.

Supported activations:

- ReLU
- Tanh
- Sigmoid
- GELU
- LeakyReLU

The final layer is linear, which is appropriate for regression.

## Training

The trainer uses:

- `torch.nn.MSELoss`
- mini-batches with `DataLoader`
- Adam, SGD, or RMSprop optimizer
- train loss tracking
- test loss tracking
- prediction snapshots every N epochs

Training runs in a background `QThread`. This prevents the GUI from freezing while the model trains.

## Educational Presets

The preset dropdown fills the GUI with ready-made experiments. After selecting a preset, click **Apply Preset**, then **Generate Data**, then **Train Model**.

### Good Fit Demo

Function:

```text
sin(x)
```

Purpose:

Shows a typical successful approximation. This is the best first demo because the function is smooth, bounded, and easy for a small network to learn.

What to observe:

- prediction curve quickly approaches the true curve
- loss decreases smoothly
- residuals become small across most of the range

### Underfitting Demo

Function:

```text
sin(5*x)
```

Purpose:

Uses a network that is intentionally too small. This shows what happens when the model does not have enough capacity.

What to observe:

- prediction may smooth over oscillations
- residual plot remains structured instead of random
- error heatmap highlights repeated difficult regions

### Overfitting Demo

Function:

```text
sin(x) + 0.2*x
```

with noisy training data.

Purpose:

Uses a larger network and noisy data to demonstrate how a model can start fitting noise instead of only the underlying trend.

What to observe:

- train loss can become very low
- test loss may not improve as much
- prediction may become unnecessarily wavy

### Noisy Data Demo

Function:

```text
exp(-x^2) * cos(3*x)
```

Purpose:

Shows the difference between learning an underlying function and fitting noisy samples.

What to observe:

- training points do not lie exactly on the true curve
- prediction should ideally follow the smooth true function rather than every noisy point

### Extrapolation Demo

Function:

```text
x^2
```

Purpose:

Shows that neural networks are usually much more reliable inside the training range than outside it.

What to try:

Train on `[-2, 2]`, then use Compare at x-values such as:

```text
2.5
3
4
```

The Compare panel will warn when the selected point is outside the training range.

### High Frequency Demo

Function:

```text
sin(8*x)
```

Purpose:

Shows that rapidly oscillating functions are harder than low-frequency functions.

What to observe:

- more epochs and more neurons may be needed
- error heatmap often highlights peaks, troughs, or phase-shifted regions

### Large Scale Normalization Demo

Function:

```text
x^2 + 3*x - 1
```

on a large x-range.

Purpose:

Demonstrates why normalization is useful. Without normalization, the model sees large input and output values, which can slow or destabilize training.

What to try:

Train once with normalization enabled, then train again with it disabled and compare convergence.

## Normalization

The **Normalize x/y during training** checkbox scales x and y values before training:

```text
x_normalized = (x - mean_x) / std_x
y_normalized = (y - mean_y) / std_y
```

The statistics are computed from the training split.

The model trains in normalized coordinates, but the app converts results back to the original function scale for:

- prediction snapshots
- function plot
- residual plot
- error heatmap
- train/test loss display
- Compare output
- exported model metadata

This means the user sees normal function values while the neural network receives numerically easier data.

### Why Normalization Helps

Neural networks often train better when inputs and targets are centered near zero and have similar scale.

For example:

```text
x^2 on [-100, 100]
```

produces y-values up to 10,000. Without normalization, gradients can be poorly scaled and training may require more careful learning-rate tuning.

Normalization is especially helpful for:

- large x-ranges
- polynomial functions with large output values
- functions with steep slopes
- mixed-scale expressions such as `100*sin(x) + x^2`

## Animation

The app stores prediction snapshots every N epochs, controlled by **Snapshot interval**.

During animation:

- the true function stays fixed
- training points stay fixed
- the prediction curve updates
- residuals update
- the error heatmap updates
- loss plot shows the current epoch with a vertical marker

Controls:

- **Play Animation** starts playback
- **Pause** stops playback
- **Step Forward** moves one snapshot forward
- **Step Back** moves one snapshot backward
- **Speed** changes playback speed

Lower snapshot intervals create smoother animations but use more memory.

## Visual Diagnostics

### True Function Plot

The top plot shows:

- true function curve
- training data points
- model prediction curve
- optional Compare markers
- error heatmap band

### Error Heatmap

The colored band at the bottom of the top plot shows absolute prediction error across the x-axis.

Interpretation:

- green: lower error
- yellow: moderate error
- red: higher error

The heatmap is normalized relative to the largest visible error for the current snapshot. This makes it useful for seeing where the model is struggling, even when the raw error values are small.

### Residual Plot

The middle plot shows:

```text
model prediction - true function
```

Interpretation:

- residual above zero: model overestimates
- residual below zero: model underestimates
- residual near zero: good approximation

The residual title reports:

- MAE: mean absolute error across the plotted curve
- max absolute error: largest visible absolute error

The residual x-axis uses 0.5 spacing so users can inspect local error more precisely.

### Loss Plot

The bottom plot shows train and test MSE loss over epochs.

Interpretation:

- both losses decrease: training is improving
- train loss decreases but test loss rises: possible overfitting
- both losses stay high: possible underfitting or bad hyperparameters
- loss jumps or becomes unstable: learning rate may be too high

## Compare Tool

The Compare panel evaluates one exact x-value after training.

It reports:

- true `f(x)`
- model output
- signed error
- absolute error
- relative error
- short interpretation

The signed error is:

```text
model output - true f(x)
```

So:

- positive signed error means the model overestimates
- negative signed error means the model underestimates

The Compare tool also checks whether the x-value is inside or outside the training range. Points outside the range are extrapolation points, so errors are expected to be less reliable.

## Settings Import And Export

The app can export the current GUI settings to JSON. This is useful for saving experiments and sharing reproducible configurations.

Settings include:

- function
- x-range
- number of points
- noise level
- model architecture
- optimizer
- learning rate
- epoch count
- batch size
- snapshot interval
- seed
- device
- normalization setting

Settings can later be imported from the Utilities panel.

## Model Export

The trained model can be exported as a PyTorch `.pt` file.

The export includes:

- model `state_dict`
- model configuration
- normalization statistics, if normalization was enabled
- GUI settings
- train loss history
- test loss history

Important: if normalization was enabled, the raw exported model expects normalized x-inputs and returns normalized y-values. The export includes the normalization statistics needed to reproduce the same input/output conversion.

## Hyperparameter Guide

### Hidden Layers

More hidden layers increase representational power. Smooth simple functions often work with 1-2 hidden layers. High-frequency or more complex functions may need 3 or more.

### Neurons Per Layer

More neurons make each hidden layer wider. Too few neurons can underfit. Too many can train slower and may overfit noisy data.

### Activation

`Tanh` is often strong for smooth mathematical functions because it is smooth and bounded.

`ReLU` can work well, but it produces piecewise-linear approximations.

`GELU` is smooth and often a good modern default.

`Sigmoid` can train slowly because it saturates.

`LeakyReLU` can behave better than ReLU in some cases.

### Learning Rate

The learning rate controls optimizer step size.

Typical starting point:

```text
0.001
```

If training is unstable, reduce it.

If training is too slow, increase it carefully.

### Epochs

More epochs give the model more chances to improve. But too many epochs can waste time or overfit noisy data.

### Batch Size

Small batches add noise to optimization. Large batches are smoother but may generalize differently.

For this app, these are good starting values:

```text
16
32
64
```

### Snapshot Interval

Controls how often animation frames are saved.

Example:

```text
epochs = 1000
snapshot interval = 25
```

stores about 40 animation snapshots.

Smaller intervals create smoother animation but use more memory.

## Troubleshooting

### The Function Does Not Parse

Check that the expression only uses supported functions and the variable `x`.

Valid:

```text
sin(x) + x^2
```

Invalid:

```text
np.sin(x)
y + x
```

### The Function Produces Too Few Points

The x-range may include invalid values.

Example:

```text
log(x)
```

is invalid for `x <= 0`.

Fix it by changing the range or shifting the function:

```text
log(x + 6)
```

### Training Loss Is NaN Or Infinite

Try:

- lower learning rate
- enable normalization
- reduce x-range
- choose Adam
- avoid functions with extreme values

### The Model Does Not Fit Well

Try:

- more epochs
- more neurons
- more hidden layers
- Tanh or GELU activation
- lower learning rate
- normalization
- more training points

### The Model Fits Training Data But Test Loss Is Worse

This may be overfitting.

Try:

- fewer layers or neurons
- fewer epochs
- less noise
- more data
- compare train and test loss curves

### Compare Looks Bad Outside The Training Range

That is expected. Neural networks generally interpolate better than they extrapolate.

Use Compare values inside the selected x-range to judge learned approximation quality.

## Development Notes

The codebase is intentionally modular:

- GUI code stays in `app/gui.py`
- function parsing stays in `app/parser.py`
- dataset creation stays in `app/dataset.py`
- model construction stays in `app/model.py`
- training logic stays in `app/trainer.py`
- animation state stays in `app/animator.py`

This makes the project easier to extend without turning the GUI file into the entire application.

## Suggested Next Improvements

High-impact additions:

- Click-to-compare directly on the plot
- Train/test point visibility toggles
- CSV export of predictions and residuals
- Parameter count display
- Training time estimate
- Derivative comparison using SymPy and PyTorch autograd
- Multiple model comparison
- Save/load complete experiment sessions
- Export animation as GIF or MP4
- Add automated tests for parser, dataset generation, and normalization

## Packaging Ideas

For a public release, consider adding:

- `LICENSE`
- `CHANGELOG.md`
- `pyproject.toml`
- pinned dependency versions
- screenshots under `docs/screenshots/`
- GitHub Actions workflow for syntax checks
- PyInstaller or Nuitka build instructions

Example PyInstaller command:

```bash
pyinstaller --noconfirm --windowed --name "ML Function Visualizer" main.py
```

PyQt6 and Matplotlib applications sometimes need extra packaging configuration, so test the generated executable on a clean machine before publishing a release.

## License

Choose a license before publishing. MIT is a common choice for educational tools, but use whichever license fits your goal.
