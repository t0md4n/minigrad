# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a neural network library built from scratch with automatic differentiation. It provides:
- **Tensor autograd engine** (`minigrad/tensor.py`): NumPy-backed tensors with automatic differentiation
- **Neural network layers** (`minigrad/nn.py`): Layer and MLP classes using efficient matrix operations
- **Synthetic datasets** (`minigrad/datasets.py`): Dataset generators returning NumPy arrays
- **Training examples** (`examples/`): Scripts demonstrating XOR, classification, regression, and checkpointing

Originally inspired by [micrograd](https://github.com/karpathy/micrograd), this library implements a full tensor-based autograd framework similar to PyTorch's design.

## Architecture

### Tensor Engine (`minigrad/tensor.py`)

The `Tensor` class wraps NumPy arrays and implements automatic differentiation:
- **Shape tracking**: Handles multi-dimensional arrays (scalars are 0-d tensors)
- **Broadcasting**: Automatic shape alignment following NumPy rules
- **Gradient computation**: Reverse-mode autodiff with topological sort
- **Operations**:
  - Element-wise: `+`, `-`, `*`, `/`, `**`
  - Matrix multiplication: `@`
  - Reductions: `sum()`, `mean()` with optional axis parameter
  - Reshaping: `reshape()`, `transpose()`, `.T` property
  - Activations: `tanh()`, `relu()`, `sigmoid()`, `softmax()`
  - Math: `exp()`, `log()`

Key implementation details:
- `_unbroadcast()`: Helper to reduce gradients along broadcasted dimensions
- `backward()`: Computes gradients via topological sort of computation graph
- Each operation stores a closure in `_backward` that computes local gradients

### Neural Networks (`minigrad/nn.py`)

**Layer** - Fully connected layer using efficient matrix operations:
```python
# Forward: y = activation(x @ W + b)
layer = Layer(nin=10, nout=20, activation='relu')
output = layer(input_tensor)  # Handles batches automatically
```

**MLP** - Multi-layer perceptron:
```python
model = MLP(nin=784, nouts=[128, 64, 10], activation='relu')
predictions = model(inputs)  # (batch_size, 784) -> (batch_size, 10)
```

All classes implement:
- `__call__(x)`: Forward pass with automatic batch handling
- `parameters()`: Returns list of all `Tensor` parameters for optimization
- `save(path)`: Save model weights and architecture to disk (MLP and Layer)
- `load(path)`: Class method to load saved models (MLP and Layer)

### Datasets (`minigrad/datasets.py`)

Four synthetic dataset generators (all return NumPy arrays):
- `make_moons(n_samples, noise)`: Two interleaving half circles
- `make_circles(n_samples, noise, factor)`: Concentric circles
- `make_linear_regression(n_samples, n_features, noise, coef)`: Linear regression data
- `make_spirals(n_samples, n_classes, noise, rotations)`: Multi-class spiral classification (challenging non-linear boundaries)

All return `(X, y)` as NumPy arrays.

### Training Loop Pattern

Standard training pattern used throughout examples:
```python
import numpy as np
from minigrad import MLP, Tensor, make_moons

X, y = make_moons(n_samples=100, noise=0.1)
model = MLP(2, [16, 16, 1])
learning_rate = 0.01

for step in range(200):
    # Forward pass (batch processing)
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.reshape(-1, 1))
    ypred = model(X_tensor)

    # Mean squared error loss
    loss = ((ypred - y_tensor) ** 2).mean()

    # Zero gradients
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)

    # Backward pass
    loss.backward()

    # Update parameters (SGD)
    for p in model.parameters():
        p.data += -learning_rate * p.grad
```

**Important**: Always zero gradients before `backward()` since gradients accumulate.

## File Structure

```
minigrad/
├── minigrad/              # Core library package
│   ├── __init__.py        # Exports: Tensor, Layer, MLP, dataset functions
│   ├── tensor.py          # Tensor class with autograd
│   ├── nn.py              # Neural network layers (tensor-based)
│   └── datasets.py        # Dataset generators (numpy arrays)
├── examples/              # Training examples
│   ├── train_xor.py       # XOR problem (4 samples)
│   ├── train_classification.py  # Binary classification (moons, 40 samples)
│   ├── train_regression.py      # Linear regression (50 samples)
│   ├── train_with_save.py # Save/load demonstration
│   └── train_long.py      # Long training with checkpointing (5-class spirals)
├── tests/                 # Test suite
│   ├── test_tensor.py     # Comprehensive tensor tests (10 test functions)
│   ├── test_save_load.py  # Save/load functionality tests
│   └── run_all_tests.py   # Master test runner (runs tests + training)
├── README.md              # User-facing documentation
└── CLAUDE.md              # This file (AI assistant guidance)
```

## Running the Code

**Dependencies**: NumPy is required for the tensor implementation.
```bash
conda create -n minigrad python=3.11 numpy -y
conda activate minigrad
```

**Training examples**:
```bash
python examples/train_xor.py
python examples/train_classification.py
python examples/train_regression.py
python examples/train_with_save.py  # Demonstrates save/load
python examples/train_long.py       # Long training with checkpointing
```

**Run all tests**:
```bash
python tests/run_all_tests.py
```

## Import Conventions

All core functionality is exported from the top-level `minigrad` package:
```python
from minigrad import Tensor, Layer, MLP, make_moons, make_circles, make_linear_regression, make_spirals
```

## Development Notes

### Key Implementation Details

1. **Broadcasting and Gradients**: When shapes don't match during operations, NumPy broadcasting applies. During backpropagation, `_unbroadcast()` sums gradients along broadcasted dimensions to match parameter shapes.

2. **Matrix Multiplication Gradients**:
   - Forward: `C = A @ B`
   - Backward: `dL/dA = dL/dC @ B.T`, `dL/dB = A.T @ dL/dC`

3. **Topological Sort**: `backward()` uses topological sort to ensure gradients flow in correct order through the computation graph.

4. **Activation Gradients**:
   - `tanh`: `(1 - tanh²(x)) * grad_output`
   - `relu`: `(x > 0) * grad_output`
   - `sigmoid`: `sigmoid(x) * (1 - sigmoid(x)) * grad_output`

5. **Xavier Initialization**: Weights initialized with `std = sqrt(2 / (nin + nout))` for stable training.

### Performance

- **Speed**: Vectorized NumPy operations provide C-level performance
- **Batch processing**: All layers handle batched inputs automatically
- **BLAS acceleration**: Matrix multiplication uses efficient BLAS routines

### Testing

The test suite (`tests/test_tensor.py`) covers:
- Basic operations and broadcasting
- Matrix multiplication
- Activation functions
- Reduction operations (sum, mean)
- Reshaping and transposing
- Gradient computation (including chain rule)
- Broadcasting gradients
- MLP forward and backward passes

Run with: `python tests/run_all_tests.py` (runs tests + all training examples)

## Common Patterns

### Creating and Training a Model

```python
from minigrad import MLP, Tensor, make_moons
import numpy as np

# Create dataset
X, y = make_moons(n_samples=40, noise=0.1)

# Initialize model
model = MLP(2, [8, 8, 1])  # 2 inputs -> 8 hidden -> 8 hidden -> 1 output

# Training loop
for step in range(200):
    # Wrap data in Tensors
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.reshape(-1, 1))

    # Forward
    ypred = model(X_tensor)
    loss = ((ypred - y_tensor) ** 2).mean()

    # Zero gradients
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)

    # Backward
    loss.backward()

    # Update
    for p in model.parameters():
        p.data += -0.05 * p.grad
```

### Accessing Tensor Data

```python
# Create tensor
x = Tensor([1.0, 2.0, 3.0])

# Access underlying NumPy array
print(x.data)  # numpy array [1. 2. 3.]

# Access gradient (after backward pass)
print(x.grad)  # numpy array [0. 0. 0.] initially

# Shape
print(x.shape)  # (3,)
```

### Working with Shapes

```python
# Scalar (0-d tensor)
x = Tensor(5.0)
print(x.shape)  # ()

# Vector (1-d tensor)
x = Tensor([1.0, 2.0, 3.0])
print(x.shape)  # (3,)

# Matrix (2-d tensor)
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)  # (2, 2)

# Reshape
x = x.reshape(4, 1)  # (2, 2) -> (4, 1)

# Transpose
x_T = x.T  # (4, 1) -> (1, 4)
```

### Saving and Loading Models

```python
# After training a model
model = MLP(2, [16, 16, 1])
# ... train model ...

# Save to disk
model.save('my_model.npz')  # .npz extension is optional

# Load later
loaded_model = MLP.load('my_model.npz')

# Use for inference
predictions = loaded_model(Tensor(X_test))

# Continue training from checkpoint
for step in range(100):
    # ... training loop ...
    if step % 50 == 0:
        loaded_model.save(f'checkpoint_{step}.npz')
```

**What gets saved**:
- All weight and bias tensors (as NumPy arrays)
- Model architecture (layer sizes)
- Activation functions for each layer
- Stored in NumPy's `.npz` format (compressed)

**Notes**:
- Saved models are fully portable (can be loaded on any system with NumPy)
- Gradients are NOT saved (only the learned parameters)
- Layer class also has save/load methods for individual layers
