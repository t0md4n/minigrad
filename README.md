# minigrad

A neural network library built from scratch with automatic differentiation, now featuring **tensor operations** for efficient training.

## Features

- **Tensor autograd engine** - Automatic differentiation with NumPy-backed tensors
- **Efficient matrix operations** - Batch processing with vectorized operations
- **Neural networks** - Layer and MLP (multi-layer perceptron) implementations
- **Flexible activations** - tanh, ReLU, sigmoid, softmax support
- **Model persistence** - Save and load trained models
- **Synthetic datasets** - Generators for binary classification and regression tasks
- **Broadcasting support** - Automatic shape handling for tensor operations

Originally inspired by [micrograd](https://github.com/karpathy/micrograd), this library has evolved from scalar-valued autograd to a full tensor-based framework similar to PyTorch's design.

## Installation

```bash
# Create conda environment with dependencies
conda create -n minigrad python=3.11 numpy -y
conda activate minigrad
```

Or install NumPy directly:
```bash
pip install numpy
```

## Quick Start

```bash
# Train on XOR problem
python examples/train_xor.py

# Binary classification (moons dataset)
python examples/train_classification.py

# Linear regression
python examples/train_regression.py

# Train with save/load demonstration
python examples/train_with_save.py

# Long training with checkpointing (5-class spirals)
python examples/train_long.py

# Run all tests
python tests/run_all_tests.py
```

## Example Usage

```python
import numpy as np
from minigrad import MLP, Tensor, make_moons

# Generate dataset (returns numpy arrays)
X, y = make_moons(n_samples=100, noise=0.1)

# Create model: 2 inputs -> 16 hidden -> 16 hidden -> 1 output
model = MLP(2, [16, 16, 1])

# Training loop with batch processing
learning_rate = 0.01
for step in range(200):
    # Forward pass (entire batch at once)
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

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.data:.4f}")
```

## Saving and Loading Models

Save trained models to disk and load them later for inference or continued training:

```python
# Train a model
model = MLP(2, [16, 16, 1])
# ... training code ...

# Save the trained model
model.save('my_model.npz')

# Load the model later
loaded_model = MLP.load('my_model.npz')

# Use for predictions
predictions = loaded_model(Tensor(X_test))

# Or continue training from checkpoint
for step in range(100):
    # ... continue training ...
    pass
```

**Features**:
- Saves all weights, biases, and architecture information
- Automatic `.npz` extension handling
- Preserves activation functions for each layer
- Models can be loaded and used immediately for inference
- Loaded models support continued training

See `examples/train_with_save.py` for a complete demonstration.

## Architecture

### Tensor Engine (`tensor.py`)

The `Tensor` class wraps NumPy arrays and implements automatic differentiation:

- **Shape tracking** - Handles multi-dimensional arrays (scalars are 0-d tensors)
- **Broadcasting** - Automatic shape alignment following NumPy rules
- **Gradient computation** - Reverse-mode autodiff with topological sort
- **Operations**:
  - Element-wise: `+`, `-`, `*`, `/`, `**`
  - Matrix multiplication: `@`
  - Reductions: `sum()`, `mean()`
  - Reshaping: `reshape()`, `transpose()`, `.T`
  - Activations: `tanh()`, `relu()`, `sigmoid()`, `softmax()`
  - Math: `exp()`, `log()`

### Neural Networks (`nn.py`)

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

### Datasets (`datasets.py`)

All dataset functions return NumPy arrays:

- `make_moons(n_samples, noise)` - Two interleaving half circles
- `make_circles(n_samples, noise, factor)` - Concentric circles
- `make_linear_regression(n_samples, n_features, noise, coef)` - Linear regression data
- `make_spirals(n_samples, n_classes, noise, rotations)` - Multi-class spiral classification (challenging non-linear boundaries)

## File Structure

```
minigrad/
├── minigrad/              # Core library package
│   ├── __init__.py        # Package exports
│   ├── tensor.py          # Tensor class with autograd
│   ├── nn.py              # Neural network layers (tensor-based)
│   └── datasets.py        # Dataset generators (numpy arrays)
├── examples/              # Training examples
│   ├── train_xor.py       # XOR problem
│   ├── train_classification.py  # Binary classification
│   ├── train_regression.py      # Linear regression
│   ├── train_with_save.py # Save/load demonstration
│   └── train_long.py      # Long training with checkpointing (5-class spirals)
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_tensor.py     # Comprehensive tensor tests
│   ├── test_save_load.py  # Save/load functionality tests
│   └── run_all_tests.py   # Master test runner
├── README.md              # This file
└── CLAUDE.md              # Project documentation for AI assistants
```

## How it Works

### Computational Graph

The `Tensor` class builds a dynamic computational graph during the forward pass. Each operation stores:
- Input tensors (parents in the graph)
- A backward function (closure capturing local gradients)
- The operation name (for debugging)

When you call `loss.backward()`:
1. Builds topological ordering of all tensors in the graph
2. Initializes output gradient to 1
3. Propagates gradients backward through the graph
4. Each tensor accumulates gradients from all paths

### Matrix Operations

Layers use efficient matrix multiplication instead of per-neuron loops:

```python
# Old scalar approach: O(nin * nout * batch) individual operations
# for each neuron:
#     for each input:
#         activate(sum(w * x) + b)

# New tensor approach: O(1) matrix multiply + O(1) activation
# activate(X @ W + b)  # Single batched operation
```

### Broadcasting & Gradients

When shapes don't match (e.g., adding bias to a matrix), NumPy broadcasting rules apply. During backpropagation, gradients are automatically summed along broadcasted dimensions to match parameter shapes.

## Performance

**Speed**: Fast training through:
- Vectorized NumPy operations (C-level performance)
- Batch processing (process all samples at once)
- Efficient BLAS routines for matrix multiplication

**Example**: Training on 40 samples for 200 iterations completes in ~0.03 seconds.

## Limitations & Future Work

**Current limitations**:
- CPU-only (no GPU support)
- No built-in optimizers (Adam, RMSprop, etc.)
- No convolutional or recurrent layers
- No automatic mixed precision

**Potential extensions**:
- Optimizer classes (SGD, Adam, RMSprop)
- More layer types (Conv2D, LSTM, BatchNorm)
- Data loaders with shuffling
- Loss functions module
- Visualization tools
- Training utilities (early stopping, learning rate scheduling)

## Learning Resources

This library demonstrates key deep learning concepts:

1. **Automatic Differentiation** - How frameworks like PyTorch compute gradients
2. **Computational Graphs** - Dynamic graph construction via operator overloading
3. **Broadcasting** - Shape alignment and gradient propagation
4. **Backpropagation** - Chain rule applied via topological sort
5. **Matrix Calculus** - Gradients for linear algebra operations
6. **Neural Network Architecture** - Composing layers into models

## Contributing

This is an educational project. Feel free to:
- Add new activation functions
- Implement optimizers
- Create new layer types
- Add visualization tools
- Improve documentation

## License

MIT
