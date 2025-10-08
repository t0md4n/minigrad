"""
Comprehensive tests for the Tensor class and autograd engine.
Tests operations, gradients, broadcasting, and edge cases.
"""

import numpy as np
from minigrad import Tensor, MLP

def test_basic_operations():
    """Test basic arithmetic operations."""
    print("Testing basic operations...")

    # Addition
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    assert np.allclose(c.data, [5.0, 7.0, 9.0]), "Addition failed"

    # Multiplication
    d = a * b
    assert np.allclose(d.data, [4.0, 10.0, 18.0]), "Multiplication failed"

    # Subtraction
    e = b - a
    assert np.allclose(e.data, [3.0, 3.0, 3.0]), "Subtraction failed"

    # Division
    f = b / a
    assert np.allclose(f.data, [4.0, 2.5, 2.0]), "Division failed"

    # Power
    g = a ** 2
    assert np.allclose(g.data, [1.0, 4.0, 9.0]), "Power failed"

    print("✓ Basic operations passed")


def test_broadcasting():
    """Test broadcasting behavior."""
    print("Testing broadcasting...")

    # Scalar + vector
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor(2.0)
    c = a + b
    assert np.allclose(c.data, [3.0, 4.0, 5.0]), "Scalar broadcast failed"

    # Matrix + vector (broadcast along last dimension)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([10.0, 20.0])
    z = x + y
    assert np.allclose(z.data, [[11.0, 22.0], [13.0, 24.0]]), "Matrix broadcast failed"

    print("✓ Broadcasting passed")


def test_matmul():
    """Test matrix multiplication."""
    print("Testing matrix multiplication...")

    # 2D @ 2D
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(C.data, expected), "2D matmul failed"

    # Matrix @ vector
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = Tensor([[1.0], [2.0], [3.0]])
    y = x @ v
    expected = np.array([[14.0], [32.0]])
    assert np.allclose(y.data, expected), "Matrix-vector matmul failed"

    print("✓ Matrix multiplication passed")


def test_activations():
    """Test activation functions."""
    print("Testing activations...")

    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Tanh
    y = x.tanh()
    expected = np.tanh([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.allclose(y.data, expected), "Tanh failed"

    # ReLU
    y = x.relu()
    expected = np.maximum(0, [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.allclose(y.data, expected), "ReLU failed"

    # Sigmoid
    y = x.sigmoid()
    expected = 1 / (1 + np.exp(-np.array([-2.0, -1.0, 0.0, 1.0, 2.0])))
    assert np.allclose(y.data, expected), "Sigmoid failed"

    # Softmax
    y = x.softmax()
    expected_exp = np.exp(x.data - np.max(x.data))
    expected = expected_exp / np.sum(expected_exp)
    assert np.allclose(y.data, expected), "Softmax failed"
    assert np.allclose(np.sum(y.data), 1.0), "Softmax doesn't sum to 1"

    print("✓ Activations passed")


def test_reductions():
    """Test reduction operations."""
    print("Testing reductions...")

    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Sum all
    y = x.sum()
    assert np.allclose(y.data, 21.0), "Sum all failed"

    # Sum axis 0
    y = x.sum(axis=0)
    assert np.allclose(y.data, [5.0, 7.0, 9.0]), "Sum axis 0 failed"

    # Sum axis 1
    y = x.sum(axis=1)
    assert np.allclose(y.data, [6.0, 15.0]), "Sum axis 1 failed"

    # Mean
    y = x.mean()
    assert np.allclose(y.data, 3.5), "Mean failed"

    # Mean axis 0
    y = x.mean(axis=0)
    assert np.allclose(y.data, [2.5, 3.5, 4.5]), "Mean axis 0 failed"

    print("✓ Reductions passed")


def test_reshaping():
    """Test reshaping and transposing."""
    print("Testing reshaping...")

    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Reshape
    y = x.reshape(3, 2)
    assert y.shape == (3, 2), "Reshape shape incorrect"
    assert np.allclose(y.data.flatten(), x.data.flatten()), "Reshape data incorrect"

    # Transpose
    y = x.T
    assert y.shape == (3, 2), "Transpose shape incorrect"
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert np.allclose(y.data, expected), "Transpose data incorrect"

    print("✓ Reshaping passed")


def test_gradients():
    """Test gradient computation."""
    print("Testing gradients...")

    # Test 1: Simple multiplication
    x = Tensor(3.0)
    y = Tensor(4.0)
    z = x * y
    z.backward()
    assert np.allclose(x.grad, 4.0), "Gradient of x incorrect"
    assert np.allclose(y.grad, 3.0), "Gradient of y incorrect"

    # Test 2: Chain of operations
    a = Tensor(2.0)
    b = a ** 2
    c = b + 1.0
    d = c * 3.0
    d.backward()
    # d = 3(a^2 + 1), dd/da = 6a = 12
    assert np.allclose(a.grad, 12.0), "Chain rule gradient incorrect"

    # Test 3: Matrix multiplication gradient
    W = Tensor([[1.0, 2.0], [3.0, 4.0]])
    x = Tensor([[1.0], [2.0]])
    y = W @ x
    loss = y.sum()
    loss.backward()
    # dL/dy = [[1], [1]] (since we sum)
    # dL/dW = dL/dy @ x.T = [[1], [1]] @ [[1, 2]] = [[1, 2], [1, 2]]
    expected_grad = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert np.allclose(W.grad, expected_grad), "Matmul gradient incorrect"

    print("✓ Gradients passed")


def test_broadcasting_gradients():
    """Test gradients with broadcasting."""
    print("Testing broadcasting gradients...")

    # Broadcast scalar to vector
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor(2.0)
    c = a + b
    loss = c.sum()
    loss.backward()

    assert np.allclose(a.grad, [1.0, 1.0, 1.0]), "Vector gradient incorrect"
    assert np.allclose(b.grad, 3.0), "Broadcasted scalar gradient should sum"

    # Broadcast vector to matrix
    W = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([10.0, 20.0])
    y = W + b  # Broadcast b to each row
    loss = y.sum()
    loss.backward()

    assert np.allclose(W.grad, np.ones_like(W.data)), "Matrix gradient incorrect"
    assert np.allclose(b.grad, [2.0, 2.0]), "Broadcast gradient should sum over rows"

    print("✓ Broadcasting gradients passed")


def test_mlp_forward():
    """Test MLP forward pass."""
    print("Testing MLP forward pass...")

    # Single sample
    model = MLP(3, [4, 2])
    x = Tensor([1.0, 2.0, 3.0])
    y = model(x)
    assert y.shape == (2,), f"MLP output shape incorrect: {y.shape}"

    # Batch
    X = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = model(X)
    assert y.shape == (2, 2), f"MLP batch output shape incorrect: {y.shape}"

    print("✓ MLP forward pass passed")


def test_mlp_gradient():
    """Test MLP gradient computation."""
    print("Testing MLP gradients...")

    model = MLP(2, [4, 1])
    X = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = Tensor([[1.0], [-1.0]])

    # Forward pass
    y_pred = model(X)
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)
    loss.backward()

    # Check all parameters have gradients
    for i, p in enumerate(model.parameters()):
        assert not np.allclose(p.grad, 0.0), f"Parameter {i} has zero gradient"

    print("✓ MLP gradients passed")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Running comprehensive tensor tests")
    print("=" * 60)

    test_basic_operations()
    test_broadcasting()
    test_matmul()
    test_activations()
    test_reductions()
    test_reshaping()
    test_gradients()
    test_broadcasting_gradients()
    test_mlp_forward()
    test_mlp_gradient()

    print("=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
