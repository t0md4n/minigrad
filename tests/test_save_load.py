"""
Test save/load functionality for Layer and MLP classes.
"""
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minigrad import Layer, MLP, Tensor


def test_layer_save_load():
    """Test that Layer can be saved and loaded correctly."""
    print("Testing Layer save/load...")

    # Create a layer with known weights
    layer = Layer(3, 5, activation='relu')

    # Set specific weights for testing
    layer.W.data = np.array([[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15]], dtype=np.float64)
    layer.b.data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    # Test forward pass before saving
    x = Tensor([[1.0, 2.0, 3.0]])
    y_original = layer(x)

    # Save the layer
    layer.save('test_layer.npz')

    # Load the layer
    loaded_layer = Layer.load('test_layer.npz')

    # Test that architecture is preserved
    assert loaded_layer.W.shape == layer.W.shape, "Weight shapes don't match"
    assert loaded_layer.b.shape == layer.b.shape, "Bias shapes don't match"
    assert loaded_layer.activation == layer.activation, "Activation doesn't match"

    # Test that weights are identical
    assert np.allclose(loaded_layer.W.data, layer.W.data), "Weights don't match"
    assert np.allclose(loaded_layer.b.data, layer.b.data), "Biases don't match"

    # Test that forward pass gives same results
    y_loaded = loaded_layer(x)
    assert np.allclose(y_loaded.data, y_original.data), "Forward pass outputs don't match"

    # Clean up
    os.remove('test_layer.npz')

    print("✓ Layer save/load test passed")


def test_mlp_save_load():
    """Test that MLP can be saved and loaded correctly."""
    print("Testing MLP save/load...")

    # Create an MLP
    model = MLP(2, [8, 8, 1], activation='relu', output_activation='sigmoid')

    # Create test input
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    # Get predictions before saving
    y_original = model(x)

    # Save the model
    model.save('test_model.npz')

    # Load the model
    loaded_model = MLP.load('test_model.npz')

    # Test architecture is preserved
    assert len(loaded_model.layers) == len(model.layers), "Number of layers doesn't match"
    for i, (orig_layer, loaded_layer) in enumerate(zip(model.layers, loaded_model.layers)):
        assert orig_layer.W.shape == loaded_layer.W.shape, f"Layer {i} weight shape doesn't match"
        assert orig_layer.b.shape == loaded_layer.b.shape, f"Layer {i} bias shape doesn't match"
        assert orig_layer.activation == loaded_layer.activation, f"Layer {i} activation doesn't match"

    # Test weights are identical
    for i, (orig_layer, loaded_layer) in enumerate(zip(model.layers, loaded_model.layers)):
        assert np.allclose(orig_layer.W.data, loaded_layer.W.data), f"Layer {i} weights don't match"
        assert np.allclose(orig_layer.b.data, loaded_layer.b.data), f"Layer {i} biases don't match"

    # Test forward pass gives same results
    y_loaded = loaded_model(x)
    assert np.allclose(y_loaded.data, y_original.data), "Forward pass outputs don't match"

    # Clean up
    os.remove('test_model.npz')

    print("✓ MLP save/load test passed")


def test_save_load_after_training():
    """Test that a trained model can be saved and loaded correctly."""
    print("Testing save/load after training...")

    # Create simple dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # Create and train a model
    model = MLP(2, [4, 4, 1], activation='tanh')
    learning_rate = 0.1

    # Train for a few steps
    for step in range(50):
        X_tensor = Tensor(X)
        y_tensor = Tensor(y)

        ypred = model(X_tensor)
        loss = ((ypred - y_tensor) ** 2).mean()

        # Zero gradients
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)

        # Backward
        loss.backward()

        # Update
        for p in model.parameters():
            p.data += -learning_rate * p.grad

    # Get final predictions
    final_pred = model(Tensor(X))

    # Save the trained model
    model.save('test_trained_model.npz')

    # Load the model
    loaded_model = MLP.load('test_trained_model.npz')

    # Test that loaded model gives same predictions
    loaded_pred = loaded_model(Tensor(X))
    assert np.allclose(loaded_pred.data, final_pred.data), "Trained model predictions don't match"

    # Test that we can continue training the loaded model
    X_tensor = Tensor(X)
    y_tensor = Tensor(y)
    ypred = loaded_model(X_tensor)
    loss = ((ypred - y_tensor) ** 2).mean()

    for p in loaded_model.parameters():
        p.grad = np.zeros_like(p.data)

    loss.backward()

    # Verify gradients were computed
    for p in loaded_model.parameters():
        assert not np.allclose(p.grad, 0), "Gradients should be non-zero after backward pass"

    # Clean up
    os.remove('test_trained_model.npz')

    print("✓ Save/load after training test passed")


def test_path_handling():
    """Test that path handling works correctly (with and without .npz extension)."""
    print("Testing path handling...")

    model = MLP(2, [4, 1])

    # Test saving without .npz extension
    model.save('test_no_ext')
    assert os.path.exists('test_no_ext.npz'), "File should have .npz extension added"

    # Test loading without .npz extension
    loaded = MLP.load('test_no_ext')
    assert loaded is not None, "Should be able to load without .npz extension"

    os.remove('test_no_ext.npz')

    # Test saving with .npz extension
    model.save('test_with_ext.npz')
    assert os.path.exists('test_with_ext.npz'), "File should exist"

    # Test loading with .npz extension
    loaded = MLP.load('test_with_ext.npz')
    assert loaded is not None, "Should be able to load with .npz extension"

    os.remove('test_with_ext.npz')

    print("✓ Path handling test passed")


if __name__ == '__main__':
    print("\n=== Running Save/Load Tests ===\n")

    test_layer_save_load()
    test_mlp_save_load()
    test_save_load_after_training()
    test_path_handling()

    print("\n=== All Save/Load Tests Passed! ===\n")
