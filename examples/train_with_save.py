"""
Example: Training a model with save/load functionality

This example demonstrates:
1. Training a model on a binary classification task
2. Saving the trained model to disk
3. Loading the model back
4. Using the loaded model for predictions
5. Continuing training from a checkpoint
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minigrad import MLP, Tensor, make_moons


def train_model(model, X, y, epochs=100, learning_rate=0.05, verbose=True):
    """Train the model and return final loss."""
    for step in range(epochs):
        # Forward pass
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

        if verbose and (step % 20 == 0 or step == epochs - 1):
            print(f"  Step {step:3d}, Loss: {loss.data:.4f}")

    return loss.data


def compute_accuracy(model, X, y):
    """Compute classification accuracy."""
    X_tensor = Tensor(X)
    predictions = model(X_tensor).data
    predictions_binary = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions_binary.flatten() == y)
    return accuracy


def main():
    print("=" * 60)
    print("Training with Save/Load Example")
    print("=" * 60)

    # 1. Create dataset
    print("\n1. Creating dataset...")
    np.random.seed(42)
    X, y = make_moons(n_samples=100, noise=0.1)
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Create and train initial model
    print("\n2. Training initial model...")
    model = MLP(2, [16, 16, 1], activation='relu', output_activation='sigmoid')
    print(f"   Model architecture: 2 -> 16 -> 16 -> 1")
    print(f"   Parameters: {sum(p.data.size for p in model.parameters())}")

    initial_loss = train_model(model, X, y, epochs=100, learning_rate=0.05)
    initial_accuracy = compute_accuracy(model, X, y)
    print(f"\n   Final loss: {initial_loss:.4f}")
    print(f"   Accuracy: {initial_accuracy:.2%}")

    # 3. Save the trained model
    print("\n3. Saving model to 'trained_model.npz'...")
    model.save('trained_model.npz')
    print("   Model saved successfully!")

    # 4. Load the model back
    print("\n4. Loading model from 'trained_model.npz'...")
    loaded_model = MLP.load('trained_model.npz')
    print("   Model loaded successfully!")

    # 5. Verify loaded model gives same predictions
    print("\n5. Verifying loaded model...")
    loaded_accuracy = compute_accuracy(loaded_model, X, y)
    print(f"   Loaded model accuracy: {loaded_accuracy:.2%}")

    # Test on a few samples
    test_samples = X[:5]
    original_preds = model(Tensor(test_samples)).data
    loaded_preds = loaded_model(Tensor(test_samples)).data

    print("\n   Sample predictions comparison:")
    print("   Input          | Original | Loaded  | Match")
    print("   " + "-" * 48)
    for i in range(5):
        match = "✓" if np.allclose(original_preds[i], loaded_preds[i]) else "✗"
        print(f"   {test_samples[i]} | {original_preds[i][0]:.4f}   | {loaded_preds[i][0]:.4f}  | {match}")

    # 6. Continue training from checkpoint
    print("\n6. Continuing training from checkpoint...")
    print("   Training for 50 more epochs...")

    # Get loss before continuing
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.reshape(-1, 1))
    ypred_before = loaded_model(X_tensor)
    loss_before = ((ypred_before - y_tensor) ** 2).mean()
    print(f"   Loss before: {loss_before.data:.4f}")

    # Continue training
    final_loss = train_model(loaded_model, X, y, epochs=50, learning_rate=0.05, verbose=False)
    final_accuracy = compute_accuracy(loaded_model, X, y)

    print(f"   Loss after:  {final_loss:.4f}")
    print(f"   Final accuracy: {final_accuracy:.2%}")

    # 7. Save the updated model
    print("\n7. Saving updated model to 'trained_model_v2.npz'...")
    loaded_model.save('trained_model_v2.npz')
    print("   Updated model saved!")

    # 8. Demonstrate loading for inference only
    print("\n8. Loading model for inference (no training)...")
    inference_model = MLP.load('trained_model_v2.npz')
    inference_accuracy = compute_accuracy(inference_model, X, y)
    print(f"   Inference accuracy: {inference_accuracy:.2%}")

    # Clean up
    print("\n9. Cleaning up saved files...")
    os.remove('trained_model.npz')
    os.remove('trained_model_v2.npz')
    print("   Cleanup complete!")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("• Use model.save('path.npz') to save a trained model")
    print("• Use MLP.load('path.npz') to load a saved model")
    print("• Loaded models preserve weights and can be used for inference")
    print("• Loaded models can continue training from checkpoints")
    print("• Both .npz extension is optional (automatically added)")


if __name__ == '__main__':
    main()
