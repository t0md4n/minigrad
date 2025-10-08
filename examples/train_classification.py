import numpy as np
from minigrad import MLP, Tensor, make_moons

# Generate moons dataset
X, y = make_moons(n_samples=40, noise=0.1)

# Initialize model: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
model = MLP(2, [8, 8, 1])

print("Binary Classification on Moons Dataset")
print(f"Training samples: {X.shape[0]}")
print(f"Model parameters: {len(model.parameters())}")
print(f"Input shape: {X.shape}, Output shape: {y.shape}\n")

# Training loop
learning_rate = 0.05
for k in range(200):
    # Forward pass (batch processing - all samples at once)
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.reshape(-1, 1))  # Reshape to (n_samples, 1)

    ypred = model(X_tensor)

    # Mean squared error loss
    loss = ((ypred - y_tensor) ** 2).mean()

    # Zero gradients
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)

    # Backward pass
    loss.backward()

    # Update (SGD)
    for p in model.parameters():
        p.data += -learning_rate * p.grad

    if k % 20 == 0:
        print(f"Step {k}: loss = {loss.data:.4f}")

print("\n--- Final Results ---")
print(f"Final loss: {loss.data:.4f}\n")

# Calculate accuracy
final_pred = model(Tensor(X))
predictions = np.where(final_pred.data > 0, 1.0, -1.0).flatten()
correct = np.sum(predictions == y)
accuracy = correct / len(y)

print(f"Accuracy: {accuracy * 100:.1f}% ({correct}/{len(y)})")
