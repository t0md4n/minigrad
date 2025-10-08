import numpy as np
from minigrad import MLP, Tensor, make_linear_regression

# Generate linear regression dataset
# y = 2.5x + noise
X, y = make_linear_regression(n_samples=50, n_features=1, noise=0.2, coef=[2.5])

# Initialize model: 1 input -> 4 hidden -> 1 output
model = MLP(1, [4, 1])

print("Linear Regression Dataset")
print(f"Training samples: {X.shape[0]}")
print(f"Model parameters: {len(model.parameters())}")
print(f"Input shape: {X.shape}, Output shape: {y.shape}\n")

# Training loop
learning_rate = 0.01
for k in range(150):
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

    if k % 15 == 0:
        print(f"Step {k}: loss = {loss.data:.4f}")

print("\n--- Final Results ---")
print(f"Final loss: {loss.data:.4f}\n")

# Show sample predictions
print("Sample predictions (first 5):")
final_pred = model(Tensor(X))
for i in range(min(5, len(X))):
    print(f"  Input: {X[i, 0]:.2f} -> Predicted: {final_pred.data[i, 0]:.2f}, Actual: {y[i]:.2f}")
