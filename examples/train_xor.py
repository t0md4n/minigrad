import numpy as np
from minigrad import MLP, Tensor

# Create a simple dataset (XOR problem)
X = np.array([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
])
y = np.array([1.0, -1.0, -1.0, 1.0])  # desired targets

# Initialize model: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
model = MLP(3, [4, 4, 1])

print("Starting training...")
print(f"Model has {len(model.parameters())} parameters")
print(f"Input shape: {X.shape}, Output shape: {y.shape}\n")

# Training loop
learning_rate = 0.01
for k in range(200):
    # Forward pass (batch processing - all samples at once)
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.reshape(-1, 1))  # Reshape to (4, 1) to match predictions

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
print("Predictions vs Targets:")
final_pred = model(Tensor(X))
for i in range(len(X)):
    print(f"  Input {X[i]} -> Predicted: {final_pred.data[i, 0]:.4f}, Target: {y[i]:.4f}")
