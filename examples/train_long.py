"""
Long Training Script with Checkpointing

This script demonstrates meaningful long-term training on a challenging multi-class
classification task. It includes:
- Automatic checkpointing every N steps
- Progress logging with metrics
- Resume capability from last checkpoint
- Best model tracking
- Training history visualization

Task: 5-class spiral classification (challenging non-linear boundaries)
Expected training time: 10-30 minutes depending on system
"""
import numpy as np
import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minigrad import MLP, Tensor, make_spirals


class TrainingSession:
    """Manages training with checkpointing and progress tracking."""

    def __init__(self, model, X_train, y_train, X_val, y_val,
                 checkpoint_dir='checkpoints', checkpoint_interval=100):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval

        # Training state
        self.current_step = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'steps': []
        }
        self.best_val_acc = 0.0
        self.best_step = 0

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.start_time = time.time()

    def one_hot_encode(self, y, n_classes):
        """Convert integer labels to one-hot vectors."""
        n_samples = len(y)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y] = 1
        return one_hot

    def compute_accuracy(self, X, y):
        """Compute classification accuracy."""
        X_tensor = Tensor(X)
        predictions = self.model(X_tensor).data
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y)
        return accuracy

    def compute_loss(self, X, y, n_classes=5):
        """Compute cross-entropy loss."""
        X_tensor = Tensor(X)
        y_one_hot = Tensor(self.one_hot_encode(y, n_classes))

        # Forward pass
        logits = self.model(X_tensor)

        # Softmax for probabilities
        probs = logits.softmax(axis=-1)

        # Cross-entropy: -sum(y_true * log(y_pred))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_probs = (probs + epsilon).log()
        loss = -(y_one_hot * log_probs).sum(axis=-1).mean()

        return loss

    def train_step(self, learning_rate=0.01, n_classes=5):
        """Perform one training step."""
        # Compute loss
        loss = self.compute_loss(self.X_train, self.y_train, n_classes)

        # Zero gradients
        for p in self.model.parameters():
            p.grad = np.zeros_like(p.data)

        # Backward pass
        loss.backward()

        # Update parameters (SGD)
        for p in self.model.parameters():
            p.data += -learning_rate * p.grad

        return loss.data

    def evaluate(self, n_classes=5):
        """Evaluate on both train and validation sets."""
        train_acc = self.compute_accuracy(self.X_train, self.y_train)
        val_acc = self.compute_accuracy(self.X_val, self.y_val)

        train_loss = self.compute_loss(self.X_train, self.y_train, n_classes).data
        val_loss = self.compute_loss(self.X_val, self.y_val, n_classes).data

        return train_loss, train_acc, val_loss, val_acc

    def save_checkpoint(self, is_best=False):
        """Save a checkpoint of the current training state."""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.current_step}.npz'
        history_path = self.checkpoint_dir / 'training_history.json'

        # Save model
        self.model.save(str(checkpoint_path))

        # Save training state
        state = {
            'current_step': self.current_step,
            'best_val_acc': self.best_val_acc,
            'best_step': self.best_step,
            'history': self.history
        }

        with open(history_path, 'w') as f:
            json.dump(state, f, indent=2)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.npz'
            self.model.save(str(best_path))
            print(f"  ✓ New best model saved! (accuracy: {self.best_val_acc:.2%})")

        return checkpoint_path

    def load_checkpoint(self):
        """Load the most recent checkpoint if it exists."""
        history_path = self.checkpoint_dir / 'training_history.json'

        if not history_path.exists():
            print("No checkpoint found. Starting training from scratch.")
            return False

        # Load training state
        with open(history_path, 'r') as f:
            state = json.load(f)

        self.current_step = state['current_step']
        self.best_val_acc = state['best_val_acc']
        self.best_step = state['best_step']
        self.history = state['history']

        # Load model
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.current_step}.npz'
        if checkpoint_path.exists():
            self.model = MLP.load(str(checkpoint_path))
            print(f"Resumed from checkpoint at step {self.current_step}")
            print(f"Best validation accuracy so far: {self.best_val_acc:.2%} (step {self.best_step})")
            return True
        else:
            print("Checkpoint metadata found but model file missing. Starting from scratch.")
            return False

    def train(self, n_steps=3000, learning_rate=0.01, eval_interval=50, n_classes=5):
        """Run the training loop."""
        print(f"\nStarting training from step {self.current_step} to {n_steps}")
        print(f"Checkpoints will be saved every {self.checkpoint_interval} steps")
        print("-" * 70)

        last_eval_time = time.time()

        for step in range(self.current_step, n_steps):
            # Training step
            train_loss = self.train_step(learning_rate, n_classes)
            self.current_step = step + 1

            # Evaluation
            if (step + 1) % eval_interval == 0 or step == 0:
                train_loss, train_acc, val_loss, val_acc = self.evaluate(n_classes)

                # Update history
                self.history['steps'].append(step + 1)
                self.history['train_loss'].append(float(train_loss))
                self.history['train_acc'].append(float(train_acc))
                self.history['val_loss'].append(float(val_loss))
                self.history['val_acc'].append(float(val_acc))

                # Calculate time and ETA
                current_time = time.time()
                elapsed = current_time - self.start_time
                steps_per_sec = (step + 1 - 0) / elapsed if elapsed > 0 else 0
                remaining_steps = n_steps - (step + 1)
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

                # Format ETA
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)

                # Print progress
                print(f"Step {step + 1:4d}/{n_steps} | "
                      f"Train: loss={train_loss:.4f} acc={train_acc:.2%} | "
                      f"Val: loss={val_loss:.4f} acc={val_acc:.2%} | "
                      f"ETA: {eta_min:2d}m {eta_sec:2d}s")

                # Check if best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_step = step + 1

            # Checkpointing
            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
                print(f"  → Checkpoint saved at step {step + 1}")

                # Also save best model if this is a new best
                if is_best:
                    self.save_checkpoint(is_best=True)

        # Final evaluation and checkpoint
        print("\n" + "=" * 70)
        print("Training completed.")
        print("=" * 70)

        train_loss, train_acc, val_loss, val_acc = self.evaluate(n_classes)
        print(f"\nFinal Results:")
        print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}")
        print(f"\nBest validation accuracy: {self.best_val_acc:.2%} (at step {self.best_step})")

        # Save final checkpoint
        self.save_checkpoint(is_best=(val_acc >= self.best_val_acc))
        print(f"\nFinal checkpoint saved to: {self.checkpoint_dir}")

        # Print training summary
        total_time = time.time() - self.start_time
        print(f"\nTotal training time: {int(total_time // 60)}m {int(total_time % 60)}s")
        print(f"Average time per step: {(total_time / n_steps) * 1000:.2f}ms")


def main():
    print("=" * 70)
    print("Long Training Session: Multi-Class Spiral Classification")
    print("=" * 70)

    # Configuration
    n_samples = 2000
    n_classes = 5
    train_split = 0.8
    n_steps = 5000
    learning_rate = 0.1  # Increased for faster learning
    checkpoint_interval = 200
    eval_interval = 100

    # Set random seed for reproducibility
    np.random.seed(42)

    print("\n1. Generating dataset...")
    X, y = make_spirals(n_samples=n_samples, n_classes=n_classes, noise=0.15, rotations=2.0)

    # Split into train/val
    n_train = int(n_samples * train_split)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"   Training samples:   {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Number of classes:  {n_classes}")
    print(f"   Input features:     {X.shape[1]}")

    # Create model
    print("\n2. Creating model...")
    model = MLP(2, [64, 64, 64, n_classes], activation='relu', output_activation=None)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"   Architecture: 2 -> 64 -> 64 -> 64 -> {n_classes}")
    print(f"   Total parameters: {n_params:,}")

    # Create training session
    print("\n3. Initializing training session...")
    session = TrainingSession(
        model, X_train, y_train, X_val, y_val,
        checkpoint_dir='checkpoints_spiral',
        checkpoint_interval=checkpoint_interval
    )

    # Try to resume from checkpoint
    session.load_checkpoint()

    # Train
    print("\n4. Training...")
    print(f"   Total steps:          {n_steps}")
    print(f"   Learning rate:        {learning_rate}")
    print(f"   Checkpoint interval:  {checkpoint_interval} steps")
    print(f"   Evaluation interval:  {eval_interval} steps")

    try:
        session.train(
            n_steps=n_steps,
            learning_rate=learning_rate,
            eval_interval=eval_interval,
            n_classes=n_classes
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        session.save_checkpoint()
        print("Checkpoint saved. You can resume training by running this script again.")
        return

    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved in: {session.checkpoint_dir}")
    print(f"Best model: {session.checkpoint_dir / 'best_model.npz'}")
    print("\nTo use the trained model:")
    print("  >>> from minigrad import MLP")
    print(f"  >>> model = MLP.load('{session.checkpoint_dir / 'best_model.npz'}')")


if __name__ == '__main__':
    main()
