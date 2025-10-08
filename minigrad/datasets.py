import math
import random
import numpy as np

def make_moons(n_samples=40, noise=0.1):
    """
    Generate two interleaving half circles (moons).

    Args:
        n_samples: Total number of samples (split evenly between two moons)
        noise: Standard deviation of Gaussian noise added to data

    Returns:
        X: NumPy array of shape (n_samples, 2) with coordinates
        y: NumPy array of shape (n_samples,) with labels (1.0 or -1.0)
    """
    n_samples_per_moon = n_samples // 2
    X = []
    y = []

    for i in range(n_samples_per_moon):
        # First moon (top)
        angle = math.pi * i / n_samples_per_moon
        x = math.cos(angle) + random.gauss(0, noise)
        y_coord = math.sin(angle) + random.gauss(0, noise)
        X.append([x, y_coord])
        y.append(1.0)

        # Second moon (bottom)
        angle = math.pi * i / n_samples_per_moon
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y_coord = -math.sin(angle) - 0.5 + random.gauss(0, noise)
        X.append([x, y_coord])
        y.append(-1.0)

    # Shuffle the data
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return np.array(X), np.array(y)

def make_circles(n_samples=40, noise=0.05, factor=0.5):
    """
    Generate concentric circles for binary classification.

    Args:
        n_samples: Total number of samples (split evenly between circles)
        noise: Standard deviation of Gaussian noise
        factor: Scale factor between inner and outer circle (0 < factor < 1)

    Returns:
        X: NumPy array of shape (n_samples, 2) with coordinates
        y: NumPy array of shape (n_samples,) with labels (1.0 for outer, -1.0 for inner)
    """
    n_samples_per_circle = n_samples // 2
    X = []
    y = []

    for i in range(n_samples_per_circle):
        # Outer circle
        angle = 2 * math.pi * i / n_samples_per_circle
        x = math.cos(angle) + random.gauss(0, noise)
        y_coord = math.sin(angle) + random.gauss(0, noise)
        X.append([x, y_coord])
        y.append(1.0)

        # Inner circle
        angle = 2 * math.pi * i / n_samples_per_circle
        x = factor * math.cos(angle) + random.gauss(0, noise)
        y_coord = factor * math.sin(angle) + random.gauss(0, noise)
        X.append([x, y_coord])
        y.append(-1.0)

    # Shuffle the data
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return np.array(X), np.array(y)

def make_linear_regression(n_samples=50, n_features=1, noise=0.1, coef=None):
    """
    Generate linear regression dataset.

    Args:
        n_samples: Number of samples
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        coef: List of coefficients for linear combination (if None, random coefficients used)

    Returns:
        X: NumPy array of shape (n_samples, n_features)
        y: NumPy array of shape (n_samples,) with continuous target values
    """
    if coef is None:
        coef = [random.uniform(-2, 2) for _ in range(n_features)]

    bias = random.uniform(-1, 1)

    X = []
    y = []

    for _ in range(n_samples):
        # Generate random features in range [-2, 2]
        features = [random.uniform(-2, 2) for _ in range(n_features)]

        # Compute target as linear combination + noise
        target = sum(c * f for c, f in zip(coef, features)) + bias
        target += random.gauss(0, noise)

        X.append(features)
        y.append(target)

    return np.array(X), np.array(y)

def make_spirals(n_samples=1000, n_classes=5, noise=0.3, rotations=2.5):
    """
    Generate multi-class spiral dataset for challenging classification.

    This creates interleaving spirals, one for each class. It's a difficult
    non-linear classification problem that requires deep networks to solve.

    Args:
        n_samples: Total number of samples (divided among classes)
        n_classes: Number of spiral arms/classes (default: 5)
        noise: Standard deviation of Gaussian noise added to points
        rotations: Number of rotations each spiral makes (default: 2.5)

    Returns:
        X: NumPy array of shape (n_samples, 2) with coordinates
        y: NumPy array of shape (n_samples,) with class labels (0 to n_classes-1)
    """
    n_samples_per_class = n_samples // n_classes
    X = []
    y = []

    for class_idx in range(n_classes):
        for i in range(n_samples_per_class):
            # Radius grows linearly from 0 to 1
            r = i / n_samples_per_class

            # Angle includes rotation and class offset
            theta = (i / n_samples_per_class) * rotations * 2 * math.pi
            theta += (class_idx * 2 * math.pi / n_classes)  # Offset each class

            # Compute point with noise
            x = r * math.cos(theta) + random.gauss(0, noise * r)  # Scale noise with radius
            y_coord = r * math.sin(theta) + random.gauss(0, noise * r)

            X.append([x, y_coord])
            y.append(class_idx)

    # Shuffle the data
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)
