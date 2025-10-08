import numpy as np
import json
from .tensor import Tensor

class Layer:
    """
    Fully connected neural network layer using matrix operations.
    More efficient than per-neuron computation.

    Forward pass: y = activation(x @ W + b)
    where x is (batch, nin), W is (nin, nout), b is (nout,)
    """

    def __init__(self, nin, nout, activation='tanh'):
        """
        Initialize a fully connected layer.

        Args:
            nin: Number of input features
            nout: Number of output features (neurons)
            activation: Activation function ('tanh', 'relu', 'sigmoid', or None)
        """
        # Xavier/Glorot initialization for better gradient flow
        std = np.sqrt(2.0 / (nin + nout))
        self.W = Tensor(np.random.randn(nin, nout) * std)
        self.b = Tensor(np.zeros(nout))
        self.activation = activation

    def __call__(self, x):
        """
        Forward pass through the layer.

        Args:
            x: Input tensor of shape (batch, nin) or (nin,) for single sample

        Returns:
            Output tensor of shape (batch, nout) or (nout,)
        """
        # Ensure x is a Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Linear transformation: x @ W + b
        out = x @ self.W + self.b

        # Apply activation function
        if self.activation == 'tanh':
            out = out.tanh()
        elif self.activation == 'relu':
            out = out.relu()
        elif self.activation == 'sigmoid':
            out = out.sigmoid()
        elif self.activation is None:
            pass  # Linear layer (no activation)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return out

    def parameters(self):
        """Return list of trainable parameters."""
        return [self.W, self.b]

    def save(self, path):
        """
        Save layer weights and configuration to a file.

        Args:
            path: File path to save to (will add .npz extension if not present)
        """
        if not path.endswith('.npz'):
            path += '.npz'

        # Save weights, biases, and metadata
        np.savez(
            path,
            W=self.W.data,
            b=self.b.data,
            nin=self.W.shape[0],
            nout=self.W.shape[1],
            activation=self.activation if self.activation is not None else ''
        )

    @classmethod
    def load(cls, path):
        """
        Load a layer from a saved file.

        Args:
            path: File path to load from

        Returns:
            Layer instance with loaded weights
        """
        if not path.endswith('.npz'):
            path += '.npz'

        # Load data
        data = np.load(path, allow_pickle=True)

        # Extract metadata
        nin = int(data['nin'])
        nout = int(data['nout'])
        activation = str(data['activation'])
        if activation == '':
            activation = None

        # Create layer with loaded architecture
        layer = cls(nin, nout, activation=activation)

        # Load weights and biases
        layer.W.data = data['W']
        layer.b.data = data['b']

        return layer


class MLP:
    """
    Multi-Layer Perceptron (feedforward neural network).
    Stacks multiple layers sequentially.
    """

    def __init__(self, nin, nouts, activation='tanh', output_activation=None):
        """
        Initialize an MLP.

        Args:
            nin: Number of input features
            nouts: List of output sizes for each layer
            activation: Activation function for hidden layers
            output_activation: Activation for final layer (default: same as hidden)
        """
        sz = [nin] + nouts
        self.layers = []

        # Create hidden layers
        for i in range(len(nouts) - 1):
            self.layers.append(Layer(sz[i], sz[i+1], activation=activation))

        # Create output layer
        final_activation = output_activation if output_activation is not None else activation
        self.layers.append(Layer(sz[-2], sz[-1], activation=final_activation))

    def __call__(self, x):
        """
        Forward pass through all layers.

        Args:
            x: Input tensor of shape (batch, nin) or (nin,) for single sample

        Returns:
            Output tensor
        """
        # Ensure x is a Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        """Return flattened list of all parameters in the network."""
        return [p for layer in self.layers for p in layer.parameters()]

    def save(self, path):
        """
        Save the entire MLP model to a file.

        Args:
            path: File path to save to (will add .npz extension if not present)
        """
        if not path.endswith('.npz'):
            path += '.npz'

        # Collect all weights and biases from all layers
        save_dict = {}
        for i, layer in enumerate(self.layers):
            save_dict[f'W_{i}'] = layer.W.data
            save_dict[f'b_{i}'] = layer.b.data
            save_dict[f'activation_{i}'] = layer.activation if layer.activation is not None else ''

        # Save architecture metadata as JSON string
        nin = self.layers[0].W.shape[0]
        nouts = [layer.W.shape[1] for layer in self.layers]

        # Determine activations
        # All hidden layers should have same activation, output might differ
        hidden_activation = self.layers[0].activation if len(self.layers) > 0 else 'tanh'
        output_activation = self.layers[-1].activation if len(self.layers) > 0 else None

        # Check if output activation is different from hidden
        if len(self.layers) > 1 and output_activation == hidden_activation:
            output_activation = None  # Same as hidden, so we'll use None in constructor

        metadata = {
            'nin': nin,
            'nouts': nouts,
            'activation': hidden_activation if hidden_activation is not None else '',
            'output_activation': output_activation if output_activation is not None else ''
        }
        save_dict['metadata'] = json.dumps(metadata)

        # Save everything
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        """
        Load an MLP model from a saved file.

        Args:
            path: File path to load from

        Returns:
            MLP instance with loaded weights
        """
        if not path.endswith('.npz'):
            path += '.npz'

        # Load data
        data = np.load(path, allow_pickle=True)

        # Parse metadata
        metadata = json.loads(str(data['metadata']))
        nin = metadata['nin']
        nouts = metadata['nouts']
        activation = metadata['activation']
        output_activation = metadata['output_activation']

        # Handle empty strings as None
        if activation == '':
            activation = None
        if output_activation == '':
            output_activation = None

        # Create MLP with saved architecture
        model = cls(nin, nouts, activation=activation, output_activation=output_activation)

        # Load weights and biases for each layer
        for i, layer in enumerate(model.layers):
            layer.W.data = data[f'W_{i}']
            layer.b.data = data[f'b_{i}']

        return model


# Backward compatibility: keep old scalar-based classes
try:
    from engine import Value

    class Neuron:
        """Legacy scalar-based neuron (for backward compatibility)."""
        def __init__(self, nin):
            self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
            self.b = Value(np.random.uniform(-1, 1))

        def __call__(self, x):
            act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
            out = act.tanh()
            return out

        def parameters(self):
            return self.w + [self.b]

    class LegacyLayer:
        """Legacy scalar-based layer (for backward compatibility)."""
        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            outs = [n(x) for n in self.neurons]
            return outs[0] if len(outs) == 1 else outs

        def parameters(self):
            return [p for neuron in self.neurons for p in neuron.parameters()]

    class LegacyMLP:
        """Legacy scalar-based MLP (for backward compatibility)."""
        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = [LegacyLayer(sz[i], sz[i+1]) for i in range(len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]

except ImportError:
    # engine.py not available, skip legacy classes
    pass
