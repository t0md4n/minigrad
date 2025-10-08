import numpy as np

class Tensor:
    """
    Tensor class with automatic differentiation support.
    Wraps numpy arrays and builds a computational graph for backpropagation.
    Scalars are represented as 0-dimensional tensors.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initialize a Tensor.

        Args:
            data: Input data (scalar, list, or numpy array)
            _children: Tuple of parent tensors in the computation graph
            _op: String describing the operation that created this tensor
            label: Optional label for debugging/visualization
        """
        # Convert input to numpy array
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        # Ensure data is at least 0-dimensional
        if self.data.ndim == 0 and not isinstance(data, (int, float, np.number)):
            self.data = self.data.reshape(())

        # Gradient: same shape as data, initialized to zeros
        self.grad = np.zeros_like(self.data, dtype=np.float64)

        # Autograd bookkeeping
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    @property
    def shape(self):
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        return f"Tensor({self.data})"

    # Element-wise addition
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # Handle broadcasting: sum gradients along broadcasted dimensions
            self.grad += self._unbroadcast(out.grad, self.shape)
            other.grad += self._unbroadcast(out.grad, other.shape)
        out._backward = _backward

        return out

    # Element-wise multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward

        return out

    # Element-wise power
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    # Element-wise division
    def __truediv__(self, other):
        return self * other ** -1

    # Negation
    def __neg__(self):
        return self * -1

    # Element-wise subtraction
    def __sub__(self, other):
        return self + (-other)

    # Reverse operations
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        Reverse broadcasting by summing gradient along broadcasted dimensions.
        This ensures gradient shape matches the original tensor shape.
        """
        # Handle scalar case
        if target_shape == ():
            return np.sum(grad)

        # Sum along dimensions that were broadcasted
        ndim_diff = grad.ndim - len(target_shape)
        if ndim_diff > 0:
            # Sum along leading dimensions that were added
            grad = np.sum(grad, axis=tuple(range(ndim_diff)))

        # Sum along dimensions that were size 1 in original
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if target_dim == 1 and grad_dim > 1:
                grad = np.sum(grad, axis=i, keepdims=True)

        return grad

    def backward(self):
        """
        Perform backpropagation through the computational graph.
        Computes gradients for all tensors that led to this one.
        """
        # Build topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient of output to 1
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # Backpropagate through the graph
        for node in reversed(topo):
            node._backward()

    # Activation functions
    def tanh(self):
        """Hyperbolic tangent activation (element-wise)."""
        x = self.data
        t = np.tanh(x)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """ReLU activation (element-wise)."""
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """Sigmoid activation (element-wise): 1 / (1 + exp(-x))."""
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, (self,), 'sigmoid')

        def _backward():
            # Derivative: sigmoid(x) * (1 - sigmoid(x))
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward

        return out

    def softmax(self, axis=-1):
        """
        Softmax activation along specified axis.
        Numerically stable implementation using max subtraction.

        Args:
            axis: Axis along which to apply softmax (default: last axis)
        """
        # Subtract max for numerical stability
        x_max = np.max(self.data, axis=axis, keepdims=True)
        exp_x = np.exp(self.data - x_max)
        softmax_out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        out = Tensor(softmax_out, (self,), 'softmax')

        def _backward():
            # Jacobian of softmax is: S_i * (δ_ij - S_j)
            # For vectors: grad_input = softmax * (grad_output - (grad_output * softmax).sum())
            s = softmax_out
            grad_out = out.grad
            sum_grad = np.sum(grad_out * s, axis=axis, keepdims=True)
            self.grad += s * (grad_out - sum_grad)
        out._backward = _backward

        return out

    def exp(self):
        """Exponential function (element-wise)."""
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        """Natural logarithm (element-wise)."""
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        """
        Sum tensor elements along given axis.

        Args:
            axis: Axis or axes along which to sum. None sums all elements.
            keepdims: If True, retains reduced dimensions as size 1
        """
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            # Gradient broadcasts back to original shape
            grad = out.grad
            if axis is not None and not keepdims:
                # Need to add back dimensions that were removed
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, axis=ax)
            # Broadcast gradient to original shape
            self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        """
        Compute mean along given axis.

        Args:
            axis: Axis or axes along which to compute mean. None means all elements.
            keepdims: If True, retains reduced dimensions as size 1
        """
        n = self.data.size if axis is None else np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            # Gradient broadcasts back to original shape, divided by count
            grad = out.grad / n
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis=axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, axis=ax)
            self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward

        return out

    def __matmul__(self, other):
        """
        Matrix multiplication operator (@).
        Supports batched matrix multiplication.

        For 2D: (m, n) @ (n, p) -> (m, p)
        For higher dims: batch dimensions are broadcasted
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # dL/dA = dL/dC @ B^T
            self.grad += out.grad @ np.swapaxes(other.data, -2, -1)
            # dL/dB = A^T @ dL/dC
            other.grad += np.swapaxes(self.data, -2, -1) @ out.grad
        out._backward = _backward

        return out

    def __rmatmul__(self, other):
        """Reverse matrix multiplication."""
        return Tensor(other) @ self

    def reshape(self, *shape):
        """
        Reshape tensor to new shape.

        Args:
            *shape: New shape (can use -1 for automatic dimension)
        """
        # Handle both reshape(2, 3) and reshape((2, 3))
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        out = Tensor(self.data.reshape(shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    @property
    def T(self):
        """
        Transpose the tensor (swap last two dimensions).
        For 2D tensors, this is standard matrix transpose.
        """
        out = Tensor(np.swapaxes(self.data, -2, -1), (self,), 'T')

        def _backward():
            self.grad += np.swapaxes(out.grad, -2, -1)
        out._backward = _backward

        return out

    def transpose(self, *axes):
        """
        Permute dimensions of the tensor.

        Args:
            *axes: Permutation of dimensions. If not specified, reverses dimensions.
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])

        out = Tensor(np.transpose(self.data, axes), (self,), 'transpose')

        def _backward():
            # Inverse permutation
            if axes is None:
                self.grad += np.transpose(out.grad)
            else:
                inv_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, inv_axes)
        out._backward = _backward

        return out

    def __getitem__(self, idx):
        """
        Indexing and slicing support.

        Examples:
            tensor[0]       # First element
            tensor[:, 1]    # All rows, second column
            tensor[0:5]     # First 5 elements
        """
        out = Tensor(self.data[idx], (self,), 'getitem')

        def _backward():
            grad = np.zeros_like(self.data)
            grad[idx] = out.grad
            self.grad += grad
        out._backward = _backward

        return out
