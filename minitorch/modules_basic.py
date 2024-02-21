"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np
import random
from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor
from .operators import prod

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim = embedding_dim  # Embedding Dimension
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(self.initialize(self.num_embeddings, self.embedding_dim))
        ### END YOUR SOLUTION

    def initialize(self, *shape):
        vals = [random.gauss(0, 1) for _ in range(int(prod(shape)))]
        tensor = Tensor.make(vals, shape, backend=self.backend)
        tensor.requires_grad_(True)
        return tensor
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN YOUR SOLUTION
        indexes = one_hot(x.view(bs*seq_len), self.num_embeddings)
        indexes.backend = self.backend
        embeddings = indexes.__matmul__(self.weights.value)
        embeddings = embeddings.view(bs, seq_len, self.embedding_dim)
        return embeddings
        ### END YOUR SOLUTION

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout
        self.training = True

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN YOUR SOLUTION
        if self.training:
            ratio = np.random.binomial(1, 1.0 - self.p_dropout, x.shape)
            ratio_tensor = tensor_from_numpy(ratio)
            return x * ratio_tensor
        else:
            return x
        ### END YOUR SOLUTION


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.in_size = in_size
        self.out_size = out_size
        self.backend = backend
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(self.initialize(self.in_size, self.out_size))
        # self.weights = Parameter(rand((self.in_size, self.out_size), self.backend, requires_grad=True))
        self.bias_test = bias
        if self.bias_test:
            self.bias = Parameter(self.initialize(self.out_size))
            # self.bias = Parameter(rand((self.out_size, ), self.backend, requires_grad=True))
        ### END YOUR SOLUTION

    def initialize(self, *shape):
        vals = [random.uniform(-1.0/np.sqrt(1/float(self.in_size)), 1.0/np.sqrt(1/float(self.in_size))) for _ in range(int(prod(shape)))]
        tensor = Tensor.make(vals, shape, backend=self.backend)
        tensor.requires_grad_(True)
        return tensor

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        weights = self.weights.value
        # out = x.__matmul__(weights).view(batch, self.out_size)
        out = (x @ weights).view(batch, self.out_size)

        if self.bias_test:
            out = out + self.bias.value.view(1, self.out_size)
        return out


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(ones_tensor_from_numpy((self.dim,), backend))
        self.bias = Parameter(zeros_tensor_from_numpy((self.dim,), backend))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN YOUR SOLUTION

        weights = self.weights.value.view(1, self.dim)
        bias = self.bias.value.view(1, self.dim)
        out = ((x - x.mean(dim=1)) / ((x.var(dim=1)+self.eps).__pow__(0.5))) * weights + bias
        return out
        ### END YOUR SOLUTION