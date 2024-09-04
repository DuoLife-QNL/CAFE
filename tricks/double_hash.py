import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DoubleHashEmbeddingBag(nn.Module):
    r"""Computes sums or means over 'bags' of embeddings using double hashing technique.
    
    This class is similar to QREmbeddingBag but uses double hashing to reduce embedding sizes.
    
    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (int): size of each embedding vector.
        num_collisions (int): number of collisions to enforce.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    Attributes:
        weight (Tensor): the learnable weights of the embedding table is the module of shape
                         `(2 * (int(np.ceil(num_categories / num_collisions)) + 1), embedding_dim)` initialized using a uniform distribution
                         with sqrt(1 / num_categories).

    Inputs: :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
        :attr:`per_index_weights` (Tensor, optional)

        - If :attr:`input` is 2D of shape `(B, N)`,

          it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
          this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
          :attr:`offsets` is ignored and required to be ``None`` in this case.

        - If :attr:`input` is 1D of shape `(N)`,

          it will be treated as a concatenation of multiple bags (sequences).
          :attr:`offsets` is required to be a 1D tensor containing the
          starting index positions of each bag in :attr:`input`. Therefore,
          for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
          having ``B`` bags. Empty bags (i.e., having 0-length) will have
          returned vectors filled by zeros.

        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.

    Output shape: `(B, embedding_dim)`

    """
    __constants__ = [
        "num_categories",
        "embedding_dim",
        "num_collisions",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
    ]

    def __init__(
        self,
        num_categories,
        embedding_dim,
        compress_rate,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        mode="mean",
        sparse=False,
        _weight=None,
    ):
        super(DoubleHashEmbeddingBag, self).__init__()

        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.compress_rate = compress_rate
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        self.hash_range = int(np.ceil(num_categories * compress_rate)) + 1
        self.num_embeddings = 2 * self.hash_range

        if _weight is None:
            self.weight = Parameter(
                torch.Tensor(self.num_embeddings, self.embedding_dim)
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                self.num_embeddings,
                self.embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight)
        self.mode = mode
        self.sparse = sparse

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -np.sqrt(1 / self.num_categories), np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None, per_sample_weights=None):
        def hash_fn(x):
            # for shift operation, since the original id in input is all positive, the shifting result is still positive.
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
            x = (x ^ (x >> 27)) * 0x94d049bb133111eb
            x = x ^ (x >> 31)
            return x

        key1 = input % self.hash_range
        key2 = hash_fn(input) % self.hash_range + self.hash_range

        embed_1 = F.embedding_bag(
            key1,
            self.weight,
            offsets,
            sparse=self.sparse,
            mode=self.mode,
            per_sample_weights=per_sample_weights,
        )
        embed_2 = F.embedding_bag(
            key2,
            self.weight,
            offsets,
            sparse=self.sparse,
            mode=self.mode,
            per_sample_weights=per_sample_weights,
        )

        return embed_1 + embed_2

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        s += ", mode={mode}"
        return s.format(**self.__dict__)