# References
# AdaEmbed: Adaptive Embedding for Large-Scale Recommendation Models
# Fan Lai, et al. OSDI, 2023.


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import random as ra
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import ctypes
import time
import random
import math


sketch_time = 0


def get_sketch_time():
    global sketch_time
    return sketch_time


def reset_sketch_time():
    global sketch_time
    sketch_time = 0


"""
class Sketch:
    def __init__(self, threshold, limit):
        self.n = 17993
        self.threshold = threshold
        self.cnt = np.zeros((self.n, 4))
        self.val = np.zeros((self.n, 4))
        self.p = 1.08
        self.top_k = []
        self.hot_num = 0
        self.limit = limit
    def insert(self, val):
        f = 0
        key = val % self.n
        for i in range(0, 4):
            if self.val[key][i] == val:
                self.cnt[key][i] += 1
                if (self.cnt[key][i] == self.threshold and (val not in h_dict) and self.hot_num < self.limit):
                    h_dict[val] = hot_num
                    hot_num += 1
                    f = 1
                    print(f"val: {val}, hot_num: {self.hot_num}")
                p = i
                while p != 0 and self.cnt[key][p] > self.cnt[key][p-1]:
                    self.cnt[key][p-1], self.cnt[key][p] = self.cnt[key][p], self.cnt[key][p-1]
                    self.val[key][p-1], self.val[key][p] = self.val[key][p], self.val[key][p-1]
                    p -= 1
                return
        for i in range(0, 4):  
            if self.cnt[key][i] == 0:
                self.cnt[key][i] = 1
                self.val[key][i] = val
                return
        
        P = ra.random()
        if (P < (1.0 / self.p) ** self.cnt[key][3]):
            self.cnt[key][3] -= 1
            if self.cnt[key][3] == 0:
                self.cnt[key][3] = 1
                self.val[key][3] = val
        return f
    def get_top_k(self):
        return self.top_k
"""


class FrequencyPrunedEmbeddingBag(nn.Module):
    r"""Computes sums or means over two 'bags' of embeddings, one using the quotient
    of the indices and the other using the remainder of the indices, without
    instantiating the intermediate embeddings, then performs an operation to combine these.

    For bags of constant length and no :attr:`per_sample_weights`, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=0)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=0)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=0)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    QREmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights`` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Known Issues:
    Autograd breaks with multiple GPUs. It breaks only with multiple embeddings.

    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table. If ``"add"``
                              or ``"mult"`` operation are used, these embedding dimensions must be
                              the same. If a single embedding_dim is used, then it will use this
                              embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    Attributes:
        weight (Tensor): the learnable weights of each embedding table is the module of shape
                         `(num_embeddings, embedding_dim)` initialized using a uniform distribution
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
    """
    __constants__ = [
        "num_categories",
        "embedding_dim",
        "num_collisions",
        "operation",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "mode",
        "sparse",
    ]
    """

    def __init__(
        self,
        num_categories,
        embedding_dim,
        hot_cat,
        device,
    ):
        super(FrequencyPrunedEmbeddingBag, self).__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.hot_features = hot_cat
        self.device = device
        self.hot_num = hot_cat.shape[0]
        self.hot_rate = self.hot_num / self.num_categories
        self.avaible_set = np.arange(1, self.hot_num + 1)
        self.head = 0
        print(f"hot_nums: {self.hot_num}")
        self.num_embeddings = self.num_categories
        if self.num_categories > 200:
            self.num_embeddings = self.hot_num + 1
            self.dic = np.zeros(self.num_categories, dtype=np.int32)  # Initialize dic with zeros
            # for i in range(self.hot_num):
            #     self.dic[self.hot_features[i]] = i + 1  # Map hot features to their indices in weight
            # All non-hot categories point to self.weight[0]
            self.dic[self.hot_features] = np.arange(1, self.hot_num + 1)
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        # nn.init.constant_(self.weight, 0)  # Set all embeddings to 0
        nn.init.uniform_(self.weight, -np.sqrt(1 / self.num_embeddings), np.sqrt(1 / self.num_embeddings))

    def forward(self, input, offsets=None, per_sample_weights=None, test=False):
        with torch.no_grad():
            self.weight[0] = 0
        if self.num_categories > 200:
            dic = self.dic[input.cpu()]
        else:
            dic = input.cpu()
        dic = torch.tensor(dic).to(self.device)
        embed = F.embedding_bag(
            dic,
            self.weight,
            offsets,
            sparse=True,
        )

        return embed

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
