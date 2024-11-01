import torch
from typing import Dict

class Sample:
    def __init__(self, dense_dict: Dict[str, torch.Tensor], sparse_dict: Dict[str, torch.Tensor]):
        self.dense_dict = dense_dict
        self.sparse_dict = sparse_dict