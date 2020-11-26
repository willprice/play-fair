from abc import ABC
from typing import Tuple, Dict, Any, List

import numpy as np
from torch.utils.data import Dataset


class FeatureWriter(ABC):
    def append(self, uid: str, features: np.ndarray, labels: Dict[str, Any]) -> None:
        raise NotImplementedError()


class FeatureReader(ABC, Dataset):
    uids: List[str]
    uid2index: Dict[str, int]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
