from typing import Dict, Any, Tuple

import h5py
import numpy as np
from torchvideo.samplers import FrameSampler, FullVideoSampler, frame_idx_to_list

from .feature_store import FeatureWriter, FeatureReader
import logging

LOG = logging.getLogger(__name__)

_FEATURE_GROUP = "features"
_LABEL_GROUP = "labels"
_UID_GROUP = "uids"


string_dtype = h5py.string_dtype(encoding="utf-8")


class HdfFeatureWriter(FeatureWriter):
    def __init__(self, root_group: h5py.Group, total_length: int, feature_dim: int):
        self.root_group = root_group
        LOG.debug(f"Creating dataset '{_FEATURE_GROUP}'")
        self._feature_dataset = self.root_group.create_dataset(
            _FEATURE_GROUP,
            (total_length,),
            dtype=h5py.vlen_dtype(np.dtype("float32")),
        )
        self._feature_dataset.attrs["feature_dim"] = feature_dim

        LOG.debug(f"Creating group '{_LABEL_GROUP}'")
        self._label_group = self.root_group.create_group(_LABEL_GROUP)
        self._label_datasets = dict()

        LOG.debug(f"Creating dataset '{_UID_GROUP}'")
        self._uid_dataset = self.root_group.create_dataset(
            _UID_GROUP, (total_length,), dtype=string_dtype
        )
        self.total_length = total_length
        self.feature_dim = feature_dim
        self._current_index = 0

    def append(self, uid: str, features: np.ndarray, labels: Dict[str, Any]) -> None:
        assert features.shape[1] == self.feature_dim
        if self._current_index == 0:
            self._initialise_label_groups(labels)
        # stored in (time, feature_dim) format
        self._feature_dataset[self._current_index] = features.ravel()
        self._uid_dataset[self._current_index] = uid
        for name, value in labels.items():
            self._label_datasets[name][self._current_index] = value
        self._current_index += 1

    def _initialise_label_groups(self, labels: Dict[str, Any]):
        for name, value in labels.items():
            LOG.debug(f"Creating dataset '{_LABEL_GROUP}/{name}'")
            self._label_datasets[name] = self._label_group.create_dataset(
                name, (self.total_length,), dtype=self._get_dtype(value)
            )

    def _get_dtype(self, value: Any):
        if isinstance(value, (int, np.int32, np.int64)):
            return np.dtype("int64")
        if isinstance(value, str):
            return h5py.string_dtype(encoding="utf-8")


class HdfFeatureReader(FeatureReader):
    def __init__(
        self,
        root_group: h5py.Group,
        in_mem: bool = True,
        sampler: FrameSampler = FullVideoSampler(),
    ):
        self.root_group = root_group
        self.feature_dataset = self.root_group[_FEATURE_GROUP]
        self.feature_dim = self.feature_dataset.attrs["feature_dim"]
        self.sampler = sampler
        if in_mem:
            self.features = [
                feature.reshape(-1, self.feature_dim)
                for feature in self.feature_dataset
            ]
        else:
            self.features = None

        self.uids = self.root_group[_UID_GROUP][:]
        self.label_sets = {
            name: dataset[:] for name, dataset in self.root_group[_LABEL_GROUP].items()
        }
        self.uid2index = {uid: i for i, uid in enumerate(self.uids)}

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        # __getitem__ methods should raise an IndexError when the passed index is
        # invalid, this supports iterating over them, e.g. for x in hdf_feature_reader
        if index >= len(self):
            raise IndexError
        if self.features is not None:
            features = self.features[index]
        else:
            features: np.ndarray = self.feature_dataset[index].reshape(
                -1, self.feature_dim
            )
        frame_idxs = self.sampler.sample(len(features))

        def decode_label(val: Any):
            if isinstance(val, bytes):
                return val.decode('utf8')  # decode from bytes to string
            return val

        labels = {name: decode_label(values[index]) for name, values in
                  self.label_sets.items()}
        return features[np.array(frame_idx_to_list(frame_idxs))], labels

    def __len__(self):
        return len(self.uids)
