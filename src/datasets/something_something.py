from pathlib import Path
from typing import Dict, Union

import h5py
import pandas as pd

from gulpio import GulpDirectory
from features import HdfFeatureReader
from features.feature_store import FeatureReader
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvideo.datasets import GulpVideoDataset
from torchvideo.datasets import LabelSet
from torchvideo.datasets.types import Label
from torchvideo.transforms import Compose
from torchvideo.transforms import NDArrayToPILVideo

from .base import Dataset

METADATA_DIR = Path(__file__).parent / "metadata" / "something_something_v2"
CLASSES_CSV_PATH = METADATA_DIR / "classes.csv"


class GulpLabelSet(LabelSet):
    def __init__(self, gulp_dir: GulpDirectory):
        self._gulp_dir = gulp_dir

    def __getitem__(self, video_name: str) -> Label:
        meta = self._gulp_dir.merged_meta_dict[video_name]["meta_data"][0]
        return {"action": meta["idx"], "label": meta["idx"], "uid": meta["id"]}


class SomethingSomethingV2Dataset(Dataset):
    def __init__(self, root: Union[str, Path]):
        super().__init__(data_type="RGB")
        self.root = Path(root)
        self._class2str = pd.read_csv(CLASSES_CSV_PATH, index_col="id")[
            "name"
        ].to_dict()
        self._str2class = {v: k for k, v in self._class2str.items()}

    def class_count(self) -> int:
        return 174

    def train_dataset(self, *args, **kwargs) -> TorchDataset:
        return self._make_dataset(self.root / "train", *args, **kwargs)

    def validation_dataset(self, *args, **kwargs) -> TorchDataset:
        return self._make_dataset(self.root / "validation", *args, **kwargs)

    def _make_dataset(self, root: Path, *args, **kwargs):
        if "transform" in kwargs:
            kwargs["transform"] = Compose([NDArrayToPILVideo(), kwargs["transform"]])
        gulp_dir = GulpDirectory(str(root))
        label_set = GulpLabelSet(gulp_dir)
        if not (root / 'data_0.gulp').exists():
            raise ValueError(f"Missing data_0.gulp in {root}")
        return GulpVideoDataset(
            root, label_set=label_set, gulp_directory=gulp_dir, *args, **kwargs
        )

    def class2str(self) -> Dict[int, str]:
        return self._class2str

    def str2class(self) -> Dict[str, int]:
        return self._str2class


class SomethingSomethingV2FeaturesDataset(Dataset):
    def __init__(self, features_group: h5py.Group, in_mem: bool = False):
        super().__init__(data_type="features")
        self.features_group = features_group
        self.feature_dim = features_group["validation"]["features"].attrs["feature_dim"]
        self._class2str = pd.read_csv(CLASSES_CSV_PATH, index_col="id")[
            "name"
        ].to_dict()
        self._str2class = {v: k for k, v in self._class2str.items()}
        self.in_mem = in_mem

    def class_count(self) -> int:
        return 174

    def train_dataset(self, *args, **kwargs) -> TorchDataset:
        return self._make_dataset(self.features_group["train"], *args, **kwargs)

    def validation_dataset(self, *args, **kwargs) -> TorchDataset:
        return self._make_dataset(self.features_group["validation"], *args, **kwargs)

    def _make_dataset(self, root: h5py.Group, *args, **kwargs) -> FeatureReader:
        return HdfFeatureReader(root, *args, in_mem=self.in_mem, **kwargs)

    def class2str(self) -> Dict[int, str]:
        return self._class2str

    def str2class(self) -> Dict[str, int]:
        return self._str2class
