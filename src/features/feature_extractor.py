from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .feature_store import FeatureWriter


class FeatureExtractor:
    """
    Extracts image features from a 2D CNN backbone for every frame in all videos.
    """

    def __init__(
        self,
        backbone_2d: nn.Module,
        device: torch.device,
        frame_batch_size: int = 128,
    ):
        self.model = backbone_2d
        self.frame_batch_size = frame_batch_size
        self.device = device

    def extract(self, dataset_loader: DataLoader, feature_writer: FeatureWriter) -> int:
        total_instances = 0
        self.model.eval()
        total = len(dataset_loader)
        for i, (batch_input, batch_labels) in tqdm(
            enumerate(dataset_loader),
            unit=" video",
            total=total,
            dynamic_ncols=True,
        ):
            # batch_input: (B, T, C, H, W)
            batch_size, n_frames = batch_input.shape[:2]
            flattened_batch_input = batch_input.view((-1, *batch_input.shape[2:]))
            n_chunks = int(np.ceil(len(flattened_batch_input) / self.frame_batch_size))
            chunks = torch.chunk(flattened_batch_input, n_chunks, dim=0)
            flattened_batch_features = []
            for chunk in chunks:
                with torch.no_grad():
                    chunk_features = self.model(chunk.to(self.device))
                    flattened_batch_features.append(chunk_features)
            flattened_batch_features = torch.cat(flattened_batch_features, dim=0)
            batch_features = flattened_batch_features.view(
                (batch_size, n_frames, *flattened_batch_features.shape[1:])
            )

            total_instances += batch_size
            self._append(batch_features, batch_labels, batch_size, feature_writer)
        return total_instances

    def _append(self, batch_features, batch_labels, batch_size, feature_writer):
        batch_uids = batch_labels["uid"]
        batch_features = batch_features.cpu().numpy()
        batch_labels = self._decollate(batch_labels)
        assert len(batch_uids) == batch_size
        assert len(batch_labels) == batch_size
        assert len(batch_features) == batch_size
        for uid, features, labels in zip(batch_uids, batch_features, batch_labels):
            feature_writer.append(uid, features, labels)

    def _move_tensor_dict_to_device(self, batch_labels: Dict[str, Any]) -> None:
        for key in batch_labels.keys():
            target = batch_labels[key]
            if isinstance(target, torch.Tensor):
                batch_labels[key] = target.to(self.device)

    def _decollate(self, batch_labels: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        label_keys = list(batch_labels.keys())
        for label_key, tensor in batch_labels.items():
            if isinstance(tensor, torch.Tensor):
                batch_labels[label_key] = tensor.cpu().numpy()
        decollated_labels = []
        batch_length = len(batch_labels[label_keys[0]])
        for i in range(batch_length):
            decollated_labels.append(
                {label_key: batch_labels[label_key][i] for label_key in label_keys}
            )
        return decollated_labels
