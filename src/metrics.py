import logging
from typing import Dict, Iterable

from ignite.metrics import Metric, TopKCategoricalAccuracy
LOG = logging.getLogger(__name__)


def accuracy_metrics(
    ks: Iterable[int], output_transform=lambda x: x, prefix=""
) -> Dict[str, Metric]:
    return {
        f"{prefix}accuracy@{k}": TopKCategoricalAccuracy(
            k=k, output_transform=output_transform
        )
        for k in ks
    }


