from typing import Union

from pydantic import BaseModel, Field

from config.base import ClassConfig
from subset_samplers import (
    ConstructiveRandomSampler,
    ExhaustiveSubsetSampler,
    ProportionalConstructiveRandomSampler,
)


class ExhaustiveSubsetSamplerConfig(ClassConfig):
    kind = Field("ExhaustiveSubsetSampler", const=True)

    def instantiate(self):
        return ExhaustiveSubsetSampler()


class ConstructiveRandomSamplerConfig(ClassConfig):
    kind = Field("ConstructiveRandomSampler", const=True)
    max_samples: int

    def instantiate(self):
        return ConstructiveRandomSampler(max_samples=self.max_samples)


class ProportionalConstructiveRandomSamplerConfig(ClassConfig):
    kind = Field("ConstructiveRandomSampler", const=True)
    min_samples: int
    p: float

    def instantiate(self):
        return ProportionalConstructiveRandomSampler(
            p=self.p, min_samples=self.min_samples
        )


SubsetSamplerConfig = Union[
    ExhaustiveSubsetSamplerConfig,
    ConstructiveRandomSamplerConfig,
    ExhaustiveSubsetSamplerConfig,
]
