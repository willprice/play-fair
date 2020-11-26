from typing import Union

from pydantic import BaseModel, Field
from torchvideo.samplers import FullVideoSampler, TemporalSegmentSampler

from config.base import ClassConfig
from frame_sampling import RandomSampler

__all__ = [
    "TemporalSegmentSamplerConfig",
    "RandomSamplerConfig",
    "FullVideoSamplerConfig",
    "FrameSamplerConfig",
    "FrameSamplersConfig",
]


class TemporalSegmentSamplerConfig(ClassConfig):
    kind: str = Field("TemporalSegmentSampler", const=True)
    frame_count: int
    snippet_length: int = 1
    test_mode: bool = False

    def instantiate(self):
        return TemporalSegmentSampler(
            segment_count=self.frame_count,
            snippet_length=self.snippet_length,
            test=self.test_mode,
        )


class RandomSamplerConfig(ClassConfig):
    kind: str = Field("RandomSampler", const=True)
    frame_count: int
    snippet_length: int = 1
    test_mode: bool = False

    def instantiate(self):
        return RandomSampler(
            frame_count=self.frame_count,
            snippet_length=self.snippet_length,
            test=self.test_mode,
        )


class FullVideoSamplerConfig(ClassConfig):
    kind: str = Field("FullVideoSampler", const=True)
    frame_step: int = 1

    def instantiate(self):
        return FullVideoSampler(frame_step=self.frame_step)


FrameSamplerConfig = Union[
    TemporalSegmentSamplerConfig, RandomSamplerConfig, FullVideoSamplerConfig
]


class FrameSamplersConfig(BaseModel):
    train: FrameSamplerConfig
    test: FrameSamplerConfig
