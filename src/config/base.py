from abc import ABC

from pydantic import BaseModel

__all__ = ['ClassConfig']

class ClassConfig(BaseModel, ABC):
    def instantiate(self):
        raise NotImplementedError()