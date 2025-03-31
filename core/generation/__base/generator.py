from abc import ABC, abstractmethod
from typing import Any

import PIL.Image as Image
import torch


class BaseImageGenerator(ABC):
    name: str
    model: object | callable = None

    @classmethod
    @abstractmethod
    def init_model(cls, config: Any):
        pass

    @classmethod
    @abstractmethod
    def _preprocess_input(cls, inputs: str) -> str:
        pass

    @classmethod
    @abstractmethod
    def _postprocess_output(cls, outputs: torch.Tensor) -> Image:
        pass


    @classmethod
    @abstractmethod
    def invoke_model(cls, **kwargs) -> torch.Tensor:
        pass

    @classmethod
    def generate(cls, inputs: str) -> Image:
        inputs = cls._preprocess_input(inputs)
        outputs = cls.invoke_model(inputs=inputs)
        return cls._postprocess_output(outputs)
