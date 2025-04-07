from abc import ABC

import numpy as np
from PIL import Image

from core.generation.__base import BaseImageGenerator


class TestImageGenerator(BaseImageGenerator, ABC):
    name = "Test Image Generator"

    @classmethod
    def init_model(cls, config):
        def test_model(inputs):
            return np.random.rand(3, 224, 224)

        cls.model = test_model

    @classmethod
    def _preprocess_input(cls, inputs):
        return inputs

    @classmethod
    def _postprocess_output(cls, outputs):
        outputs = outputs.numpy()
        image = Image.fromarray(outputs.astype(np.uint8).transpose(1, 2, 0))

        return image
    
    @classmethod
    def invoke_model(cls, inputs):
        return cls.model(inputs)
