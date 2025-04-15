import torch
from PIL.Image import Image
import numpy as np

from core.generation.junas_generator import JanusImageGenerator
from core.pipelines.__base.pipeline import Pipeline
from core.projection import AudioProjection
from utils import logs

logger = logs.create_logger(__name__.split(".")[-1])


class MockAudio2JanusPipeline(Pipeline):
    def __init__(self, **kwargs):
        super(MockAudio2JanusPipeline, self).__init__(**kwargs)
        self.generator = JanusImageGenerator
        self.generator.init_model(**kwargs)
        self.projector = AudioProjection(input_dim=512, output_dim=2048)

    def invoke(self, audio_inputs: np.ndarray, *args, **kwargs):
        # read audio and make random inputs
        logger.info("Processing audio..")
        audio_embeds_shape = [512, 512]  # self.generator.audio_embeds_shape
        random_tensor = torch.randn(audio_embeds_shape)
        with torch.no_grad():
            projection = self.projector(random_tensor)
            image: Image = self.generator.generate_from_embeds(projection.numpy())

        return image
