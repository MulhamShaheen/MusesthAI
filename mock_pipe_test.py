import numpy as np

from core.pipelines.audio2img.mock import MockAudio2JanusPipeline

pip = MockAudio2JanusPipeline(config={})

audio = np.ndarray(shape=(1, 2048), dtype=np.float32)
img = pip.invoke(audio)
