from typing import List

import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class ImageBindEmbedder:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    def embed_image(self, image_paths):
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings["vision"].cpu().numpy()

    def embed_audio(self, audio_paths):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings["audio"].cpu().numpy()

    def embed_text(self, texts: List[str]):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(texts, self.device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings["text"].cpu().numpy()
