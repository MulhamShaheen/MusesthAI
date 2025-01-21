from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np


text_list = ["A dog.", "A car", "A bird"]
image_paths = ["ImageBind/.assets/dog_image.jpg", "ImageBind/.assets/car_image.jpg", "ImageBind/.assets/bird_image.jpg"]
audio_paths = ["ImageBind/.assets/dog_audio.wav", "ImageBind/.assets/car_audio.wav", "ImageBind/.assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(embeddings)
