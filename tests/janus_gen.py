import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
import torch
import numpy as np
from PIL import Image
from core.generation.junas_generator.generator import JanusImageGenerator


@pytest.fixture(scope="module")
def mock_config():
    return {
        "model_name": "Janus-Pro-1B",
        "sys_prompt": "Abstract art for representing emotions"
    }


@pytest.fixture
def mock_images():
    # Create a list of mock PIL images
    return [Image.new("RGB", (384, 384), color=(255, 255, 255)) for _ in range(2)]


@pytest.fixture
def mock_inputs():
    # Create mock numpy input data
    return np.random.rand(2, 2048).astype(np.float32)

@pytest.fixture(scope="module")
def janus_generator(mock_config):
    print("Initializing JanusImageGenerator...")
    JanusImageGenerator.init_model(mock_config)
    return JanusImageGenerator


def test_init_model(mock_config):
    JanusImageGenerator.init_model(mock_config)
    assert JanusImageGenerator.model is not None
    assert JanusImageGenerator.vl_chat_processor is not None
    assert JanusImageGenerator.tokenizer is not None


def test_preprocess_input():
    prompt = "Generate an image of a sunset."
    processed_input = JanusImageGenerator._preprocess_input(prompt)
    assert isinstance(processed_input, str)
    assert len(processed_input) > 0


def test_get_image_embeds(mock_config, mock_images, janus_generator):
    with torch.no_grad():
        image_ids, gen_embeds = JanusImageGenerator.get_image_embeds(mock_images)
    assert isinstance(image_ids, torch.Tensor)
    assert isinstance(gen_embeds, torch.Tensor)
    print("Image IDs shape:", image_ids.shape)
    print("gen_embeds shape:", gen_embeds.shape)
    assert image_ids.shape[1] > 0
    assert image_ids.shape[-1] == 576
    assert gen_embeds.shape[1] > 0
    assert gen_embeds.shape[-1] == 2048
#

def test_shift_image_tokens(mock_config, janus_generator):
    mock_image_ids = torch.randint(0, 100, (2, 576)).cuda()
    shifted_tokens = JanusImageGenerator.shift_image_tokens(mock_image_ids)
    assert isinstance(shifted_tokens, torch.Tensor)
    assert shifted_tokens.shape == mock_image_ids.shape


def test_get_text_embeds(mock_config, janus_generator):
    text = "Describe a futuristic city."
    with torch.no_grad():
        text_embeds = JanusImageGenerator.get_text_embeds(text)
    assert isinstance(text_embeds, torch.Tensor)
    print("Shape of text embeds: ", text_embeds.shape)
    assert text_embeds.ndim == 1
    assert text_embeds.dtype == torch.long
