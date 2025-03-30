import logging
from abc import ABC

import numpy as np
import torch
from PIL import Image

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from core.generation.__base import BaseImageGenerator

AVAILABLE_MODELS = ["Janus-1.3B", "Janus-Pro-7B", "JanusFlow-1.3B"]


class JanusImageGenerator(BaseImageGenerator, ABC):
    name = "Janus Image Generator"

    @classmethod
    def init_model(cls, config):
        model_name = config.get("model_name", "Janus-1.3B")
        if model_name not in AVAILABLE_MODELS:
            logging.warning(f"Model {model_name} not available. Using {AVAILABLE_MODELS[0]} instead.")
            model_name = AVAILABLE_MODELS[0]

        model_path = f"deepseek-ai/{model_name}"
        cls.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        cls.tokenizer = cls.vl_chat_processor.tokenizer

        cls.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        cls.vl_gpt = cls.vl_gpt.to(torch.bfloat16).cuda().eval()

        def test_model(inputs):
            return np.random.rand(3, 224, 224)

        cls.model = test_model

    @classmethod
    def _preprocess_input(cls, inputs):
        conversation = [
            {
                "role": "User",
                "content": inputs,
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = cls.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=cls.vl_chat_processor.sft_format,
            system_prompt="",
        )

        return sft_format + cls.vl_chat_processor.image_start_tag

    @classmethod
    def _postprocess_output(cls, outputs):
        outputs = outputs.numpy()
        outputs = np.clip((outputs + 1) / 2 * 255, 0, 255)

        image = Image.fromarray(outputs.astype(np.uint8))

        return image

    @classmethod
    def invoke_model(cls, prompt: str,
                     temperature: float = 1,
                     cfg_weight: float = 5,
                     image_token_num_per_image: int = 576,
                     img_size: int = 384,
                     patch_size: int = 16, **kwargs):

        parallel_size = 1  # hardcoded need to learn more about this

        input_ids = cls.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = cls.vl_chat_processor.pad_id

        inputs_embeds = cls.vl_gpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = cls.vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                                      past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = cls.vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = cls.vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec: torch.Tensor = cls.vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                      shape=[parallel_size, 8, img_size // patch_size,
                                                             img_size // patch_size])
        dec = dec.to(torch.float32).cpu().permute(0, 2, 3, 1)[0]

        return dec
