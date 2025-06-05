import os

import numpy as np
import torch
from PIL import Image


def generate_sample(music_embedding, batched_prompt_embeds, image_start_embeds, image_token_num_per_image, processor,
                    model, file_prefix):
    cfg_weight = 5
    temperature = 1
    img_size = 384
    patch_size = 16

    parallel_size = music_embedding.shape[0]
    conditional_embeds = torch.concat([music_embedding, batched_prompt_embeds, image_start_embeds], dim=1)
    unconditional_tokens = torch.zeros((1, conditional_embeds.shape[-2]), dtype=torch.int).cuda()
    unconditional_tokens[0, 1:-1] = processor.pad_id
    unconditional_embeds = model.language_model.get_input_embeddings()(unconditional_tokens)
    gen_input_embeds = torch.zeros(
        (conditional_embeds.shape[0] * 2, conditional_embeds.shape[1], conditional_embeds.shape[2]),
        dtype=torch.bfloat16).cuda()

    for i in range(parallel_size * 2):
        if i % 2 != 0:
            gen_input_embeds[i] = unconditional_embeds
        else:
            gen_input_embeds[i] = conditional_embeds[i // 2]

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    inputs_embeds = gen_input_embeds
    for i in range(image_token_num_per_image):
        outputs = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                             past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                             shape=[parallel_size, 8, img_size // patch_size,
                                                    img_size // patch_size])
    dec = dec.to(torch.float32)
    dec = torch.clamp((dec + 1) / 2 * 255, min=0, max=255)

    visual_img = dec.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "{}_{}.jpg".format(file_prefix, i))
        Image.fromarray(visual_img[i]).save(save_path)

    visual_img = dec.to(torch.uint8)
    return visual_img
