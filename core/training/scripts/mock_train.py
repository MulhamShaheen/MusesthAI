import torch
from PIL import Image
from torch.nn.functional import cross_entropy

from core.generation.junas_generator import JanusImageGenerator as JG

JG.init_model({
    "model_name": "Janus-Pro-1B",
    "sys_prompt": "Abstract art for representing emotions"
    })

img = Image.open("test.jpg")

with torch.no_grad():
    image_ids, gen_embeds = JG.get_image_embeds([img])

    random_tensor = torch.randn(1, 32, 2048).cuda().to(torch.bfloat16)
    input_tensor = torch.concat([random_tensor, gen_embeds], dim=1)

    outputs = JG.vl_gpt.language_model.model(inputs_embeds=input_tensor, use_cache=False, past_key_values=None,
                                          decoder_input_ids=1)
    hidden_states = outputs.last_hidden_state  # torch.Size([1, 608, 2048])

    logits = JG.vl_gpt.gen_head(hidden_states)
    logits = logits.permute(0, 2, 1)

    image_ids = image_ids.squeeze(-1)  # torch.Size([1, 608, 2048])
    shifted_image_ids = JG.shift_image_tokens(image_ids)  # torch.Size([1, 576])
    loss = cross_entropy(logits[:, :, -image_ids.shape[-1]:], shifted_image_ids, ignore_index=-100)