from PIL import Image
import os
import torch
import numpy as np
from tqdm.auto import tqdm  # Make sure tqdm is imported if not already

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from core.training.scripts.train_val_loops import TrainConfig


# (Assuming other necessary imports like torch, nn, models are already in your script)

def generate_sample(music_embedding, batched_prompt_embeds, image_start_embeds, image_token_num_per_image, processor,
                    model, file_prefix, save_to_disk=False, output_dir="generated_images_output"):
    """
    Generates an image based on music, prompt, and start image embeddings.

    Args:
        music_embedding (torch.Tensor or None): Embeddings for the system prompt. Can be None.
        batched_prompt_embeds (torch.Tensor): Embeddings for the projected audio.
        image_start_embeds (torch.Tensor): Embeddings for the image start tag.
        image_token_num_per_image (int): Number of tokens to generate for the image.
        processor: The VLChatProcessor.
        model: The MultiModalityCausalLM model.
        file_prefix (str): Prefix for the output file name if saving.
        save_to_disk (bool): If True, saves the generated image to disk.
        output_dir (str): Directory to save the image.
    """
    cfg_weight = 5
    temperature = 1
    img_size = 384  # Should match model's training
    patch_size = 16  # Should match model's training

    # Ensure model is in evaluation mode and on the correct device
    # It's good practice, though this might be set outside this function too
    model.eval()
    device = model.device  # Assuming model has a .device attribute

    # Handle inputs and move to device
    if music_embedding is not None:
        music_embedding = music_embedding.to(device, dtype=torch.bfloat16)
    batched_prompt_embeds = batched_prompt_embeds.to(device, dtype=torch.bfloat16)
    image_start_embeds = image_start_embeds.to(device, dtype=torch.bfloat16)

    parallel_size = batched_prompt_embeds.shape[0]  # Batch size of the audio embeds

    # Construct conditional_embeds carefully if music_embedding (system prompt) is None
    elements_to_concat = []
    if music_embedding is not None:
        if music_embedding.shape[0] != parallel_size:  # Repeat system prompt if necessary
            music_embedding = music_embedding.repeat(parallel_size, 1, 1)
        elements_to_concat.append(music_embedding)

    elements_to_concat.append(batched_prompt_embeds)  # This is the projected audio input

    if image_start_embeds.shape[0] != parallel_size:  # Repeat image start if necessary
        image_start_embeds = image_start_embeds.repeat(parallel_size, 1, 1)
    elements_to_concat.append(image_start_embeds)

    conditional_embeds = torch.cat(elements_to_concat, dim=1)

    # Unconditional embeddings
    # Assuming processor.pad_id is an integer
    unconditional_tokens = torch.full((1, conditional_embeds.shape[-2]), processor.pad_id, dtype=torch.int,
                                      device=device)
    if conditional_embeds.shape[-2] > 2:  # Ensure there's space for non-pad tokens
        unconditional_tokens[0, 0] = processor.tokenizer.bos_token_id if hasattr(processor.tokenizer,
                                                                                 'bos_token_id') and processor.tokenizer.bos_token_id is not None else processor.pad_id  # Or some other start token
        unconditional_tokens[0, -1] = processor.tokenizer.eos_token_id if hasattr(processor.tokenizer,
                                                                                  'eos_token_id') and processor.tokenizer.eos_token_id is not None else processor.pad_id  # Or some other end token

    unconditional_embeds = model.language_model.get_input_embeddings()(unconditional_tokens).to(dtype=torch.bfloat16)
    if unconditional_embeds.shape[0] != parallel_size:
        unconditional_embeds = unconditional_embeds.repeat(parallel_size, 1, 1)

    gen_input_embeds = torch.zeros(
        (parallel_size * 2, conditional_embeds.shape[1], conditional_embeds.shape[2]),
        dtype=torch.bfloat16, device=device)

    for i in range(parallel_size * 2):
        if i % 2 != 0:  # Unconditional
            gen_input_embeds[i] = unconditional_embeds[i // 2]
        else:  # Conditional
            gen_input_embeds[i] = conditional_embeds[i // 2]

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int, device=device)
    inputs_embeds = gen_input_embeds

    current_past_key_values = None
    for i in range(image_token_num_per_image):
        outputs = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                             past_key_values=current_past_key_values)
        current_past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # Prepare next input: interleave conditional and unconditional next tokens
        next_token_interleaved = torch.cat([next_token, next_token], dim=1).view(-1,
                                                                                 1)  # Shape becomes (parallel_size*2, 1)
        img_embeds = model.prepare_gen_img_embeds(next_token_interleaved)  # expects (B, 1)
        inputs_embeds = img_embeds.unsqueeze(dim=1)  # Becomes (B, 1, Dim)

    dec = model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                             shape=[parallel_size, 8, img_size // patch_size,
                                                    img_size // patch_size])
    dec = dec.to(torch.float32)  # Ensure float32 for clamping and scaling
    dec = torch.clamp((dec + 1) / 2 * 255, min=0, max=255)

    visual_img_tensor = dec.to(torch.uint8)  # Return tensor as uint8

    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)
        # Transpose from (B, C, H, W) to (B, H, W, C) for PIL
        processed_imgs_for_saving = visual_img_tensor.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(parallel_size):  # parallel_size is batch size of this generation call
            save_path = os.path.join(output_dir, "{}_{}.jpg".format(file_prefix, i))
            Image.fromarray(processed_imgs_for_saving[i]).save(save_path)
            print(f"Saved image to {save_path}")

    return visual_img_tensor


def generate_images_from_embeddings_loop(
        model: MultiModalityCausalLM,
        processor: VLChatProcessor,
        music_embeddings_list: list,  # List of Tensors, each (1, proj_seq_len, projector_output_dim)
        train_config: TrainConfig,  # To get sys_prompt and other configs
        output_dir: str = "generated_track_images",
        image_token_num: int = 576,  # Default: (384//16) * (384//16) = 24*24 = 576
        # cfg_weight and temperature are currently hardcoded in generate_sample
):
    """
    Generates and saves images for a list of provided music embeddings.

    Args:
        model: The MultiModalityCausalLM model.
        processor: The VLChatProcessor.
        music_embeddings_list (list): A list of music embeddings (output of the audio projection model).
                                      Each element should be a tensor of shape (1, seq_len, dim)
                                      on the correct device and dtype (e.g., bfloat16 on CUDA).
        train_config (TrainConfig): Configuration object, mainly for sys_prompt.
        output_dir (str): Directory where generated images will be saved.
        image_token_num (int): Number of image tokens to generate.
    """
    model.eval()  # Ensure model is in evaluation mode
    device = model.device  # Assuming the model has a .device attribute (e.g., 'cuda:0')

    # 1. Prepare system prompt embeddings (if sys_prompt is defined)
    system_prompt_embeds = None
    if train_config.sys_prompt:
        conversation = [
            {"role": "User", "content": train_config.sys_prompt},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",  # System prompt is part of the conversation here
        )
        prompt_ids = processor.tokenizer.encode(sft_format)
        prompt_ids = torch.LongTensor([prompt_ids]).to(device)  # Add batch dim
        system_prompt_embeds = model.language_model.get_input_embeddings()(prompt_ids).to(dtype=torch.bfloat16)
        # system_prompt_embeds shape: (1, num_prompt_tokens, hidden_dim)

    # 2. Prepare image_start_tag embeddings
    image_start_tag_ids = processor.tokenizer.encode(processor.image_start_tag)
    image_start_tag_ids = torch.LongTensor([image_start_tag_ids]).to(device)  # Add batch dim
    image_start_embeds = model.language_model.get_input_embeddings()(image_start_tag_ids).to(dtype=torch.bfloat16)
    # image_start_embeds shape: (1, num_start_tokens, hidden_dim)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting image generation. Output will be saved to: {output_dir}")

    for idx, music_emb_item in enumerate(tqdm(music_embeddings_list, desc="Generating images")):
        # Ensure music_emb_item is a tensor on the correct device and dtype
        if not isinstance(music_emb_item, torch.Tensor):
            # Attempt to convert, assuming it's a NumPy array or similar
            music_emb_item = torch.from_numpy(music_emb_item)

        music_emb_item = music_emb_item.to(device, dtype=torch.bfloat16)

        # Ensure batch dimension for music_emb_item: (1, seq_len, dim)
        if music_emb_item.ndim == 2:
            music_emb_item = music_emb_item.unsqueeze(0)
        elif music_emb_item.ndim != 3 or music_emb_item.shape[0] != 1:
            print(
                f"Warning: Music embedding at index {idx} has unexpected shape {music_emb_item.shape}. Expected (1, seq_len, dim). Skipping.")
            continue

        file_prefix_for_saving = f"track_{idx:03d}_img"

        # Call the modified generate_sample function
        # Note the argument order for generate_sample:
        # 1st arg (music_embedding in generate_sample) is for system prompt.
        # 2nd arg (batched_prompt_embeds in generate_sample) is for projected audio.
        with torch.no_grad():  # Important for inference
            _ = generate_sample(
                music_embedding=system_prompt_embeds,  # System prompt (can be None)
                batched_prompt_embeds=music_emb_item,  # Actual projected music embedding
                image_start_embeds=image_start_embeds,  # Image start tag
                image_token_num_per_image=image_token_num,
                processor=processor,
                model=model,
                file_prefix=file_prefix_for_saving,
                save_to_disk=True,
                output_dir=output_dir
            )
    print(f"Finished generating images. Saved in {output_dir}")


if __name__ == "__main__":
    # ... (your existing setup code for argument parsing, TrainConfig)

    # --- Initialize Models (similar to your existing code) ---
    # Load Audio Projection Model (if needed to generate embeddings first)
    projection = ImprovedAudioProjection(
        input_dim=TrainConfig.projector_input_dim,
        output_dim=TrainConfig.projector_output_dim,
        seq_len=TrainConfig.proj_seq_len,
        # ... other parameters ...
    ).to(TrainConfig.device)  # Use device from TrainConfig
    projection.eval()  # Set to evaluation mode

    # Load main MultiModality Model and Processor
    model_path = "deepseek-ai/Janus-1.3B"  # Or your model path
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).to(
        TrainConfig.device)
    vl_gpt.eval()  # Set to evaluation mode

    music_embeddings_list = []
    with torch.no_grad():
        for raw_features in raw_audio_features_list:
            # raw_features: e.g., tensor of shape [1, input_seq_len, projector_input_dim]
            raw_features = raw_features.to(TrainConfig.device)
            projected_embedding = projection(raw_features).to(torch.bfloat16) # Shape [1, proj_seq_len, projector_output_dim]
            music_embeddings_list.append(projected_embedding)

    dummy_embedding_1 = torch.randn(1, TrainConfig.proj_seq_len, TrainConfig.projector_output_dim,
                                    dtype=torch.bfloat16, device=TrainConfig.device)
    dummy_embedding_2 = torch.randn(1, TrainConfig.proj_seq_len, TrainConfig.projector_output_dim,
                                    dtype=torch.bfloat16, device=TrainConfig.device)
    music_embeddings_list_for_generation = [dummy_embedding_1, dummy_embedding_2]

    print(f"Number of embeddings to process: {len(music_embeddings_list_for_generation)}")
    print(f"Shape of first embedding: {music_embeddings_list_for_generation[0].shape}")

    # --- Call the generation loop ---
    generate_images_from_embeddings_loop(
        model=vl_gpt,
        processor=vl_chat_processor,
        music_embeddings_list=music_embeddings_list_for_generation,
        train_config=TrainConfig,  # Pass your TrainConfig class/instance
        output_dir="my_generated_images",  # Specify your desired output folder
        # image_token_num can be adjusted if needed, default is 576
    )

    # ... (rest of your script, if any)