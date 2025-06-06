import argparse
from typing import List

from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # Make sure tqdm is imported if not already

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from core.projection.projector import ImprovedAudioProjection
from core.training.scripts.train_val_loops import TrainConfig, freeze_model
from core.utils import generate_sample
from research.train_loop_experements import music_embedding


def generate_images_from_embeddings_loop(
        model: MultiModalityCausalLM,
        processor: VLChatProcessor,
        music_embeddings: torch.Torch,
        train_config: TrainConfig,  # To get sys_prompt and other configs
        output_dir: str = "generated_track_images",
        image_token_num: int = 576,  # Default: (384//16) * (384//16) = 24*24 = 576
        music_titles: List[str] = None,
        metric_logger: wandb.sdk.wandb_run.Run = None,
        # cfg_weight and temperature are currently hardcoded in generate_sample
):

    model.eval()  # Ensure model is in evaluation mode
    device = model.device  # Assuming the model has a .device attribute (e.g., 'cuda:0')

    # 1. Prepare system prompt embeddings (if sys_prompt is defined)
    prompt_embeds = None
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
        prompt_embeds = model.language_model.get_input_embeddings()(prompt_ids).to(dtype=torch.bfloat16).unsqueeze(0)

    # 2. Prepare image_start_tag embeddings
    image_start_tag_ids = processor.tokenizer.encode(processor.image_start_tag)
    image_start_tag_ids = torch.LongTensor([image_start_tag_ids]).to(device)  # Add batch dim
    image_start_embeds = model.language_model.get_input_embeddings()(image_start_tag_ids).to(dtype=torch.bfloat16).unsqueeze(0)
    # image_start_embeds shape: (1, num_start_tokens, hidden_dim)

    B = music_embeddings.shape[0]
    batched_image_start_embeds = image_start_embeds.repeat(B, 1, 1)
    if prompt_embeds is not None:
        batched_prompt_embeds = prompt_embeds.repeat(music_embeddings.shape[0], 1, 1)

    with torch.no_grad():
        visual_img = generate_sample(batched_prompt_embeds, music_embeddings, batched_image_start_embeds,
                                     image_token_num, processor, model,
                                     file_prefix=f"val_set_")
    wandb_images = []
    for i, img in enumerate(visual_img):
        wandb_image = wandb.Image(img, caption=f"{music_titles[i]}" if music_titles else f"Image {i}")
        wandb_images.append(wandb_image)

    metric_logger.log({"generated_images": wandb_images})

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting image generation. Output will be saved to: {output_dir}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_path", type=str, default=None, help="Path to the saved projection model")
    args = parser.parse_args()

    projection = ImprovedAudioProjection(
        input_dim=TrainConfig.projector_input_dim,
        output_dim=TrainConfig.projector_output_dim,
        seq_len=TrainConfig.proj_seq_len,
        num_layers=TrainConfig.proj_num_layers,
        dropout=TrainConfig.proj_dropout,
        activation=TrainConfig.proj_activation,
        use_l2=TrainConfig.proj_use_l2,
        scale_up=TrainConfig.proj_scale_up
    ).cuda()

    if args.proj_path and os.path.exists(args.proj_path):
        projection.load_state_dict(torch.load(args.proj_path))
        print(f"Loaded projection model from {args.proj_path}")

    print(TrainConfig.get_attributes())
    model_path = "deepseek-ai/Janus-1.3B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()

    freeze_model(vl_gpt)

    df = pd.read_pickle("data/val_set.pkl")
    music_embedding = np.stack(df['embeddings'].to_numpy())
    music_embedding_tensor = torch.tensor(music_embedding, dtype=torch.bfloat16).cuda()
    music_embedding_tensor = music_embedding_tensor.squeeze(0)


    print(f"Shape of music embedding: music_embedding_tensor")

    # --- Call the generation loop ---
    with wandb.init(project="musesthai-demo", config=TrainConfig.get_attributes()) as metric_logger:
        generate_images_from_embeddings_loop(
            model=vl_gpt,
            processor=vl_chat_processor,
            music_embeddings=music_embedding_tensor,
            train_config=TrainConfig,
            music_titles = df['audio_path'].apply(lambda x: x.split("/")[-1].split(".")[0]).tolist(),
            output_dir="my_generated_images",
            metric_logger=metric_logger,
        )
