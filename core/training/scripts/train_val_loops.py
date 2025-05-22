import argparse
import os
import random

import accelerate
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm.auto import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from accelerate import Accelerator
from torch.optim import Adam, AdamW
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from core.projection import AudioProjection
import wandb

from core.projection.projector import ImprovedAudioProjection

wandb.login(key="af7a0ee3c846dd80ff62ec27a2046eb9b809b646")


def generate_sample(music_embedding, batched_prompt_embeds, image_start_embeds, image_token_num_per_image, processor, model, file_prefix):
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


class ImageAudioDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        current_path = os.getcwd()

        self.dataframe["image_path"] = self.dataframe["image_path"].apply(
            lambda x: os.path.abspath(
                os.path.join(current_path, "data/images/" + x.split("/images/")[-1].replace("\\", "/"))))

        self.dataframe["audio_path"] = self.dataframe["audio_path"].apply(
            lambda x: os.path.abspath(
                os.path.join(current_path, "data/music/" + x.split("/music")[-1].replace("\\", "/"))))
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx) -> dict:
        image_path = self.dataframe.iloc[idx]["image_path"]
        audio_path = self.dataframe.iloc[idx]["audio_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)
            image = nn.functional.interpolate(image.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False)

        image_embedding = self.dataframe.iloc[idx]["image_embedding"]
        music_embedding = self.dataframe.iloc[idx]["music_embedding"]
        return {
            "audio_path": audio_path,
            "image": image,
            "music_embedding": music_embedding,
            "image_embedding": image_embedding,
        }


def get_collate_fn(gen_model: MultiModalityCausalLM):
    def collate_fn(items):
        result = dict()
        with torch.no_grad():
            images = torch.cat([item["image"] for item in items], dim=0)
            quant, _, info = gen_model.gen_vision_model.encode(
                images.to(dtype=torch.bfloat16).cuda())
            B, C, Hq, Wq = quant.shape
            _, _, min_encoding_indices = info
            image_ids = min_encoding_indices.view(B, Hq * Wq)
            gen_embeds = gen_model.prepare_gen_img_embeds(image_ids)

        result["image_ids"] = image_ids.squeeze(-1)
        result["image_gen_embeds"] = gen_embeds
        result["music_embedding"] = torch.stack([torch.from_numpy(item["music_embedding"]) for item in items], dim=0)
        result["images"] = images
        result["audio_paths"] = [item["audio_path"] for item in items]

        return result

    return collate_fn


def post_process_batch(gen_model, batch):
    result = dict()
    with torch.no_grad():
        images = torch.cat([image for image in batch["image"]], dim=0)
        quant, _, info = gen_model.gen_vision_model.encode(
            images.to(dtype=torch.bfloat16).cuda())
        B, C, Hq, Wq = quant.shape
        _, _, min_encoding_indices = info
        image_ids = min_encoding_indices.view(B, Hq * Wq)
        gen_embeds = gen_model.prepare_gen_img_embeds(image_ids)

    result["image_ids"] = image_ids.squeeze(-1)
    result["image_gen_embeds"] = gen_embeds
    result["music_embedding"] = torch.stack([item for item in batch["music_embedding"]], dim=0)
    result["images"] = images
    result["audio_paths"] = [item for item in batch["audio_path"]]

    return result


def train_loop(accelerator, model, processor, projection, optimizer, train_dataloader, epoch, criterion, train_config,
               metric_logger: wandb.sdk.wandb_run.Run,
               mock_run: bool = False, generate_freq=20, metrics: dict = None,):
    model.eval()
    projection.train()
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch}")
    total_loss = 0
    num_batches = 0
    prompt_embeds = None

    if train_config.sys_prompt is not None:
        conversation = [
            {
                "role": "User",
                "content": train_config.sys_prompt,
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format

        prompt_ids = processor.tokenizer.encode(prompt)
        prompt_ids = torch.LongTensor(prompt_ids).cuda()
        prompt_embeds = model.language_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16).unsqueeze(0)

    image_start_tag_ids = processor.tokenizer.encode(processor.image_start_tag)
    image_start_tag_ids = torch.LongTensor(image_start_tag_ids).cuda()
    image_start_embeds = model.language_model.get_input_embeddings()(image_start_tag_ids).to(
        torch.bfloat16).unsqueeze(0)

    for batch in train_dataloader:
        batch = post_process_batch(model, batch)

        with accelerator.accumulate(projection):
            music_embedding = batch["music_embedding"].to(model.device)
            audio_input = projection(music_embedding).to(torch.bfloat16)
            B = audio_input.shape[0]


            image_ids = batch["image_ids"]
            image_gen_embeds = batch["image_gen_embeds"].to(torch.bfloat16)

            batched_image_start_embeds = image_start_embeds.repeat(B, 1, 1)
            if prompt_embeds is not None:
                batched_prompt_embeds = prompt_embeds.repeat(B, 1, 1)
                input_embeds = torch.concat(
                    [batched_prompt_embeds, audio_input, image_gen_embeds], dim=1)

            else:
                input_embeds = torch.concat([audio_input, image_gen_embeds], dim=1)

            if mock_run:
                hidden_states = torch.rand(input_embeds.shape).cuda().to(torch.bfloat16)
            else:
                outputs = model.language_model.model(inputs_embeds=input_embeds, use_cache=False, past_key_values=None)
                hidden_states = outputs.last_hidden_state

            logits = model.gen_head(hidden_states)

            loss = criterion(logits[:, -576:-1, :].flatten(0, 1), image_ids[:, 1:].flatten(0, 1))

            total_loss += loss.item()
            step_metrics = {
                "train_loss": loss.item(),
                "epoch": epoch,
            }

            # model.zero_grad()
            # projection.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if num_batches % generate_freq == 0:
            image_token_num = image_ids.shape[-1]
            rand_index = random.randint(0, audio_input.shape[0] - 1)
            audio_sample = audio_input[rand_index].unsqueeze(0)
            if prompt_embeds is not None:
                prompt_sample = batched_prompt_embeds[rand_index].unsqueeze(0)
            else:
                prompt_sample = None
            audio_input_float = audio_input.to(torch.float32).detach()

            audio_input_norm = torch.linalg.norm(audio_input_float, 'fro', dim=(-2, -1)).mean().item()
            text_input_norm = torch.linalg.norm(batched_prompt_embeds.float(), 'fro', dim=(-2, -1)).mean().item()

            with torch.no_grad():
                visual_img = generate_sample(prompt_sample, audio_sample, image_start_embeds,
                                             image_token_num, processor, model,
                                             file_prefix=f"train_epoch_{epoch}_batch_{num_batches}")

            wandb_image = wandb.Image(visual_img[0], caption=f"Epoch {epoch}, batch: {num_batches}, "
                                                             f"loss: {step_metrics['train_loss']} "
                                                             f"audio: {batch['audio_paths'][rand_index]}")
            step_metrics["Generated Example"] = wandb_image
            step_metrics["audio_input_norm"] = audio_input_norm
            step_metrics["text_input_norm"] = text_input_norm

            target_images = batch["images"].cuda()

            if target_images.dtype != torch.uint8:
                target_images = (target_images * 255.0).to(torch.uint8)
            if target_images.shape[1] != 3:
                target_images = target_images.permute(0, 3, 1, 2)

            if metrics is not None:
                if "fid" in metrics:
                    metrics["fid"].update(target_images, real=True)
                    metrics["fid"].update(visual_img, real=False)
                    try:
                        step_metrics["fid"] = metrics["fid"].compute().item()
                    except:
                        pass
                if "inception_score" in metrics:
                    metrics["inception_score"].update(visual_img)
                    try:
                        step_metrics["inception_score_mean"], step_metrics[
                            "inception_score_std"] = metrics["inception_score"].compute().item()
                    except:
                        pass

        num_batches += 1
        progress_bar.update(1)
        progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f}')

        metric_logger.log(step_metrics)

    average_loss = total_loss / len(train_dataloader)
    return average_loss


@torch.no_grad()
def val_loop(model, processor, projection, val_dataloader, train_config,
             metric_logger: wandb.sdk.wandb_run.Run,
             metrics: dict = None, epoch=1, no_loss=True,
             generate_freq=0, mock_run: bool = False):
    criterion = nn.CrossEntropyLoss(ignore_index=processor.pad_id)
    sumloss = 0
    num_batches = 0
    prompt_embeds = None

    if train_config.sys_prompt is not None:
        conversation = [
            {
                "role": "User",
                "content": train_config.sys_prompt,
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format

        prompt_ids = processor.tokenizer.encode(prompt)
        prompt_ids = torch.LongTensor(prompt_ids).cuda()
        prompt_embeds = model.language_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16).unsqueeze(0)

    image_start_tag_ids = processor.tokenizer.encode(processor.image_start_tag)
    image_start_tag_ids = torch.LongTensor(image_start_tag_ids).cuda()
    image_start_embeds = model.language_model.get_input_embeddings()(image_start_tag_ids).to(
        torch.bfloat16).unsqueeze(0)

    if not metrics:
        fid = FrechetInceptionDistance(feature=2048).to(model.device)
        inception_score = InceptionScore(feature='logits_unbiased', splits=10).to(model.device)
    else:
        fid = metrics.get('fid', None)
        inception_score = metrics.get('inception_score', None)


    if not generate_freq:
        generate_freq = len(val_dataloader) + 1

    model.eval()
    projection.eval()
    for batch in tqdm(val_dataloader):
        batch = post_process_batch(model, batch)

        batch_input_ids = batch['image_ids'].to(model.device)
        music_embedding = batch["music_embedding"].to(model.device)
        music_embedding = projection(music_embedding).to(torch.bfloat16)
        image_gen_embeds = batch["image_gen_embeds"].to(model.device).to(torch.bfloat16)

        B = music_embedding.shape[0]

        batched_image_start_embeds = image_start_embeds.repeat(B, 1, 1)

        if prompt_embeds is not None:
            batched_prompt_embeds = prompt_embeds.repeat(music_embedding.shape[0], 1, 1)

            input_embeds = torch.concat(
                [batched_prompt_embeds, music_embedding, image_gen_embeds], dim=1)
        else:
            input_embeds = torch.concat([music_embedding, image_gen_embeds], dim=1)

        if not no_loss:
            if mock_run:
                hidden_states = torch.rand(input_embeds.shape).cuda().to(torch.bfloat16)
            else:
                outputs = model.language_model.model(inputs_embeds=input_embeds, use_cache=False, past_key_values=None)
                hidden_states = outputs.last_hidden_state
            logits = model.gen_head(hidden_states)

            loss = criterion(logits[:, -576:-1, :].flatten(0, 1), batch_input_ids[:, 1:].flatten(0, 1))
            sumloss += loss.item()

        num_batches += 1

        val_res = {
            "loss": sumloss / num_batches if num_batches > 0 else 0,
            "num_batches": num_batches,
            # mock values
            "imagebind_sim": 0,
        }

        if num_batches % generate_freq == 0:
            rand_index = random.randint(0, batch_input_ids.shape[0] - 1)
            image_token_num = batch_input_ids.shape[-1]
            with torch.no_grad():
                visual_img = generate_sample(batched_prompt_embeds, music_embedding, batched_image_start_embeds,
                                             image_token_num, processor, model,
                                             file_prefix=f"epoch_{epoch}_batch_{num_batches}")
            wandb_image = wandb.Image(visual_img[rand_index], caption=f"Epoch {epoch}, batch: {num_batches}, "
                                                             f"loss: {val_res['loss']} "
                                                             f"audio: {batch['audio_paths'][rand_index]}")



            target_images = batch["images"].cuda()

            if target_images.dtype != torch.uint8:
                target_images = (target_images * 255.0).to(torch.uint8)
            if target_images.shape[1] != 3:
                target_images = target_images.permute(0, 3, 1, 2)

            wandb_target = wandb.Image(target_images[rand_index].cpu(), caption=f"Epoch {epoch}, batch: {num_batches}, "
                                                                            f"loss: {val_res['loss']} ")
            val_res["Generated Example"] = wandb_image
            val_res["Target Image"] = wandb_target

            if fid is not None:
                fid.update(target_images, real=True)
                fid.update(visual_img, real=False)
            if inception_score is not None:
                inception_score.update(visual_img)

            try:
                if fid is not None:
                    val_res["fid"] = fid.compute()
                if inception_score is not None:
                    val_res["inception_score_mean"], val_res["inception_score_std"] = inception_score.compute()
            except RuntimeError as e:
                val_res["fid"] = 0
                val_res["inception_score_mean"], val_res["inception_score_std"] = 0, 0

        val_res = {f"val_{k}": v for k, v in val_res.items()}
        metric_logger.log(val_res)

    return val_res


class TrainConfig:
    log_level = "DEBUG"

    num_epochs = 30
    train_batch_size = 60
    val_batch_size = 10
    log_grad_norm = True
    learning_rate = 1e-4
    gradient_accumulation_steps = 1

    evaluate_every_epoch_mod = 3
    save_model_every_epoch_mod = 5
    device = "cuda:0"

    projector_input_dim = 1024
    projector_output_dim = 2048
    proj_seq_len = 24
    proj_num_layers = 3
    proj_dropout = 0.1
    proj_activation = 'gelu'
    proj_use_l2 = True
    proj_scale_up = 3

    gen_freq = 110

    few_val_samples = 60
    dataloader_num_workers = 0
    dataset_name = "matched_dataset_0_15.pkl"
    val_dataset_name = "val_dataset.pkl"

    sys_prompt = "Detailed art that contains abstract figures"
    # sys_prompt = None

    @classmethod
    def get_attributes(cls):
        return {key: value for key, value in cls.__dict__.items() if not key.startswith("__") and not callable(value)
                and key != "get_attributes"}


def train(
        model: MultiModalityCausalLM,
        projection: AudioProjection,
        processor: VLChatProcessor,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_config: TrainConfig,
        metric_logger: wandb.sdk.wandb_run.Run,
        device_placement=True,
        mock_run: bool = False,
):
    best_fid = 0

    trainable_parameters = list(projection.parameters())
    optimizer = AdamW(trainable_parameters, lr=train_config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=processor.image_start_id)

    accelerator = accelerate.Accelerator(device_placement=device_placement, log_with="wandb")
    accelerator.gradient_accumulation_steps = train_config.gradient_accumulation_steps

    projection, optimizer, train_dataloader, val_dataloader = accelerator.prepare(projection, optimizer,
                                                                                  train_dataloader,
                                                                                  val_dataloader)
    metrics = {
        "fid": FrechetInceptionDistance(feature=192).to(model.device),
        "inception_score": InceptionScore(feature='logits_unbiased', splits=10).to(model.device)
    }

    for epoch in range(train_config.num_epochs):
        train_loss = train_loop(accelerator, model, processor, projection, optimizer, train_dataloader, epoch=epoch,
                                criterion=criterion, metrics=metrics,
                                train_config=train_config, mock_run=mock_run, metric_logger=metric_logger,
                                generate_freq=train_config.gen_freq)
        metric_logger.log({"epoch_loss": train_loss})
        if epoch % train_config.save_model_every_epoch_mod == 0:
            torch.save(projection.state_dict(), f"proj_seq_{TrainConfig.proj_seq_len}_epoch_{epoch}_projection.pt")
            print(f"Model saved at epoch {epoch}")

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            print("Evaluating model for epoch: ", epoch)
            validation_metrics = val_loop(model, processor, projection, val_dataloader, train_config=train_config,
                                          epoch=epoch, metrics=metrics, no_loss=False, metric_logger=metric_logger,
                                          generate_freq=1, mock_run=mock_run)
            final_fid_score = validation_metrics["fid"]
            print(f"Epoch {epoch} validation metrics: {validation_metrics}")

            if final_fid_score < best_fid or best_fid == 0:
                best_fid = final_fid_score
                print("New best fid: ", best_fid)

    torch.save(projection.state_dict(), f"proj_seq_{TrainConfig.proj_seq_len}_fid_{best_fid}.pt")

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return


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

    # Example dataset
    import pandas as pd

    matched_df_path = os.path.join(os.getcwd(), f"data/{TrainConfig.dataset_name}")
    val_df_path = os.path.join(os.getcwd(), f"data/{TrainConfig.val_dataset_name}")
    matched_df = pd.read_pickle(matched_df_path)
    val_df = pd.read_pickle(val_df_path)

    dataset = ImageAudioDataset(matched_df)
    val_dataset = ImageAudioDataset(val_df)

    train_dataloader = DataLoader(dataset, batch_size=TrainConfig.train_batch_size, shuffle=True,
                                  num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=TrainConfig.val_batch_size, shuffle=False,
                                num_workers=6, pin_memory=True)

    with wandb.init(project="musesthai", config=TrainConfig.get_attributes()) as metric_logger:
        train(
            model=vl_gpt,
            projection=projection,
            processor=vl_chat_processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_config=TrainConfig,
            mock_run=False,
            metric_logger=metric_logger
        )
