import os

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
from torch.optim import Adam
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from core.projection import AudioProjection
import wandb


class ImageAudioDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        current_path = os.getcwd()

        self.dataframe["image_path"] = self.dataframe["image_path"].apply(
            lambda x: os.path.abspath(os.path.join(current_path, "data/" + x.split("data")[-1].replace("\\", "/"))))

        self.dataframe["audio_path"] = self.dataframe["audio_path"].apply(
            lambda x: os.path.abspath(os.path.join(current_path, "data/" + x.split("data")[-1].replace("\\", "/"))))
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

        return result

    return collate_fn


def train_loop(accelerator, model, projection, optimizer, train_dataloader, epoch, criterion, train_config,
               metric_logger: wandb.sdk.wandb_run.Run,
               mock_run: bool = False):
    model.eval()
    projection.train()
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch}")
    total_loss = 0
    for batch in train_dataloader:
        with accelerator.accumulate(projection):
            music_embedding = batch["music_embedding"].to(model.device)
            audio_input = projection(music_embedding).to(torch.bfloat16)
            image_gen_embeds = batch["image_gen_embeds"].to(torch.bfloat16)
            image_ids = batch["image_ids"]

            input_embeds = torch.concat([audio_input, image_gen_embeds], dim=1)
            if mock_run:
                hidden_states = torch.rand(input_embeds.shape).cuda().to(torch.bfloat16)
            else:
                outputs = model.language_model.model(inputs_embeds=input_embeds, use_cache=False, past_key_values=None,
                                                     decoder_input_ids=1)
                hidden_states = outputs.last_hidden_state

            logits = model.gen_head(hidden_states)
            logits = logits.permute(0, 2, 1)
            loss = criterion(logits[:, :, -576:], image_ids)
            total_loss += loss.item()
            step_metrics = {"train_loss": loss.item(), "epoch": epoch}
            metric_logger.log(step_metrics)

            model.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f}')

    average_loss = total_loss / len(train_dataloader)
    return average_loss


@torch.no_grad()
def val_loop(model, processor, projection, val_dataloader, metrics: dict = None, epoch=1, no_loss=False,
             generate_freq=0, mock_run: bool = False):
    criterion = nn.CrossEntropyLoss(ignore_index=processor.pad_id)
    sumloss = 0
    num_batches = 0
    if not metrics:
        fid = FrechetInceptionDistance(feature=2048).to(model.device)
        inception_score = InceptionScore(feature='logits_unbiased', splits=10).to(model.device)
    else:
        fid = metrics['fid']
        inception_score = metrics['inception_score']

    # hardcoded values
    cfg_weight = 5
    temperature = 1
    img_size = 384
    patch_size = 16

    if not generate_freq:
        generate_freq = len(val_dataloader) + 1

    model.eval()
    projection.eval()
    for batch in tqdm(val_dataloader):
        batch_input_ids = batch['image_ids'].to(model.device)
        music_embedding = batch["music_embedding"].to(model.device)
        music_embedding = projection(music_embedding).to(torch.bfloat16)
        image_gen_embeds = batch["image_gen_embeds"].to(model.device).to(torch.bfloat16)

        input_embeds = torch.concat([music_embedding, image_gen_embeds], dim=1)

        if not no_loss:
            if mock_run:
                hidden_states = torch.rand(input_embeds.shape).cuda().to(torch.bfloat16)
            else:
                outputs = model.language_model.model(inputs_embeds=input_embeds, use_cache=False, past_key_values=None,
                                                     decoder_input_ids=1)
                hidden_states = outputs.last_hidden_state
            logits = model.gen_head(hidden_states)
            logits = logits.permute(0, 2, 1)
            loss = criterion(logits[:, :, -576:].cpu(), batch_input_ids.cpu())
            sumloss += loss.item()

        num_batches += 1

        # generate images and metrics
        if num_batches % generate_freq == 0:
            parallel_size = music_embedding.shape[0]
            unconditional_tokens = torch.zeros((1, input_embeds.shape[-2]), dtype=torch.int).cuda()
            unconditional_tokens[0, 1:-1] = processor.pad_id
            unconditional_embeds = model.language_model.get_input_embeddings()(unconditional_tokens)
            gen_input_embeds = torch.zeros((input_embeds.shape[0] * 2, *input_embeds.shape[1:]),
                                           dtype=torch.bfloat16).cuda()

            for i in range(parallel_size * 2):
                if i % 2 != 0:
                    gen_input_embeds[i] = unconditional_embeds
                else:
                    gen_input_embeds[i] = input_embeds[i // 2]

            generated_tokens = torch.zeros((parallel_size, batch_input_ids.shape[-1]), dtype=torch.int).cuda()
            inputs_embeds = gen_input_embeds
            for i in range(batch_input_ids.shape[-1]):
                if mock_run:
                    hidden_states = torch.rand(input_embeds.shape).cuda().to(torch.bfloat16)
                else:
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
                save_path = os.path.join('generated_samples', "img_{}_epoch_{}_batch_{}.jpg".format(i, epoch,
                                                                                                    num_batches))

                Image.fromarray(visual_img[i]).save(save_path)

            visual_img = dec.to(torch.uint8)
            target_images = batch["images"].cuda()

            if target_images.dtype != torch.uint8:
                target_images = (target_images * 255.0).to(torch.uint8)
            if target_images.shape[1] != 3:
                target_images = target_images.permute(0, 3, 1, 2)

            fid.update(target_images, real=True)
            fid.update(visual_img, real=False)
            inception_score.update(visual_img)


    val_res = {
        "loss": sumloss / num_batches if num_batches > 0 else 0,
        "num_batches": num_batches,
        # mock values
        "imagebind_sim": 0,
    }

    try:
        val_res["fid"] = fid.compute()
        val_res["inception_score_mean"], val_res["inception_score_std"] = inception_score.compute()
    except RuntimeError as e:
        val_res["fid"] = 0
        val_res["inception_score_mean"], val_res["inception_score_std"] = 0, 0

    return val_res

class TrainConfig:
    log_level = "DEBUG"

    num_epochs = 2
    train_batch_size = 20
    val_batch_size = 5
    log_grad_norm = True
    learning_rate = 1e-4
    gradient_accumulation_steps = 1

    evaluate_every_epoch_mod = 1
    save_model_every_epoch_mod = 1
    device = "cuda:0"

    # Projector
    projector_input_dim = 1024
    proj_seq_len = 128

    few_val_samples = 20
    dataloader_num_workers = 0

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
    metric_logger

    trainable_parameters = list(projection.parameters())
    optimizer = Adam(trainable_parameters, lr=train_config.learning_rate)
    criterion = nn.CrossEntropyLoss()

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
        train_loss = train_loop(accelerator, model, projection, optimizer, train_dataloader, epoch=epoch, criterion=criterion,
                   train_config=train_config, mock_run=mock_run, metric_logger=metric_logger)

        wandb.log({"train/batch_loss": train_loss}, step=epoch)
        if epoch % train_config.save_model_every_epoch_mod == 0:
            accelerator.save_state(f"proj_seq_{TrainConfig.proj_seq_len}_epoch_{epoch}.pt")
            wandb.save(f"model_epoch_{epoch}.pt/pytorch_model_0.pt")
            print(f"Model saved at epoch {epoch}")

        if epoch % train_config.evaluate_every_epoch_mod == 0:
            print("Evaluating model for epoch: ", epoch)
            validation_metrics = val_loop(model, processor, projection, val_dataloader, epoch=epoch, metrics=metrics,
                                          generate_freq=1, mock_run=mock_run)
            final_fid_score = validation_metrics["fid"]
            print(f"Epoch {epoch} validation metrics: {validation_metrics}")

            metric_logger.log(validation_metrics)

            if final_fid_score < best_fid or best_fid == 0:
                best_fid = final_fid_score
                print("New best fid: ", best_fid)


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return


if __name__ == "__main__":
    print(TrainConfig.get_attributes())
    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()

    freeze_model(vl_gpt)

    # Example dataset
    import pandas as pd

    matched_df_path = os.path.join(os.getcwd(), "data/matched_dataset_concat.pkl")
    matched_df = pd.read_pickle(matched_df_path)
    dataset = ImageAudioDataset(matched_df)
    val_dataset = ImageAudioDataset(matched_df.sample(n=TrainConfig.few_val_samples, random_state=42))

    # Dataloaders
    train_dataloader = DataLoader(dataset, batch_size=TrainConfig.train_batch_size, shuffle=True, collate_fn=get_collate_fn(vl_gpt))
    val_dataloader = DataLoader(val_dataset, batch_size=TrainConfig.val_batch_size, shuffle=False, collate_fn=get_collate_fn(vl_gpt))

    # Projection model
    projection = AudioProjection(1024, 2048, scale_factor=2, sequal_len=TrainConfig.proj_seq_len).cuda()

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
