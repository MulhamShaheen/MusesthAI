import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from accelerate import Accelerator
from torch.optim import Adam
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor


class ImageAudioDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.dataframe["image_path"] = self.dataframe["image_path"].apply(
            lambda x: os.path.abspath(x.replace("\\", "/")))
        self.dataframe["audio_path"] = self.dataframe["audio_path"].apply(
            lambda x: os.path.abspath(x.replace("\\", "/")))
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
            images = torch.stack([item["image"] for item in items], dim=1)
            quant, _, info = gen_model.gen_vision_model.encode(
                images.squeeze(0).to(dtype=torch.bfloat16).cuda())
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


def train_loop(accelerator, model, projection, optimizer, train_dataloader, epoch, criterion, train_config):
    model.eval()
    projection.train()
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch}")
    for batch in train_dataloader:
        with accelerator.accumulate(projection):
            audio_input = projection(batch["music_embedding"]).to(torch.bfloat16)
            image_gen_embeds = batch["image_gen_embeds"].to(torch.bfloat16)
            image_ids = batch["image_ids"]

            input_embeds = torch.concat([audio_input, image_gen_embeds], dim=1)
            outputs = model.language_model.model(inputs_embeds=input_embeds, use_cache=False, past_key_values=None,
                                                 decoder_input_ids=1)
            hidden_states = outputs.last_hidden_state
            logits = model.gen_head(hidden_states)
            logits = logits.permute(0, 2, 1)
            loss = criterion(logits[:, :, -576:], image_ids)
            model.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_description(f'Epoch={epoch} Loss={loss.item():.3f}')


@torch.no_grad()
def val_loop(model, processor, projection, val_dataloader, epoch=1, no_loss=False, generate_freq=0):
    criterion = nn.CrossEntropyLoss(ignore_index=processor.pad_id)
    sumloss = 0
    num_batches = 0
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
        music_embedding = projection(batch["music_embedding"]).to(torch.bfloat16).cuda()
        image_gen_embeds = batch["image_gen_embeds"].to(torch.bfloat16)
        input_embeds = torch.concat([music_embedding, image_gen_embeds], dim=1)

        if not no_loss:
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

            for i in range(batch_input_ids.shape[-1]):
                outputs = model.language_model.model(inputs_embeds=gen_input_embeds, use_cache=True,
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
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            # get Fréchet Inception Distance
            # get Inception Score
            # get ImageBind similarity?

    val_res = {
        "loss": sumloss / num_batches if num_batches > 0 else 0,
        "num_batches": num_batches,
        # mock values
        "fid": 0,
        "is": 0,
        "imagebind_sim": 0,
    }

    return val_res


if __name__ == "__main__":
    # Example usage
    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()

    # Example dataset
    import pandas as pd
    matched_df_path = "../data/notebooks/matched_dataset_concat.pkl"
    matched_df = pd.read_pickle(matched_df_path)
    dataset = ImageAudioDataset(matched_df)

    # Dataloaders
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=get_collate_fn(vl_gpt))
    val_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=get_collate_fn(vl_gpt))

    # Projection model
    projection = nn.Linear(1024, 2048)

    # Training and validation
    accelerator = Accelerator()
    optimizer = Adam(projection.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loop(accelerator, vl_gpt, projection, optimizer, train_dataloader, epoch=1, criterion=criterion, train_config=None)
    val_loop(vl_gpt, vl_chat_processor, projection, val_dataloader, epoch=1)