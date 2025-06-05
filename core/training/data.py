import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


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

