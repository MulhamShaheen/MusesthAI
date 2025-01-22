import os

import pandas as pd

from embeddings.embedder import ImageBindEmbedder


def create_images_dataset(images_dir: str, count: int = 10) -> pd.DataFrame:
    image_files = os.listdir(images_dir)
    image_files = [f for f in image_files if f.endswith(".jpg")]

    step = int(len(image_files) / count)
    image_files = image_files[::step]
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    print(f"Number of images: {len(image_files)}")
    for i, f in enumerate(image_files):
        print(f"{i}: {f}")

    df = pd.DataFrame(image_files, columns=["image_path"], index=range(len(image_files)))

    embedder = ImageBindEmbedder()
    embeddings = embedder.embed_image(image_paths)
    df["embeddings"] = embeddings.tolist()

    df.to_csv("../data/images_dataset.csv", index=False)
    return df


def create_audio_dataset(audio_dir: str, count: int = 10) -> pd.DataFrame:
    audio_files = os.listdir(audio_dir)

    step = int(len(audio_files) / count)
    step = max(step, 1)
    audio_files = audio_files[::step]
    audio_paths = [os.path.join(audio_dir, f) for f in audio_files]
    print(f"Number of audio files: {len(audio_files)}")
    for i, f in enumerate(audio_files):
        print(f"{i}: {f}")

    df = pd.DataFrame(audio_files, columns=["audio_path"], index=range(len(audio_files)))

    embedder = ImageBindEmbedder()
    embeddings = embedder.embed_audio(audio_paths)
    df["embeddings"] = embeddings.tolist()

    df.to_csv("../data/audio_dataset.csv", index=False)
    return df


create_audio_dataset("../data/music", count=1)
