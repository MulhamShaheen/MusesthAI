import os

import pandas as pd

from embeddings.embedder import ImageBindEmbedder


def create_images_dataset(images_dir: str, output_path: str = "../data/images_dataset.csv",
                          count: int = 10) -> pd.DataFrame:
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

    df.to_csv(output_path, index=False)
    return df


def batch_create_images_dataset(images_dir: str, output_dir: str, batch_size: int = 10, count: int = 100):
    image_files = os.listdir(images_dir)
    image_files = [f for f in image_files if f.endswith(".jpg")]
    embedded_count = 0
    image_paths = [os.path.join(images_dir, f) for f in image_files][:count]
    embeddings = []
    print(f"Number of images: {len(image_files)}")
    embedder = ImageBindEmbedder()
    output_path = os.path.join(output_dir, f"images_dataset.pkl")
    for i in range(0, min(len(image_files), count), batch_size):
        if embedded_count >= count:
            break
        batch_image_files = image_files[i:i + batch_size]
        batch_image_paths = [os.path.join(images_dir, f) for f in batch_image_files]
        for j, f in enumerate(batch_image_files):
            print(f"{j}: {f}")

        batch_embeddings = embedder.embed_image(batch_image_paths)
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"image_path": image_paths[:count], "embeddings": embeddings}, columns=["image_path", "embeddings"],
                      index=range(count))
    df.to_pickle(output_path)


def create_audio_dataset(audio_dir: str, count: int = 10) -> pd.DataFrame:
    genres = []
    audio_paths = []

    genres_dir = os.listdir(audio_dir)
    for genre_dir in genres_dir:
        if not os.path.isdir(os.path.join(audio_dir, genre_dir)):
            continue
        genre_dir_path = os.path.join(audio_dir, genre_dir)
        genre_audio_files = os.listdir(genre_dir_path)
        audio_paths.extend([os.path.join(genre_dir_path, f) for f in genre_audio_files])
        genres.extend([genre_dir] * len(genre_audio_files))
        print(f"Number of audio files in {genre_dir}: {len(audio_paths)}")
        for i, f in enumerate(audio_paths):
            print(f"{i}: {f}")

    df = pd.DataFrame({"audio_path": audio_paths, "genre": genres}, columns=["audio_path", "genre"],
                      index=range(len(audio_paths)))

    embedder = ImageBindEmbedder()
    embeddings = embedder.embed_audio(audio_paths)
    df["embeddings"] = embeddings.tolist()

    df.to_csv("../data/audio_dataset.csv", index=False)
    return df
