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


def batch_create_images_dataset(images_dir: str, output_dir: str, batch_size: int = 10, count: int = 100, offset: int = 0):
    image_files = os.listdir(images_dir)
    image_files = [f for f in image_files if f.endswith(".jpg")][offset:]
    embedded_count = 0
    image_paths = [os.path.join(images_dir, f) for f in image_files][offset: offset + count]
    embeddings = []
    print(f"Number of images: {len(image_files)}")
    embedder = ImageBindEmbedder()
    output_path = os.path.join(output_dir, f"images_dataset_offset_{offset}.pkl")
    for i in range(0, min(len(image_files), count), batch_size):
        if embedded_count + batch_size >= count:
            break
        batch_image_files = image_files[i:i + batch_size]
        batch_image_paths = [os.path.join(images_dir, f) for f in batch_image_files]
        for j, f in enumerate(batch_image_files):
            print(f"{j + i}: {f}")

        batch_embeddings = embedder.embed_image(batch_image_paths)
        embeddings.extend(batch_embeddings)
        embedded_count += len(batch_embeddings)
        temp_df = pd.DataFrame({"image_path": image_paths[:embedded_count], "embeddings": embeddings[:embedded_count]}, columns=["image_path", "embeddings"],
                      index=range(embedded_count))
        temp_df.to_pickle(output_path.replace(".pkl", "_temp.pkl"))


    df = pd.DataFrame({"image_path": image_paths[:count], "embeddings": embeddings}, columns=["image_path", "embeddings"],
                      index=range(count))
    df.to_pickle(output_path)


def create_audio_dataset(audio_dir: str, output_path: str, count: int = 10) -> pd.DataFrame:
    audio_paths = []
    audio_count = 0
    sub_dirs = os.listdir(audio_dir)
    for sub_dir in sub_dirs:
        if audio_count >= count:
            break
        if not os.path.isdir(os.path.join(audio_dir, sub_dir)):
            continue
        sub_dir_path = os.path.join(audio_dir, sub_dir)
        for audio_file in os.listdir(sub_dir_path):
            if audio_count >= count:
                break
            if not audio_file.endswith(".mp3"):
                continue

            audio_paths.append(os.path.join(sub_dir_path, audio_file))
            audio_count += 1

    df = pd.DataFrame({"audio_path": audio_paths}, columns=["audio_path"],
                      index=range(len(audio_paths)))

    df.to_csv(output_path, index=False)
    return df


def batch_create_audio_embeddings(audio_df_path: str, output_dir: str, batch_size: int = 10, count: int = 100, offset: int = 0):
    audio_df = pd.read_csv(audio_df_path)
    audio_paths = audio_df["audio_path"].tolist()[offset: offset + count]
    embeddings = []
    print(f"Number of audio files: {len(audio_paths)}")
    embedder = ImageBindEmbedder()
    output_path = os.path.join(output_dir, f"audio_dataset_offset_{offset}.pkl")
    embedded_count = 0
    for i in range(0, min(len(audio_paths), count), batch_size):
        if embedded_count + batch_size >= count:
            break
        batch_audio_paths = audio_paths[i:i + batch_size]
        for j, f in enumerate(batch_audio_paths):
            print(f"{j + i}: {f}")

        batch_embeddings = embedder.embed_audio(batch_audio_paths)
        embeddings.extend(batch_embeddings)
        embedded_count += len(batch_embeddings)
        temp_df = pd.DataFrame({"audio_path": audio_paths[:embedded_count], "embeddings": embeddings[:embedded_count]},
                               columns=["audio_path", "embeddings"],
                               index=range(embedded_count))
        temp_df.to_pickle(output_path.replace(".pkl", "_temp.pkl"))

    df = pd.DataFrame({"audio_path": audio_paths[:embedded_count], "embeddings": embeddings},
                      columns=["audio_path", "embeddings"],
                      index=range(count))
    df.to_pickle(output_path)