import ast
import json
from typing import List

import numpy as np
import pandas as pd

from embeddings.embedder import ImageBindEmbedder
from embeddings.scorer import EmbeddingScorer

GENRES_DICT = json.load(open("../data/text/genres.json", "r"))

def create_genre_text_embeddings(genres: List[str] = None, save_file: bool = False) -> pd.DataFrame:
    if genres is None:
        genres = GENRES_DICT.keys()
    embedder = ImageBindEmbedder()
    embeddings = embedder.embed_text([GENRES_DICT[genre] for genre in genres])

    df = pd.DataFrame({"genre": genres, "embeddings": embeddings.tolist()}, columns=["genre", "embeddings"],
                      index=range(len(genres)))

    if save_file:
        df.to_csv("../data/text/genre_embeddings.csv", index=False)

    return df


def match_images_with_genres(images_embeddings_df: pd.DataFrame, genre_embeddings_df: pd.DataFrame) -> pd.DataFrame:
    images_embeddings = images_embeddings_df["embeddings"].tolist()
    images_embeddings = [np.array(ast.literal_eval(e)) for e in images_embeddings]

    genre_embeddings = genre_embeddings_df["embeddings"].tolist()
    genre_embeddings = [np.array(ast.literal_eval(e)) for e in genre_embeddings]

    scorer = EmbeddingScorer()
    scores = []
    genres = []
    image_paths = []
    for i, image_embedding in enumerate(images_embeddings):
        images_top_genres, images_top_scores = scorer.find_topk(image_embedding.reshape(1, -1),
                                                                np.array(genre_embeddings))
        scores.extend(images_top_scores)
        genres.extend([genre_embeddings_df.iloc[i]["genre"] for i in images_top_genres])
        image_paths.extend([images_embeddings_df.iloc[i]["image_path"]] * len(images_top_genres))

    data_dict = {
        "image_path": image_paths,
        "genre": genres,
        "score": scores
    }
    scores_df = pd.DataFrame(data_dict)

    scores_df.to_csv("../data/images_genre_scores.csv", index=False)
    return scores_df
