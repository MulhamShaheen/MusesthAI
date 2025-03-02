import ast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def match_datasets(image_embeddings_df: pd.DataFrame, music_embeddings_df: pd.DataFrame) -> pd.DataFrame:
    image_embeddings = image_embeddings_df[["embeddings"]]
    music_embeddings = music_embeddings_df[["embeddings"]]

    image_embeddings = np.array([np.array(ast.literal_eval(e)) for e in image_embeddings["embeddings"].tolist()])
    music_embeddings = np.array([np.array(ast.literal_eval(e)) for e in music_embeddings["embeddings"].tolist()])

    similarity_matrix = cosine_similarity(image_embeddings, music_embeddings)

    cost_matrix = 1 - similarity_matrix
    image_ind, music_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = pd.DataFrame()
    matched_pairs["image_path"] = image_embeddings_df.iloc[image_ind]["image_path"].values
    matched_pairs["music_path"] = music_embeddings_df.iloc[music_ind]["music_path"].values
    matched_pairs["music_embedding"] = music_embeddings_df.iloc[music_ind]["embeddings"].values
    matched_pairs["image_embedding"] = image_embeddings_df.iloc[image_ind]["embeddings"].values
    matched_pairs["score"] = 1 - cost_matrix[image_ind, music_ind]

    return matched_pairs
