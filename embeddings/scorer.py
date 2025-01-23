from typing import Tuple

import numpy as np
from imagebind.models.imagebind_model import ModalityType
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingScorer:
    def __init__(self, query_type: str = ModalityType.AUDIO, target_type: str = ModalityType.VISION, metric='cosine'):
        self.query_type = query_type
        self.target_type = target_type
        self.metric = metric

    def score(self, query: np.ndarray, target: np.ndarray) -> float:
        if self.metric == 'cosine':
            return cosine_similarity(query, target)
        else:
            raise ValueError('Unknown metric: {}'.format(self.metric))

    def find_topk(self, query: np.ndarray, targets: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(query, targets)
        topk_indices = np.argsort(scores)[0][::-1][:top_k]
        return topk_indices, scores[0][topk_indices]
