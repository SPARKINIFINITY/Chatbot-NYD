import os
import json
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger
        self.model = SentenceTransformer(config.embedding_model_name)
        self.cache_dir = os.path.join(config.vector_dir, 'emb_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f'{key}.npy')

    def embed(self, texts: List[str], cache_key: str = None) -> np.ndarray:
        if cache_key:
            path = self._cache_path(cache_key)
            if os.path.exists(path):
                try:
                    return np.load(path)
                except Exception:
                    pass
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if cache_key:
            try:
                np.save(self._cache_path(cache_key), vectors)
            except Exception:
                self.logger.warning('Failed to save embedding cache for %s', cache_key)
        return vectors
