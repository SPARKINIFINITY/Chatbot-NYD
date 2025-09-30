import os
import json
from typing import List, Dict, Any

import numpy as np
import faiss


class VectorStore:
    def __init__(self, config, logger, embedding_service) -> None:
        self.config = config
        self.logger = logger
        self.embedding = embedding_service
        self.index_path = os.path.join(config.vector_dir, 'faiss.index')
        self.meta_path = os.path.join(config.vector_dir, 'metadata.jsonl')
        self.tabular_frames: List[Any] = []
        os.makedirs(config.vector_dir, exist_ok=True)
        self._index = faiss.IndexFlatIP(config.faiss_dim)
        self._metas: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self._index = faiss.read_index(self.index_path)
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    self._metas = [json.loads(line) for line in f]
            except Exception:
                self.logger.warning('Failed to load existing index, starting fresh')

    def _persist(self) -> None:
        try:
            faiss.write_index(self._index, self.index_path)
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                for m in self._metas:
                    f.write(json.dumps(m, ensure_ascii=False) + '\n')
        except Exception:
            self.logger.warning('Failed to persist index')

    def index_chunks(self, chunks: List[Dict[str, Any]], file_hash: str, filename: str, file_type: str) -> None:
        texts = []
        metas = []
        for ch in chunks:
            if ch['type'] == 'tabular':
                self.tabular_frames.append(ch['dataframe'])
                text_repr = ch['dataframe'].to_csv(index=False)
                texts.append(text_repr)
            else:
                texts.append(ch.get('text', ''))
            meta = {
                'file_hash': file_hash,
                'filename': filename,
                'file_type': file_type,
                'chunk_type': ch['type'],
                'metadata': ch.get('metadata', {})
            }
            metas.append(meta)
        if not texts:
            return
        vectors = self.embedding.embed(texts, cache_key=file_hash)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self._index.d:
            # rebuild index with correct dim
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._metas.extend(metas)
        self._persist()

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        qv = self.embedding.embed([query])[0].reshape(1, -1)
        scores, ids = self._index.search(qv, top_k)
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ids[0]):
            if idx == -1 or idx >= len(self._metas):
                continue
            meta = self._metas[idx]
            results.append({'score': float(scores[0][rank]), **meta})
        return results

    def get_tabular_frames(self):
        return self.tabular_frames
