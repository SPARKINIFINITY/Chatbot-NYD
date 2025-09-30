import os
import json
import io
from typing import List, Dict, Any

import pandas as pd
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
                # Reconstruct tabular frames from persisted CSV text if available
                self.tabular_frames = []
                for m in self._metas:
                    if m.get('chunk_type') == 'tabular' and m.get('text'):
                        try:
                            df = pd.read_csv(io.StringIO(m['text']))
                            self.tabular_frames.append(df)
                        except Exception:
                            # Skip if cannot reconstruct
                            pass
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
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for ch in chunks:
            if ch['type'] == 'tabular':
                self.tabular_frames.append(ch['dataframe'])
                text_repr = ch['dataframe'].to_csv(index=False)
                texts.append(text_repr)
                content_text = text_repr
            else:
                content_text = ch.get('text', '')
                texts.append(content_text)
            meta = {
                'file_hash': file_hash,
                'filename': filename,
                'file_type': file_type,
                'chunk_type': ch['type'],
                'metadata': ch.get('metadata', {}),
                # Persist content text to enable BM25 and LLM context
                'text': content_text
            }
            metas.append(meta)
        if not texts:
            return
        # Use a cache key that depends on file hash and number of chunks to avoid shape mismatch
        cache_key = f"{file_hash}_{len(texts)}"
        vectors = self.embedding.embed(texts, cache_key=cache_key)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self._index.d:
            # rebuild index with correct dim
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._metas.extend(metas)
        self._persist()

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if getattr(self._index, 'ntotal', 0) == 0:
            return []
        qv = self.embedding.embed([query])[0].reshape(1, -1)
        scores, ids = self._index.search(qv, top_k)
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(ids[0]):
            if idx == -1 or idx >= len(self._metas):
                continue
            meta = self._metas[idx]
            results.append({'score': float(scores[0][rank]), **meta})
        return results

    def list_files(self) -> List[Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for m in self._metas:
            key = m.get('file_hash') or m.get('filename')
            if key not in summary:
                summary[key] = {
                    'file_hash': m.get('file_hash'),
                    'filename': m.get('filename'),
                    'file_type': m.get('file_type'),
                    'num_chunks': 0
                }
            summary[key]['num_chunks'] += 1
        return list(summary.values())

    def _rebuild_index_from_metas(self) -> None:
        texts: List[str] = []
        for m in self._metas:
            texts.append(m.get('text', ''))
        if not texts:
            # fresh empty index
            self._index = faiss.IndexFlatIP(self._index.d if hasattr(self._index, 'd') else self.config.faiss_dim)
            self.tabular_frames = []
            self._persist()
            return
        vectors = self.embedding.embed(texts)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        # rebuild tabular frames
        self.tabular_frames = []
        for m in self._metas:
            if m.get('chunk_type') == 'tabular' and m.get('text'):
                try:
                    df = pd.read_csv(io.StringIO(m['text']))
                    self.tabular_frames.append(df)
                except Exception:
                    pass
        self._persist()

    def remove_file(self, file_hash: str = None, filename: str = None) -> int:
        before = len(self._metas)
        if not file_hash and not filename:
            return 0
        new_metas: List[Dict[str, Any]] = []
        for m in self._metas:
            if file_hash and m.get('file_hash') == file_hash:
                continue
            if filename and m.get('filename') == filename:
                continue
            new_metas.append(m)
        removed = before - len(new_metas)
        if removed > 0:
            self._metas = new_metas
            self._rebuild_index_from_metas()
        return removed

    def get_tabular_frames(self):
        return self.tabular_frames
