from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class HybridRetriever:
    def __init__(self, config, logger, vector_store, embedding_service) -> None:
        self.config = config
        self.logger = logger
        self.store = vector_store
        self.embedding = embedding_service
        self.cross = CrossEncoder(config.cross_encoder_name)
        self._bm25_corpus = []
        self._bm25 = None

    def _ensure_bm25(self):
        if self._bm25 is None:
            corpus = []
            for m in getattr(self.store, '_metas', []):
                # crude representation: combine filename and types
                corpus.append(' '.join([str(m.get('filename', '')), str(m.get('file_type', '')), str(m.get('chunk_type', ''))]))
            self._bm25_corpus = [doc.split() for doc in corpus]
            if self._bm25_corpus:
                self._bm25 = BM25Okapi(self._bm25_corpus)

    def _expand_queries(self, q: str, enable: bool) -> List[str]:
        if not enable:
            return [q]
        # naive multi-query: synonyms via simple templates
        variants = [q]
        variants.append(f"In other words: {q}")
        variants.append(f"Detailed: {q}")
        return variants

    def retrieve(self, q: str, top_k: int, use_bm25: bool, use_multiquery: bool, use_rerank: bool) -> List[Dict[str, Any]]:
        sem_results = self.store.search(q, top_k * 3)

        bm25_results: List[Dict[str, Any]] = []
        if use_bm25:
            self._ensure_bm25()
            if self._bm25 is not None:
                scores = self._bm25.get_scores(q.split())
                pairs = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[: top_k * 2]
                for idx, sc in pairs:
                    if idx < len(getattr(self.store, '_metas', [])):
                        meta = self.store._metas[idx]
                        bm25_results.append({'score': float(sc), **meta})

        combined = sem_results + bm25_results
        # Deduplicate by identity
        seen = set()
        unique = []
        for r in combined:
            key = (r.get('file_hash'), r.get('metadata', {}).get('chunk_id'), r.get('chunk_type'))
            if key not in seen:
                seen.add(key)
                unique.append(r)

        if use_rerank and unique:
            # Build pseudo text for each meta for reranking context
            texts = [f"{u.get('filename')} {u.get('chunk_type')}" for u in unique]
            scores = self.cross.predict([[q, t] for t in texts])
            rescored = [{**u, 'rerank_score': float(s)} for u, s in zip(unique, scores)]
            rescored.sort(key=lambda x: x['rerank_score'], reverse=True)
            return rescored[:top_k]

        return unique[:top_k]

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        lines = []
        for r in results:
            src = f"[{r.get('filename')} | {r.get('chunk_type')}]"
            lines.append(src)
        return '\n'.join(lines)
