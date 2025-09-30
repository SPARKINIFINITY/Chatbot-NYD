from typing import List, Dict, Any
import math
import pandas as pd


def _split_text(text: str, min_tokens: int, max_tokens: int) -> List[str]:
    # Approximate tokens by whitespace; for simplicity in absence of tokenizer
    words = text.split()
    chunk_size = max(min_tokens, min(max_tokens, 400))
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
    return chunks


def _split_code(text: str, min_lines: int, max_lines: int) -> List[str]:
    lines = text.splitlines()
    size = max(min_lines, min(max_lines, 80))
    chunks = []
    for i in range(0, len(lines), size):
        chunk_lines = lines[i:i + size]
        if chunk_lines:
            chunks.append('\n'.join(chunk_lines))
    return chunks


def _split_tabular(df: pd.DataFrame, min_rows: int, max_rows: int) -> List[pd.DataFrame]:
    n = len(df)
    size = max(min_rows, min(max_rows, 25))
    return [df.iloc[i:i + size] for i in range(0, n, size)]


def chunk_records(records: List[Dict[str, Any]], file_type: str, logger) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get('type') == 'text':
            text_chunks = _split_text(rec.get('text', ''), 300, 500)
            for idx, ch in enumerate(text_chunks):
                chunks.append({
                    'type': 'text',
                    'text': ch,
                    'metadata': {**rec.get('metadata', {}), 'chunk_id': idx}
                })
        elif rec.get('type') == 'code':
            code_chunks = _split_code(rec.get('text', ''), 50, 100)
            for idx, ch in enumerate(code_chunks):
                chunks.append({
                    'type': 'code',
                    'text': ch,
                    'metadata': {**rec.get('metadata', {}), 'chunk_id': idx}
                })
        elif rec.get('type') == 'tabular' and 'dataframe' in rec:
            frames = _split_tabular(rec['dataframe'], 10, 50)
            for idx, frame in enumerate(frames):
                chunks.append({
                    'type': 'tabular',
                    'dataframe': frame,
                    'metadata': {'chunk_id': idx}
                })
    return chunks
