from typing import List, Dict, Any
import pandas as pd
import numpy as np


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize headers
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    # Type inference
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            if not np.issubdtype(df[col].dtype, np.datetime64):
                df[col] = pd.to_numeric(df[col], errors='ignore')
    # Missing values
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())
        elif np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        else:
            df[col] = df[col].fillna('')
    # Duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # Outliers (clip 1st-99th percentile)
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            low, high = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def _clean_text(text: str) -> str:
    # Normalize whitespace and multi-line artifacts
    if not isinstance(text, str):
        text = str(text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = '\n'.join([ln for ln in lines if ln])
    return cleaned


def clean_records(records: List[Dict[str, Any]], logger) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for rec in records:
        rtype = rec.get('type')
        if rtype == 'tabular' and 'dataframe' in rec:
            df = _clean_dataframe(rec['dataframe'])
            cleaned.append({**rec, 'dataframe': df})
        elif rtype in {'text', 'code'} and 'text' in rec:
            cleaned.append({**rec, 'text': _clean_text(rec['text'])})
        else:
            cleaned.append(rec)
    return cleaned
