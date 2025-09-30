from typing import List, Optional, Any
import re
import pandas as pd


class TabularQueryEngine:
    def __init__(self, logger) -> None:
        self.logger = logger

    def looks_tabular_query(self, q: str) -> bool:
        ql = q.lower()
        return any(k in ql for k in ['sum', 'average', 'avg', 'top', 'max', 'min']) and any(k in ql for k in ['column', 'col', 'by', 'rows'])

    def execute(self, q: str, frames: List[pd.DataFrame]) -> Optional[Any]:
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        ql = q.lower()

        # top rows
        m = re.search(r'top\s+(\d+)', ql)
        if m:
            n = int(m.group(1))
            return df.head(n).to_dict(orient='records')

        # sum/avg over a column
        col_match = re.search(r'column\s+([a-zA-Z0-9_\- ]+)', ql)
        if col_match:
            col = col_match.group(1).strip().replace(' ', '_')
            if col in df.columns:
                if 'sum' in ql:
                    return float(df[col].sum())
                if 'avg' in ql or 'average' in ql:
                    return float(df[col].mean())
                if 'max' in ql:
                    return float(df[col].max())
                if 'min' in ql:
                    return float(df[col].min())
        return None
