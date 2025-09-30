import io
import json
from typing import List, Dict, Any

import pandas as pd
from docx import Document
from PyPDF2 import PdfReader


def parse_file_to_records(filename: str, data: bytes, file_type: str, logger) -> List[Dict[str, Any]]:
    if file_type == 'tabular':
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(data))
        else:
            df = pd.read_excel(io.BytesIO(data))
        return [{"type": "tabular", "dataframe": df}]

    if file_type == 'json':
        try:
            obj = json.loads(data.decode('utf-8', errors='ignore'))
        except Exception:
            obj = json.loads(data.decode(errors='ignore'))
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        return [{"type": "text", "text": text, "metadata": {"format": "json"}}]

    if file_type == 'document':
        if filename.lower().endswith('.pdf'):
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n".join(pages)
        else:
            doc = Document(io.BytesIO(data))
            text = "\n".join([p.text for p in doc.paragraphs])
        return [{"type": "text", "text": text, "metadata": {"format": "document"}}]

    if file_type == 'code':
        try:
            text = data.decode('utf-8', errors='ignore')
        except Exception:
            text = data.decode(errors='ignore')
        return [{"type": "code", "text": text, "metadata": {"language": filename.rsplit('.', 1)[-1].lower()}}]

    # default to text
    try:
        text = data.decode('utf-8', errors='ignore')
    except Exception:
        text = data.decode(errors='ignore')
    return [{"type": "text", "text": text, "metadata": {"format": "text"}}]
