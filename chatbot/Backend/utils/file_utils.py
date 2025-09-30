import hashlib
import mimetypes
from typing import Optional

from utils.config import AppConfig


def allowed_file(filename: str) -> bool:
    config = AppConfig()
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in config.allowed_extensions


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_file_type(filename: str, data: Optional[bytes] = None) -> str:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in {'csv', 'xlsx'}:
        return 'tabular'
    if ext in {'json'}:
        return 'json'
    if ext in {'txt', 'md'}:
        return 'text'
    if ext in {'pdf', 'docx'}:
        return 'document'
    if ext in {'py', 'js', 'ts', 'java', 'cpp'}:
        return 'code'
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and 'text' in guessed:
        return 'text'
    return 'text'


def load_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()
