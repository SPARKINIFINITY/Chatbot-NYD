import os


class AppConfig:
    def __init__(self) -> None:
        root = os.path.dirname(os.path.dirname(__file__))
        self.upload_dir = os.path.join(root, 'uploads')
        self.vector_dir = os.path.join(root, 'vectorstore')

        self.host = os.environ.get('HOST', '0.0.0.0')
        self.port = int(os.environ.get('PORT', 5000))
        self.debug = os.environ.get('DEBUG', 'false').lower() == 'true'

        self.embedding_model_name = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.cross_encoder_name = os.environ.get('CROSS_ENCODER', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.generation_model_name = os.environ.get('GEN_MODEL', 'google/flan-t5-base')

        self.faiss_dim = int(os.environ.get('FAISS_DIM', 384))
        self.max_context_chars = int(os.environ.get('MAX_CONTEXT_CHARS', 8000))

        self.allowed_extensions = set(
            (os.environ.get('ALLOWED_EXT', 'csv,xlsx,json,txt,pdf,docx,py,js,ts,java,cpp,md')).split(',')
        )
