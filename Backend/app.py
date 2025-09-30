from flask import Flask, request, jsonify, send_from_directory
import os

from utils.config import AppConfig
from utils.logger import get_logger
from utils.file_utils import compute_file_hash, detect_file_type, allowed_file, load_bytes
from utils.parsers import parse_file_to_records
from utils.cleaning import clean_records
from utils.chunking import chunk_records
from utils.embeddings import EmbeddingService
from utils.vectorstore import VectorStore
from utils.retrieval import HybridRetriever
from utils.llm import LLMService
from utils.tabular import TabularQueryEngine

try:
    from flask_cors import CORS
except Exception:
    CORS = None

app = Flask(__name__)
config = AppConfig()
logger = get_logger(__name__, config)

if CORS:
    CORS(app)

os.makedirs(config.upload_dir, exist_ok=True)
os.makedirs(config.vector_dir, exist_ok=True)

embedding_service = EmbeddingService(config, logger)
vector_store = VectorStore(config, logger, embedding_service)
retriever = HybridRetriever(config, logger, vector_store, embedding_service)
llm_service = LLMService(config, logger)
tabular_engine = TabularQueryEngine(logger)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


# Serve frontend files without modifying them
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Frontend')


@app.route('/')
def serve_home():
    return send_from_directory(_frontend_dir, 'home.html')


@app.route('/upload.html')
def serve_upload():
    return send_from_directory(_frontend_dir, 'upload.html')


@app.route('/report.html')
def serve_report():
    return send_from_directory(_frontend_dir, 'report.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({"status": "error", "error": "No file provided"}), 400

        filename = file.filename
        if not allowed_file(filename):
            return jsonify({"status": "error", "error": "Unsupported file type"}), 400

        save_path = os.path.join(config.upload_dir, filename)
        file.save(save_path)

        file_bytes = load_bytes(save_path)
        file_hash = compute_file_hash(file_bytes)
        file_type = detect_file_type(filename, file_bytes)

        records = parse_file_to_records(filename, file_bytes, file_type, logger)
        cleaned_records = clean_records(records, logger)
        chunks = chunk_records(cleaned_records, file_type, logger)

        vector_store.index_chunks(chunks, file_hash=file_hash, filename=filename, file_type=file_type)
        # invalidate BM25 cache after mutation
        retriever.invalidate_bm25()

        return jsonify({
            "status": "success",
            "message": "File processed and indexed",
            "filename": filename,
            "file_hash": file_hash,
            "file_type": file_type,
            "num_chunks": len(chunks)
        }), 200
    except Exception as exc:
        logger.exception("/upload failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        payload = request.get_json(force=True, silent=False)
        q = payload.get('query')
        top_k = int(payload.get('top_k', 5))
        mode = payload.get('mode', 'auto')
        use_bm25 = bool(payload.get('use_bm25', True))
        use_multiquery = bool(payload.get('use_multiquery', True))
        use_rerank = bool(payload.get('use_rerank', True))

        if not q:
            return jsonify({"error": "Missing 'query'"}), 400

        if mode in ('auto', 'tabular') and tabular_engine.looks_tabular_query(q):
            tabular_result = tabular_engine.execute(q, vector_store.get_tabular_frames())
            if tabular_result is not None:
                return jsonify({"answer": tabular_result, "mode": "tabular"}), 200

        restrict_hash = payload.get('file_hash')
        restrict_name = payload.get('filename')
        retrieved = retriever.retrieve(
            q, top_k=top_k, use_bm25=use_bm25, use_multiquery=use_multiquery, use_rerank=use_rerank,
            restrict_filename=restrict_name, restrict_file_hash=restrict_hash
        )
        context = retriever.format_context(retrieved)
        answer = llm_service.answer(q, context)

        return jsonify({"answer": answer, "chunks": retrieved, "mode": "text"}), 200
    except Exception as exc:
        logger.exception("/query failed")
        return jsonify({"error": str(exc)}), 500


# Compatibility endpoint for frontend's /ask contract
@app.route('/ask', methods=['POST'])
def ask():
    try:
        payload = request.get_json(force=True, silent=False)
        question = payload.get('question')
        restrict_hash = payload.get('file_hash')
        # accept either 'filename' (preferred) or legacy 'dataset'
        restrict_name = payload.get('filename') or payload.get('dataset')
        if not question:
            return jsonify({"type": "text", "answer": "Missing question"}), 400

        if tabular_engine.looks_tabular_query(question):
            tab_res = tabular_engine.execute(question, vector_store.get_tabular_frames())
            if isinstance(tab_res, list):
                if not tab_res:
                    return jsonify({"type": "text", "answer": "No data available"}), 200
                columns = list(tab_res[0].keys())
                rows = [[row.get(c) for c in columns] for row in tab_res]
                return jsonify({"type": "table", "answer": "Here are the top rows.", "table": {"columns": columns, "rows": rows}}), 200
            if isinstance(tab_res, (int, float)):
                return jsonify({"type": "text", "answer": str(tab_res)}), 200

        results = retriever.retrieve(
            question, top_k=5, use_bm25=True, use_multiquery=True, use_rerank=True,
            restrict_filename=restrict_name, restrict_file_hash=restrict_hash
        )
        context = retriever.format_context(results)
        answer = llm_service.answer(question, context)
        return jsonify({"type": "text", "answer": answer}), 200
    except Exception as exc:
        logger.exception("/ask failed")
        return jsonify({"type": "text", "answer": f"Error: {str(exc)}"}), 500


if __name__ == '__main__':
    app.run(host=config.host, port=config.port, debug=config.debug)

# Management APIs
@app.route('/files', methods=['GET'])
def list_files():
    return jsonify({"files": vector_store.list_files()}), 200

@app.route('/files', methods=['DELETE'])
def delete_file():
    try:
        payload = request.get_json(force=True, silent=False)
        removed = vector_store.remove_file(file_hash=payload.get('file_hash'), filename=payload.get('filename'))
        retriever.invalidate_bm25()
        return jsonify({"removed_chunks": removed}), 200
    except Exception as exc:
        logger.exception("/files DELETE failed")
        return jsonify({"error": str(exc)}), 500
