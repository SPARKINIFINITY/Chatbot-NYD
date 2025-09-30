from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class LLMService:
    def __init__(self, config, logger) -> None:
        self.config = config
        self.logger = logger
        try:
            self.generator = pipeline('text2text-generation', model=config.generation_model_name)
        except Exception:
            # Fallback small model
            self.generator = pipeline('text2text-generation', model='google/flan-t5-small')

    def answer(self, question: str, context: str) -> str:
        prompt = f"You are a helpful dataset assistant. Use the provided context to answer.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        out = self.generator(prompt, max_new_tokens=256, do_sample=False)
        return out[0]['generated_text']
