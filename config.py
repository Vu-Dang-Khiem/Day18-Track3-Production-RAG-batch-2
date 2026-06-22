"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- LLM Provider ---
# Supported: "openai", "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def get_llm_client():
    """Get appropriate LLM client based on config."""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY), GROQ_MODEL, "groq"
    elif OPENAI_API_KEY:
        from openai import OpenAI
        return OpenAI(), LLM_MODEL, "openai"
    return None, "", ""

class GroqLLMWrapper:
    """Wraps LangchainLLMWrapper to fix Groq API limitations (n=1 only)."""
    def __init__(self, llm):
        self._llm = llm

    def set_run_config(self, run_config):
        self._llm.set_run_config(run_config)

    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
        return self._llm.generate_text(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks)

    async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
        return await self._llm.agenerate_text(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks)

    async def generate(self, prompt, n=1, temperature=None, stop=None, callbacks=None, is_async=True):
        return await self._llm.generate(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks, is_async=is_async)

def get_ragas_llm():
    """Get RAGAS LLM wrapper."""
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=GROQ_MODEL,
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0,
        )
        return GroqLLMWrapper(LangchainLLMWrapper(llm, run_config=RunConfig()))
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL)
        return LangchainLLMWrapper(llm, run_config=RunConfig())
    return None

def get_ragas_embeddings():
    """Get embeddings for RAGAS (sentence-transformers fallback)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf, run_config=RunConfig())

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
