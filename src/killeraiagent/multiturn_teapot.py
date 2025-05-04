"""
multiturn_teapot.py

Transforms a Teapot-based model (Flan-T5 variant) into a multi-turn chat agent with retrieval augmentation (RAG).
It loads a local FAQ file (.jsonl), builds embeddings using a feature-extraction pipeline, and retrieves the top-k FAQ snippets
to augment the conversation prompt.

The embedding model is now set to "teapotai/teapotembedding" with truncation enabled.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from killeraiagent.models.base import ModelInfo
from killeraiagent.models.transformers.huggingface_llm import TransformersLLM
from killeraiagent.paths import get_data_paths
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Teapot, an AI assistant specialized in KAIA setup and configuration. "
    "Answer questions succinctly and provide practical guidance. If you find relevant FAQ snippets, "
    "incorporate them into your response."
)

# Documentation lookup dictionary.
# Adjust the paths as needed to point to the actual documentation man pages.
BASE_DOCUMENTATION_PATH = get_data_paths().base
DOCUMENTATION_PATHS = {
    "llama": BASE_DOCUMENTATION_PATH / "docs" / "llamacpp_usage.md",
    "setup": BASE_DOCUMENTATION_PATH / "docs" / "kaia_setup.md",
    "teapot": BASE_DOCUMENTATION_PATH / "docs" / "teapot_usage.md"
}

class FAQEntry:
    """Structure for a single FAQ record."""
    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answer = answer
        self.embedding: Optional[np.ndarray] = None

class MultiturnTeapot:
    """
    Multi-turn chat agent for KAIA that uses a Transformers-based Teapot model and RAG over a local FAQ file.
    """
    def __init__(
        self,
        faq_path: Optional[Path] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_tokens: int = 1024,
        rag_top_k: int = 2,
        model_info: Optional[ModelInfo] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.rag_top_k = rag_top_k
        self.conversation_history: List[Dict[str, str]] = []
        self.rag_context: str = ""

        if model_info is None:
            from killeraiagent.models.base import ModelInfo
            model_info = ModelInfo(
                model_id="teapotllm",
                model_engine="transformers",
                description="TeapotLLM with RAG for KAIA",
                context_length=1024,
                requires_gpu=False,
                model_size_gb=1.0
            )
        self.llm = TransformersLLM(
            model_info=model_info,
            pipeline_type="text2text-generation",
            model_name_or_path="teapotai/teapotllm",
            chat_format="flan"
        )
        self.llm.load()

        if faq_path is None:
            from killeraiagent.setup.setup_core import get_data_paths
            paths = get_data_paths()
            faq_path = paths.base / "faq" / "kaia_faq.jsonl"
        self.faq_path = faq_path
        self.faq_entries: List[FAQEntry] = []
        self._embedding_pipeline = None

        self._load_faq()
        self._build_embeddings()

    def _load_faq(self) -> None:
        """Loads FAQ entries from a JSON Lines file."""
        if not self.faq_path.exists():
            logger.warning(f"FAQ file not found at {self.faq_path}")
            return
        with open(self.faq_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question = data.get("question", "")
                answer = data.get("answer", "")
                if question and answer:
                    self.faq_entries.append(FAQEntry(question, answer))
        logger.info(f"Loaded {len(self.faq_entries)} FAQ entries from {self.faq_path}")

    def _build_embeddings(self) -> None:
        """Build embeddings for each FAQ entry using a feature-extraction pipeline."""
        if not self.faq_entries:
            return
        logger.info("Building embeddings for FAQ entries using teapotai/teapotembedding.")
        try:
            self._embedding_pipeline = pipeline(
                "feature-extraction",
                model="teapotai/teapotembedding",
                truncation=True,
                device=-1  # CPU mode; adjust if necessary.
            )
            for entry in self.faq_entries:
                text_to_embed = entry.question + " " + entry.answer
                result = self._embedding_pipeline(text_to_embed)
                if result is not None:
                    emb_output = result if isinstance(result, list) else [result]
                    emb_array = np.array(emb_output[0])
                    vec = np.mean(emb_array, axis=0)
                    entry.embedding = vec
        except Exception as e:
            logger.error(f"Error building FAQ embeddings: {e}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _handle_doc_request(self, query: str) -> str:
        """
        Handle documentation requests if the query starts with "doc" or "man".
        Looks up the topic and returns the corresponding documentation file path.
        """
        parts = query.strip().split()
        if len(parts) >= 2:
            topic = parts[1].lower()
            doc_path = DOCUMENTATION_PATHS.get(topic)
            if doc_path:
                return f"Documentation for '{topic}' is available at: {doc_path}"
            else:
                return f"Sorry, no documentation found for topic '{topic}'. Available topics are: {list(DOCUMENTATION_PATHS.keys())}"
        else:
            return "Usage: 'doc <topic>' (e.g. 'doc llama')."

    def chat(self, user_input: str) -> str:
        """Process a user query, retrieve relevant FAQ snippets, and generate a response.
           Also handles documentation lookup if requested."""
        # Check if user asked for documentation help:
        if user_input.strip().lower().startswith(("doc ", "man ")):
            return self._handle_doc_request(user_input)

        self.add_message("user", user_input)
        relevant_faqs = self._retrieve_faq(user_input, self.rag_top_k)
        prompt = self._format_prompt(user_input, relevant_faqs)
        text, _ = self.llm.generate(
            prompt,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.0,
            do_sample=True
        )
        text = text.strip()
        self.add_message("assistant", text)
        return text

    def _retrieve_faq(self, query: str, k: int) -> List[str]:
        """Retrieve top-k FAQ snippets based on cosine similarity with the query."""
        if not self.faq_entries or not self._embedding_pipeline:
            return []
        result = self._embedding_pipeline(query)
        if isinstance(result, list):
            query_emb = result[0]
        else:
            query_emb = np.array(result)
        query_vec = np.mean(query_emb, axis=0)
        sims = []
        for entry in self.faq_entries:
            if entry.embedding is None:
                continue
            sim = cosine_similarity([query_vec], [entry.embedding])[0][0]
            sims.append((sim, entry))
        sims.sort(key=lambda x: x[0], reverse=True)
        top_entries = [entry for sim, entry in sims if sim >= 0.2][:k]
        snippets = [f"(Q) {entry.question}\n(A) {entry.answer}" for entry in top_entries]
        return snippets

    def _format_prompt(self, user_input: str, faq_snippets: List[str]) -> str:
        """Construct the prompt using system prompt, additional RAG context, FAQ snippets, and conversation history."""
        history_lines = []
        for msg in self.conversation_history[:-1]:
            history_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
        history_lines.append(f"User: {user_input}\nAssistant:")
        snippet_block = ""
        if faq_snippets:
            snippet_block = "[Relevant FAQ Snippets]\n" + "\n\n".join(faq_snippets) + "\n\n"
        prompt = f"{self.system_prompt}\n\n{self.rag_context}\n\n{snippet_block}" + "\n".join(history_lines)
        return prompt

    def close(self) -> None:
        """Close the underlying LLM."""
        self.llm.close()
