# src/llm/llm_interface.py

import os
import logging
import requests
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    """
    Interface for interacting with Large Language Models (LLMs).
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the LLM interface.

        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. API calls will fail.")

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM with retrieved context.
        """
        formatted_context = self._format_context(context_chunks)
        prompt = self._create_prompt(query, formatted_context)
        response = self._call_llm_api(prompt)
        return response

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks into a string for the prompt.
        """
        if not context_chunks:
            return "No relevant context found."

        formatted_chunks = []
        for i, chunk in enumerate(context_chunks):
            title = chunk["metadata"].get("title", "Untitled")
            source = chunk["metadata"].get("source", "Unknown source")
            text = chunk["text"]
            formatted_chunk = f"[Document {i+1}] {title}\nSource: {source}\n\n{text}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n".join(formatted_chunks)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM.
        """
        return f"""You are a helpful AI assistant that answers questions based on the provided context.

CONTEXT:
{context}

USER QUERY:
{query}

Please answer the query based only on the provided context. If the context doesn't contain relevant information, state that you don't have enough information. Include citations to specific documents when possible.

ANSWER:
"""

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API with the prompt.
        """
        if not self.api_key:
            raise RuntimeError("API key not configured. Please set the OPENAI_API_KEY environment variable.")

        url = f"{self.api_base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"API call failed with status code {response.status_code}: {response.text}")
                raise RuntimeError(f"LLM API call failed: {response.text}")

        except requests.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            raise RuntimeError(f"Error calling LLM API: {e}")
