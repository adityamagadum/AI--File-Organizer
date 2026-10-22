# nexa/gguf.py
"""
Compatibility placeholder for NexaVLMInference and NexaTextInference.

Provides common method names used across the project:
  - _chat (may return iterator)
  - create_completion (returns dict with choices -> [{'text': ...}])
  - create, create_chat_completion, completion (aliases)
This keeps your pipeline running for testing until you plug in real models.
"""

import time
import os
from typing import Optional, Any, Iterator


def _to_single_iterator(s: str) -> Iterator[str]:
    """Return a simple single-item iterator for compatibility with next(generator)."""
    yield s


class NexaVLMInference:
    def __init__(self, model_path: Optional[str] = None, local_path: Optional[str] = None,
                 stop_words: Optional[list] = None, temperature: float = 0.0,
                 max_new_tokens: int = 512, top_k: int = 1, top_p: float = 1.0,
                 profiling: bool = False, **kwargs):
        self.model_path = model_path
        self.local_path = local_path
        self.stop_words = stop_words or []
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.profiling = profiling
        print("⚙️ NexaVLMInference initialized (placeholder).")

    def _chat(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> Iterator[str]:
        """
        Simulated image+text chat/description method.
        Returns a generator/iterator (single-item) so callers using next() work.
        """
        time.sleep(0.03)
        fname = image_path or "unknown_image"
        try:
            fname = os.path.basename(fname)
        except Exception:
            pass

        simulated_description = (
            f"Simulated VLM description for '{fname}'. Prompt: {prompt[:200]}..."
        )
        return _to_single_iterator(simulated_description)

    # convenience alias in case other code calls infer()
    def infer(self, image_path: str, prompt: str = "") -> str:
        it = self._chat(prompt, image_path)
        # consume iterator and return string
        try:
            return next(it)
        except StopIteration:
            return ""

    # Provide alias names some code may call
    def create_completion(self, prompt: str, **kwargs) -> dict:
        text = self.infer(prompt if not kwargs.get('image_path') else kwargs['image_path'], prompt)
        return {"choices": [{"text": text}]}

    create = create_completion
    create_chat_completion = create_completion
    completion = create_completion


class NexaTextInference:
    def __init__(self, model_path: Optional[str] = None, local_path: Optional[str] = None,
                 stop_words: Optional[list] = None, temperature: float = 0.0,
                 max_new_tokens: int = 512, top_k: int = 1, top_p: float = 1.0,
                 profiling: bool = False, **kwargs):
        self.model_path = model_path
        self.local_path = local_path
        self.stop_words = stop_words or []
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.profiling = profiling
        print("⚙️ NexaTextInference initialized (placeholder).")

    # core text-generation: return a dict shaped like real completions
    def create_completion(self, prompt: str, **kwargs) -> dict:
        time.sleep(0.02)
        # produce a short, safe filename-like response for filename prompts,
        # otherwise echo a short summary for generic prompts.
        low = prompt.lower()
        if "filename" in low:
            text = "sample_filename"
        elif "category" in low or "category:" in low:
            text = "misc"
        else:
            # produce a brief "summary"
            summary = (prompt.strip().splitlines()[0]) if prompt else "simulated response"
            text = f"Simulated text response: {summary[:120]}"

        return {"choices": [{"text": text}]}

    # alias names used elsewhere
    def create(self, prompt: str, **kwargs) -> dict:
        return self.create_completion(prompt, **kwargs)

    def create_chat_completion(self, prompt: str, **kwargs) -> dict:
        return self.create_completion(prompt, **kwargs)

    def _chat(self, prompt: str, **kwargs) -> Iterator[str]:
        # Return iterator for compatibility if someone uses next(_chat(...))
        result = self.create_completion(prompt)
        text = ""
        try:
            text = result['choices'][0].get('text', '')
        except Exception:
            text = str(result)
        return _to_single_iterator(text)

    def completion(self, prompt: str, **kwargs) -> dict:
        return self.create_completion(prompt, **kwargs)

    # convenience wrapper
    def infer_text(self, text: str, **kwargs) -> str:
        resp = self.create_completion(text)
        return resp['choices'][0]['text'] if resp and 'choices' in resp else str(resp)
