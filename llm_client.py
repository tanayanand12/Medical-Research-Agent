"""
llm_client.py — Phase 1: LiteLLM abstraction for all LLM calls.

Singleton wrapper around LiteLLM that provides a provider-agnostic
interface for chat completions and embeddings.  All application code
must call LLMClient instead of importing openai/anthropic directly.

Configuration is loaded from models.yaml at startup.
"""

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm # type: ignore
from litellm.exceptions import ( # type: ignore
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
)
import yaml # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER_CONFIG = {
    "ollama/": {
        "api_base": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    },
    "openai/": {},
    "text-embedding": {},        # OpenAI native prefix
    "gemini/": {
        "api_key": os.environ.get("GEMINI_API_KEY")
    },
    "vertex_ai/": {
        "vertex_project": os.environ.get("VERTEX_PROJECT"),
        "vertex_location": os.environ.get("VERTEX_LOCATION", "us-central1")
    },
    "azure/": {
        "api_base": os.environ.get("AZURE_API_BASE"),
        "api_key": os.environ.get("AZURE_API_KEY"),
        "api_version": os.environ.get("AZURE_API_VERSION", "2024-02-01")
    },
    "deepseek/": {
        "api_base": "https://api.deepseek.com",
        "api_key": os.environ.get("DEEPSEEK_API_KEY")
    },
    "anthropic/": {},            # Claude does not support embeddings — raise immediately
    "huggingface/": {
        "api_key": os.environ.get("HUGGINGFACE_API_KEY")
    },
    "cohere/": {
        "api_key": os.environ.get("COHERE_API_KEY")
    },
}

# Suppress noisy litellm debug logs
litellm.suppress_debug_info = True

_MODELS_YAML = Path(__file__).parent / "models.yaml"


def _key_name_for_model(model: str) -> str:
    """Return the .env variable name that holds the API key for the given model."""
    _prefixes = {
        "openai/": "OPENAI_API_KEY",
        "text-embedding": "OPENAI_API_KEY",
        "anthropic/": "ANTHROPIC_API_KEY",
        "gemini/": "GEMINI_API_KEY",
        "deepseek/": "DEEPSEEK_API_KEY",
        "cohere/": "COHERE_API_KEY",
        "huggingface/": "HUGGINGFACE_API_KEY",
        "azure/": "AZURE_API_KEY",
        "ollama/": "no key needed, check ollama serve is running",
    }
    for prefix, key_name in _prefixes.items():
        if model.startswith(prefix):
            return key_name
    return "OPENAI_API_KEY"


def _call_with_retry(
    call: Any,
    operation: str,
    max_attempts: Optional[int] = None,
) -> Any:
    """Retry transient provider failures while preserving hard failures."""
    attempts = max_attempts or int(os.getenv("LLM_MAX_RETRIES", "5"))
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            message = str(exc).lower()
            transient = any(
                marker in message
                for marker in (
                    "429",
                    "rate limit",
                    "resource_exhausted",
                    "503",
                    "service unavailable",
                    "temporarily unavailable",
                    "timeout",
                    "timed out",
                )
            )
            if not transient or attempt == attempts:
                raise

            retry_match = re.search(r"retry in ([\d.]+)s", message)
            delay = (
                float(retry_match.group(1)) + 1.0
                if retry_match
                else min(60.0, float(2 ** (attempt - 1)))
            )
            logger.warning(
                "%s transient failure; retrying in %.1fs (%d/%d): %s",
                operation,
                delay,
                attempt,
                attempts,
                exc,
            )
            time.sleep(delay)

    raise RuntimeError(f"{operation} retry loop terminated unexpectedly")


class LLMClient:
    """Provider-agnostic LLM client backed by LiteLLM.

    Implements singleton pattern — every ``LLMClient()`` call returns the
    same instance.  The registry is loaded once from ``models.yaml``.
    """

    _instance: Optional["LLMClient"] = None
    _lock = threading.Lock()

    # ---- singleton --------------------------------------------------------

    def __new__(cls, config_path: Optional[str] = None) -> "LLMClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._initialised = False
                    cls._instance = obj
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        if self._initialised:
            return
        self._initialised = True

        self._config_path = Path(config_path) if config_path else _MODELS_YAML
        self._models: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, Dict[str, Any]] = {}
        self._default_model: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
        self._default_embedding: str = os.getenv(
            "DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large"
        )

        # Cumulative metrics (Phase 5 will export to Prometheus)
        self._total_calls = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_cost_usd = 0.0

        self._load_config()

    # ---- config -----------------------------------------------------------

    def _load_config(self) -> None:
        """Load models.yaml and build lookup dicts."""
        if not self._config_path.exists():
            logger.warning(
                "models.yaml not found at %s — LLMClient will rely on "
                "LiteLLM defaults and environment variables.",
                self._config_path,
            )
            return

        with open(self._config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        for entry in raw.get("models", []):
            name = entry["model_name"]
            self._models[name] = entry

        for entry in raw.get("embeddings", []):
            name = entry["model_name"]
            self._embeddings[name] = entry

        logger.info(
            "LLMClient initialised — %d chat models, %d embedding models. "
            "Default chat=%s, default embed=%s",
            len(self._models),
            len(self._embeddings),
            self._default_model,
            self._default_embedding,
        )

    # ---- public API -------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request via LiteLLM.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list (role + content).
        model : str, optional
            Model name as registered in models.yaml.  Falls back to
            ``DEFAULT_LLM_MODEL`` env var, then ``gpt-4o``.
        temperature : float
            Sampling temperature (default 0.7).
        max_tokens : int, optional
            Maximum tokens in the response.
        **kwargs
            Passed through to ``litellm.completion()``.

        Returns
        -------
        str
            The assistant's reply text.
        """
        model = model or self._default_model

        if model.startswith("ollama/"):
            import requests  # type: ignore

            model_name = model.replace("ollama/", "", 1)
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            options: Dict[str, Any] = {"temperature": temperature}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            payload: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": options,
            }
            if kwargs.get("format") is not None:
                payload["format"] = kwargs["format"]

            start = time.time()

            def request_chat() -> Any:
                response = requests.post(
                    f"{base}/api/chat",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                return response

            response = _call_with_retry(
                request_chat,
                operation=f"ollama_chat:{model_name}",
            )
            body = response.json()
            text = body.get("message", {}).get("content", "")
            tokens_in = int(body.get("prompt_eval_count", 0) or 0)
            tokens_out = int(body.get("eval_count", 0) or 0)
            self._total_calls += 1
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out
            logger.info(
                "LLMClient.chat model=%s tokens_in=%d tokens_out=%d "
                "cost=$0.000000 latency=%.0fms",
                model,
                tokens_in,
                tokens_out,
                (time.time() - start) * 1000,
            )
            return text

        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        call_kwargs.update(kwargs)

        start = time.time()
        try:
            response = _call_with_retry(
                lambda: litellm.completion(**call_kwargs),
                operation=f"chat:{model}",
            )
        except NotFoundError as e:
            raise RuntimeError(
                f"Model '{model}' not found. "
                f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                f"For Ollama: run 'ollama pull <model>'. "
                f"Original: {e}"
            ) from e
        except AuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed for model '{model}'. "
                f"Check API key in .env: {_key_name_for_model(model)}. "
                f"Original: {e}"
            ) from e
        except APIConnectionError as e:
            # Ollama surfaces "model not found" as APIConnectionError (HTTP 404 quirk)
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{model}' not found. "
                    f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                    f"For Ollama: run 'ollama pull <model>'. "
                    f"Original: {e}"
                ) from e
            logger.error("LLMClient.chat() failed for model=%s", model, exc_info=True)
            raise
        except Exception:
            logger.error("LLMClient.chat() failed for model=%s", model, exc_info=True)
            raise

        latency_ms = (time.time() - start) * 1000
        text = response.choices[0].message.content or ""

        # Track usage
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = self.get_cost(model, tokens_in, tokens_out)

        self._total_calls += 1
        self._total_tokens_in += tokens_in
        self._total_tokens_out += tokens_out
        self._total_cost_usd += cost

        logger.info(
            "LLMClient.chat model=%s tokens_in=%d tokens_out=%d "
            "cost=$%.6f latency=%.0fms",
            model,
            tokens_in,
            tokens_out,
            cost,
            latency_ms,
        )

        return text

    def _get_embedding_kwargs(self, model: str) -> dict:
        """Return provider-specific kwargs for litellm.embedding()."""
        for prefix, kwargs in EMBEDDING_PROVIDER_CONFIG.items():
            if model.startswith(prefix):
                if prefix == "anthropic/":
                    raise RuntimeError(
                        f"Model '{model}' is an Anthropic LLM — "
                        f"Anthropic does not provide an embeddings API. "
                        f"Use openai/text-embedding-3-large or "
                        f"ollama/nomic-embed-text instead."
                    )
                # Strip None values — litellm ignores missing keys
                return {k: v for k, v in kwargs.items() if v is not None}
        return {}  # unknown provider — let litellm attempt with defaults

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """Generate an embedding vector via LiteLLM.

        Parameters
        ----------
        text : str
            Input text to embed.
        model : str, optional
            Embedding model name.  Falls back to
            ``DEFAULT_EMBEDDING_MODEL`` env var.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        model = model or self._default_embedding

        try:
            if model.startswith("ollama/"):
                import requests # type: ignore
                model_name = model.replace("ollama/", "")
                base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                def request_embedding() -> Any:
                    response = requests.post(
                        f"{base}/api/embeddings",
                        json={"model": model_name, "prompt": text},
                        timeout=30,
                    )
                    response.raise_for_status()
                    return response

                resp = _call_with_retry(
                    request_embedding,
                    operation=f"ollama_embed:{model_name}",
                )
                return resp.json()["embedding"]

            kwargs = self._get_embedding_kwargs(model)
            response = _call_with_retry(
                lambda: litellm.embedding(model=model, input=[text], **kwargs),
                operation=f"embed:{model}",
            )
        except NotFoundError as e:
            raise RuntimeError(
                f"Model '{model}' not found. "
                f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                f"For Ollama: run 'ollama pull <model>'. "
                f"Original: {e}"
            ) from e
        except AuthenticationError as e:
            raise RuntimeError(
                f"Authentication failed for model '{model}'. "
                f"Check API key in .env: {_key_name_for_model(model)}. "
                f"Original: {e}"
            ) from e
        except APIConnectionError as e:
            # Ollama surfaces "model not found" as APIConnectionError (HTTP 404 quirk)
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{model}' not found. "
                    f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                    f"For Ollama: run 'ollama pull <model>'. "
                    f"Original: {e}"
                ) from e
            logger.error(
                "LLMClient.embed() failed for model=%s", model, exc_info=True
            )
            raise
        except Exception:
            logger.error(
                "LLMClient.embed() failed for model=%s", model, exc_info=True
            )
            raise

        vector = response.data[0]["embedding"]

        # Track cost
        usage = getattr(response, "usage", None)
        tokens_used = getattr(usage, "total_tokens", 0) if usage else 0
        embed_info = self._embeddings.get(model, {}).get("model_info", {})
        cost = tokens_used * embed_info.get("cost_per_1k_tokens", 0.0) / 1000
        self._total_cost_usd += cost

        logger.info(
            "LLMClient.embed model=%s dim=%d tokens=%d cost=$%.6f",
            model,
            len(vector),
            tokens_used,
            cost,
        )

        return vector

    def get_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate USD cost for a chat call.

        Looks up per-token rates in models.yaml.  Returns 0.0 for
        models not found in the registry (e.g. local Ollama).
        """
        info = self._models.get(model, {}).get("model_info", {})
        cost_in = info.get("cost_per_1k_input_tokens", 0.0)
        cost_out = info.get("cost_per_1k_output_tokens", 0.0)
        return (tokens_in * cost_in + tokens_out * cost_out) / 1000
    
    def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed a list of texts using a single batched API call per batch.

        Routes through the same provider kwargs as :meth:`embed`, so all
        provider configuration (API keys, base URLs) is inherited correctly.

        Gemini ``gemini-embedding-001`` accepts up to 100 inputs per request.
        Texts exceeding ``batch_size`` are split and concatenated automatically.

        Ollama does not support native batch embedding and falls back to
        sequential :meth:`embed` calls automatically.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        model : str, optional
            Embedding model name. Falls back to ``DEFAULT_EMBEDDING_MODEL``.
        batch_size : int
            Max texts per API call. Default 100 (Gemini limit).

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, in the same order.
        """
        if not texts:
            return []

        model = model or self._default_embedding

        # Modern Ollama exposes native batch embeddings at /api/embed.
        if model.startswith("ollama/"):
            import requests  # type: ignore

            model_name = model.replace("ollama/", "")
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            vectors: List[List[float]] = []
            logger.info(
                "LLMClient.embed_batch: using Ollama native batch endpoint "
                "(%d texts)",
                len(texts),
            )
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]

                def request_batch() -> Any:
                    response = requests.post(
                        f"{base}/api/embed",
                        json={"model": model_name, "input": batch},
                        timeout=120,
                    )
                    response.raise_for_status()
                    return response

                response = _call_with_retry(
                    request_batch,
                    operation=f"ollama_embed_batch:{model_name}",
                )
                vectors.extend(response.json()["embeddings"])
            return vectors

        kwargs = self._get_embedding_kwargs(model)
        all_embeddings: List[List[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            try:
                response = _call_with_retry(
                    lambda: litellm.embedding(model=model, input=batch, **kwargs),
                    operation=f"embed_batch:{model}:{batch_num}/{total_batches}",
                )
            except NotFoundError as e:
                raise RuntimeError(
                    f"Embedding model '{model}' not found. "
                    f"Check models.yaml or DEFAULT_EMBEDDING_MODEL. "
                    f"Original: {e}"
                ) from e
            except AuthenticationError as e:
                raise RuntimeError(
                    f"Authentication failed for embedding model '{model}'. "
                    f"Check {_key_name_for_model(model)} in .env. "
                    f"Original: {e}"
                ) from e
            except Exception:
                logger.error(
                    "LLMClient.embed_batch failed at batch %d/%d for model=%s",
                    batch_num, total_batches, model, exc_info=True,
                )
                raise

            batch_vecs = [item["embedding"] for item in response.data]
            all_embeddings.extend(batch_vecs)

            usage = getattr(response, "usage", None)
            tokens_used = getattr(usage, "total_tokens", 0) if usage else 0
            embed_info = self._embeddings.get(model, {}).get("model_info", {})
            cost = tokens_used * embed_info.get("cost_per_1k_tokens", 0.0) / 1000
            self._total_cost_usd += cost

            logger.info(
                "LLMClient.embed_batch model=%s batch=%d/%d texts=%d "
                "dim=%d tokens=%d cost=$%.6f",
                model, batch_num, total_batches, len(batch),
                len(batch_vecs[0]) if batch_vecs else 0,
                tokens_used, cost,
            )

        return all_embeddings

    # ---- introspection ----------------------------------------------------

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_embedding_model(self) -> str:
        return self._default_embedding

    @property
    def available_models(self) -> List[str]:
        return list(self._models.keys())

    @property
    def available_embeddings(self) -> List[str]:
        return list(self._embeddings.keys())

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return cumulative usage metrics (Phase 5 will push to Prometheus)."""
        return {
            "total_calls": self._total_calls,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_out": self._total_tokens_out,
            "total_cost_usd": self._total_cost_usd,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        with cls._lock:
            cls._instance = None

