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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import litellm # type: ignore
from litellm.exceptions import ( # type: ignore
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
)
import yaml # type: ignore
from dotenv import load_dotenv # type: ignore
from evaluation_core.deadline import RuntimeDeadlineExceeded, remaining_seconds

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallResult:
    """Backward-compatible structured metadata for one chat completion."""

    text: str
    model: str
    model_revision: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_sec: float
    finish_reason: str
    provider_metadata: Dict[str, Any]
    status: str = "success"
    error_type: str = ""


def _split_model_revision(model: str) -> tuple[str, str]:
    if "@" not in model:
        return model, ""
    name, revision = model.rsplit("@", 1)
    return name, revision

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
    deadline_at: Optional[float] = None,
    on_attempt_failure: Optional[
        Callable[[BaseException, int, float], None]
    ] = None,
) -> Any:
    """Retry transient provider failures while preserving hard failures."""
    attempts = max_attempts or int(os.getenv("LLM_MAX_RETRIES", "5"))
    for attempt in range(1, attempts + 1):
        if (
            deadline_at is not None
            and time.monotonic() >= float(deadline_at)
        ):
            raise RuntimeDeadlineExceeded(
                f"{operation} request deadline exhausted"
            )
        attempt_started_at = time.monotonic()
        try:
            return call()
        except Exception as exc:
            if on_attempt_failure is not None:
                try:
                    on_attempt_failure(
                        exc,
                        attempt,
                        max(
                            0.0,
                            time.monotonic() - attempt_started_at,
                        ),
                    )
                except Exception as telemetry_exc:
                    logger.warning(
                        "%s retry telemetry failed error_type=%s",
                        operation,
                        type(telemetry_exc).__name__,
                    )
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
            if deadline_at is not None:
                remaining = float(deadline_at) - time.monotonic()
                if remaining <= 0 or delay >= remaining:
                    raise RuntimeDeadlineExceeded(
                        f"{operation} request deadline exhausted before retry"
                    ) from exc
            logger.warning(
                "%s transient failure; retrying in %.1fs (%d/%d) error_type=%s",
                operation,
                delay,
                attempt,
                attempts,
                type(exc).__name__,
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
        self._call_local = threading.local()

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
        client_max_attempts = kwargs.pop("client_max_attempts", None)
        deadline_at = kwargs.pop("deadline_at", None)
        telemetry_context = {
            "telemetry_stage": str(
                kwargs.pop("_telemetry_stage", "llm_call")
            ),
            "telemetry_attempt_id": str(
                kwargs.pop("_telemetry_attempt_id", "")
            ),
            "telemetry_parent_attempt_id": str(
                kwargs.pop("_telemetry_parent_attempt_id", "")
            ),
            "telemetry_repair_status": str(
                kwargs.pop("_telemetry_repair_status", "initial")
            ),
        }
        provider_attempt = {"current": 1}
        callback_failures: set[int] = set()

        def record_failed_call(
            exc: BaseException,
            *,
            started_at: float,
            provider: str,
            attempt: Optional[int] = None,
            latency_sec: Optional[float] = None,
            from_retry_callback: bool = False,
        ) -> None:
            if (
                not from_retry_callback
                and id(exc) in callback_failures
            ):
                return
            if from_retry_callback:
                callback_failures.add(id(exc))
            attempt_number = int(
                attempt or provider_attempt["current"]
            )
            provider_attempt["current"] = max(
                provider_attempt["current"], attempt_number + 1
            )
            exact_model, revision = _split_model_revision(model)
            self._record_call_result(
                LLMCallResult(
                    text="",
                    model=exact_model,
                    model_revision=revision,
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_sec=max(
                        0.0,
                        (
                            float(latency_sec)
                            if latency_sec is not None
                            else time.time() - started_at
                        ),
                    ),
                    finish_reason="error",
                    provider_metadata={
                        "provider": provider,
                        "provider_attempt": attempt_number,
                        "error_type": type(exc).__name__,
                        **telemetry_context,
                    },
                    status="error",
                    error_type=type(exc).__name__,
                )
            )

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
            request_timeout = float(kwargs.get("timeout", 300))

            start = time.time()

            def request_chat() -> Any:
                response = requests.post(
                    f"{base}/api/chat",
                    json=payload,
                    timeout=min(
                        request_timeout,
                        remaining_seconds(
                            deadline_at, default=request_timeout
                        ),
                    ),
                )
                response.raise_for_status()
                return response

            try:
                response = _call_with_retry(
                    request_chat,
                    operation=f"ollama_chat:{model_name}",
                    max_attempts=client_max_attempts,
                    deadline_at=deadline_at,
                    on_attempt_failure=lambda exc, attempt, latency: (
                        record_failed_call(
                            exc,
                            started_at=start,
                            provider="ollama",
                            attempt=attempt,
                            latency_sec=latency,
                            from_retry_callback=True,
                        )
                    ),
                )
            except Exception as exc:
                record_failed_call(
                    exc,
                    started_at=start,
                    provider="ollama",
                )
                raise
            body = response.json()
            text = body.get("message", {}).get("content", "")
            tokens_in = int(body.get("prompt_eval_count", 0) or 0)
            tokens_out = int(body.get("eval_count", 0) or 0)
            self._total_calls += 1
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out
            exact_model, revision = _split_model_revision(model)
            self._record_call_result(LLMCallResult(
                text=text,
                model=exact_model,
                model_revision=revision,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=0.0,
                latency_sec=max(0.0, time.time() - start),
                finish_reason=str(body.get("done_reason") or ""),
                provider_metadata={
                    "provider": "ollama",
                    "provider_attempt": provider_attempt["current"],
                    **telemetry_context,
                },
            ))
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

        def request_completion() -> Any:
            request_kwargs = dict(call_kwargs)
            if deadline_at is not None:
                remaining = remaining_seconds(deadline_at)
                configured_timeout = request_kwargs.get("timeout")
                request_kwargs["timeout"] = (
                    min(float(configured_timeout), remaining)
                    if isinstance(configured_timeout, (int, float))
                    and not isinstance(configured_timeout, bool)
                    else remaining
                )
            return litellm.completion(**request_kwargs)

        try:
            response = _call_with_retry(
                request_completion,
                operation=f"chat:{model}",
                max_attempts=client_max_attempts,
                deadline_at=deadline_at,
                on_attempt_failure=lambda exc, attempt, latency: (
                    record_failed_call(
                        exc,
                        started_at=start,
                        provider="litellm",
                        attempt=attempt,
                        latency_sec=latency,
                        from_retry_callback=True,
                    )
                ),
            )
        except NotFoundError as e:
            record_failed_call(e, started_at=start, provider="litellm")
            raise RuntimeError(
                f"Model '{model}' not found. "
                f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                f"For Ollama: run 'ollama pull <model>'."
            ) from e
        except AuthenticationError as e:
            record_failed_call(e, started_at=start, provider="litellm")
            raise RuntimeError(
                f"Authentication failed for model '{model}'. "
                f"Check API key in .env: {_key_name_for_model(model)}."
            ) from e
        except APIConnectionError as e:
            record_failed_call(e, started_at=start, provider="litellm")
            # Ollama surfaces "model not found" as APIConnectionError (HTTP 404 quirk)
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{model}' not found. "
                    f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                    f"For Ollama: run 'ollama pull <model>'."
                ) from e
            logger.error(
                "LLMClient.chat() failed for model=%s error_type=%s",
                model,
                type(e).__name__,
            )
            raise
        except Exception as exc:
            record_failed_call(exc, started_at=start, provider="litellm")
            logger.error(
                "LLMClient.chat() failed for model=%s error_type=%s",
                model,
                type(exc).__name__,
            )
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
        response_model = str(getattr(response, "model", None) or model)
        exact_model, revision = _split_model_revision(response_model)
        if not revision:
            _, revision = _split_model_revision(model)
        finish_reason = str(
            getattr(response.choices[0], "finish_reason", None) or ""
        )
        provider = model.split("/", 1)[0] if "/" in model else "litellm"
        self._record_call_result(LLMCallResult(
            text=text,
            model=exact_model,
            model_revision=revision,
            tokens_in=int(tokens_in or 0),
            tokens_out=int(tokens_out or 0),
            cost_usd=float(cost or 0.0),
            latency_sec=max(0.0, latency_ms / 1000.0),
            finish_reason=finish_reason,
            provider_metadata={
                "provider": provider,
                "provider_attempt": provider_attempt["current"],
                "response_id": str(getattr(response, "id", "") or ""),
                **telemetry_context,
            },
        ))

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

    def chat_with_metadata(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCallResult:
        """Return one call's metadata while preserving ``chat() -> str``."""
        self._call_local.last_result = None
        requested_model = str(model or self._default_model)
        started_at = time.monotonic()
        text = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        result = getattr(self._call_local, "last_result", None)
        if isinstance(result, LLMCallResult):
            return result
        exact_model, revision = _split_model_revision(requested_model)
        return LLMCallResult(
            text=text,
            model=exact_model,
            model_revision=revision,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_sec=max(0.0, time.monotonic() - started_at),
            finish_reason="",
            provider_metadata={"provider": "unknown"},
        )

    def _record_call_result(self, result: LLMCallResult) -> None:
        local = self._call_local
        local.last_result = result
        history = list(getattr(local, "call_history", []))
        history.append(result)
        local.call_history = history
        local.calls = int(getattr(local, "calls", 0)) + 1
        local.tokens_in = int(getattr(local, "tokens_in", 0)) + result.tokens_in
        local.tokens_out = int(getattr(local, "tokens_out", 0)) + result.tokens_out
        local.cost_usd = float(getattr(local, "cost_usd", 0.0)) + result.cost_usd
        local.latency_sec = float(
            getattr(local, "latency_sec", 0.0)
        ) + result.latency_sec

    def thread_metrics(self) -> Dict[str, Any]:
        """Return cumulative metrics scoped to the current worker thread."""
        local = self._call_local
        last = getattr(local, "last_result", None)
        return {
            "calls": int(getattr(local, "calls", 0)),
            "tokens_in": int(getattr(local, "tokens_in", 0)),
            "tokens_out": int(getattr(local, "tokens_out", 0)),
            "cost_usd": float(getattr(local, "cost_usd", 0.0)),
            "latency_sec": float(getattr(local, "latency_sec", 0.0)),
            "last_model": last.model if isinstance(last, LLMCallResult) else "",
            "last_model_revision": (
                last.model_revision if isinstance(last, LLMCallResult) else ""
            ),
        }

    def thread_call_history(self) -> List[LLMCallResult]:
        """Return successful structured calls scoped to this worker thread."""
        return list(getattr(self._call_local, "call_history", []))

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

    def _record_embedding_result(
        self,
        *,
        model: str,
        started_at: float,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        provider: str = "",
        status: str = "success",
        error_type: str = "",
        input_count: int = 1,
        provider_attempt: int = 1,
        latency_sec: Optional[float] = None,
    ) -> None:
        """Record one embedding provider call without input content."""
        exact_model, revision = _split_model_revision(model)
        safe_tokens = max(0, int(tokens_used or 0))
        safe_cost = max(0.0, float(cost_usd or 0.0))
        if status == "success":
            self._total_calls += 1
            self._total_tokens_in += safe_tokens
            self._total_cost_usd += safe_cost
        self._record_call_result(
            LLMCallResult(
                text="",
                model=exact_model,
                model_revision=revision,
                tokens_in=safe_tokens,
                tokens_out=0,
                cost_usd=safe_cost,
                latency_sec=max(
                    0.0,
                    (
                        float(latency_sec)
                        if latency_sec is not None
                        else time.monotonic() - started_at
                    ),
                ),
                finish_reason=(
                    "embedded" if status == "success" else "error"
                ),
                provider_metadata={
                    "provider": provider
                    or (
                        model.split("/", 1)[0]
                        if "/" in model
                        else "litellm"
                    ),
                    "telemetry_stage": "embedding",
                    "input_count": max(0, int(input_count)),
                    "provider_attempt": max(
                        1, int(provider_attempt)
                    ),
                    "error_type": error_type,
                },
                status=status,
                error_type=error_type,
            )
        )

    def embed(
        self,
        text: str,
        model: Optional[str] = None,
        *,
        deadline_at: Optional[float] = None,
        client_max_attempts: Optional[int] = None,
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
        started_at = time.monotonic()
        provider = (
            model.split("/", 1)[0] if "/" in model else "litellm"
        )
        provider_attempt = {"current": 1}
        callback_failures: set[int] = set()

        def record_embedding_failure(
            exc: BaseException,
            attempt: int,
            latency_sec: Optional[float] = None,
            *,
            from_retry_callback: bool = False,
        ) -> None:
            if (
                not from_retry_callback
                and id(exc) in callback_failures
            ):
                return
            if from_retry_callback:
                callback_failures.add(id(exc))
            provider_attempt["current"] = max(
                provider_attempt["current"], int(attempt) + 1
            )
            self._record_embedding_result(
                model=model,
                started_at=started_at,
                provider=provider,
                status="error",
                error_type=type(exc).__name__,
                provider_attempt=attempt,
                latency_sec=latency_sec,
            )

        try:
            if model.startswith("ollama/"):
                import requests # type: ignore
                model_name = model.replace("ollama/", "")
                base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                def request_embedding() -> Any:
                    response = requests.post(
                        f"{base}/api/embeddings",
                        json={"model": model_name, "prompt": text},
                        timeout=remaining_seconds(
                            deadline_at, default=30
                        ),
                    )
                    response.raise_for_status()
                    return response

                resp = _call_with_retry(
                    request_embedding,
                    operation=f"ollama_embed:{model_name}",
                    max_attempts=client_max_attempts,
                    deadline_at=deadline_at,
                    on_attempt_failure=(
                        lambda exc, attempt, latency: (
                            record_embedding_failure(
                                exc,
                                attempt,
                                latency,
                                from_retry_callback=True,
                            )
                        )
                    ),
                )
                body = resp.json()
                vector = body["embedding"]
                self._record_embedding_result(
                    model=model,
                    started_at=started_at,
                    tokens_used=int(
                        body.get("prompt_eval_count", 0) or 0
                    ),
                    provider="ollama",
                    provider_attempt=provider_attempt["current"],
                )
                return vector

            kwargs = self._get_embedding_kwargs(model)

            def request_litellm_embedding() -> Any:
                request_kwargs = dict(kwargs)
                if deadline_at is not None:
                    request_kwargs["timeout"] = remaining_seconds(
                        deadline_at
                    )
                return litellm.embedding(
                    model=model,
                    input=[text],
                    **request_kwargs,
                )

            response = _call_with_retry(
                request_litellm_embedding,
                operation=f"embed:{model}",
                max_attempts=client_max_attempts,
                deadline_at=deadline_at,
                on_attempt_failure=(
                    lambda exc, attempt, latency: (
                        record_embedding_failure(
                            exc,
                            attempt,
                            latency,
                            from_retry_callback=True,
                        )
                    )
                ),
            )
        except NotFoundError as e:
            record_embedding_failure(
                e, provider_attempt["current"]
            )
            raise RuntimeError(
                f"Model '{model}' not found. "
                f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                f"For Ollama: run 'ollama pull <model>'."
            ) from e
        except AuthenticationError as e:
            record_embedding_failure(
                e, provider_attempt["current"]
            )
            raise RuntimeError(
                f"Authentication failed for model '{model}'. "
                f"Check API key in .env: {_key_name_for_model(model)}."
            ) from e
        except APIConnectionError as e:
            record_embedding_failure(
                e, provider_attempt["current"]
            )
            # Ollama surfaces "model not found" as APIConnectionError (HTTP 404 quirk)
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{model}' not found. "
                    f"Check models.yaml or set DEFAULT_LLM_MODEL to an available model. "
                    f"For Ollama: run 'ollama pull <model>'."
                ) from e
            logger.error(
                "LLMClient.embed() failed for model=%s error_type=%s",
                model,
                type(e).__name__,
            )
            raise
        except Exception as exc:
            record_embedding_failure(
                exc, provider_attempt["current"]
            )
            logger.error(
                "LLMClient.embed() failed for model=%s error_type=%s",
                model,
                type(exc).__name__,
            )
            raise

        vector = response.data[0]["embedding"]

        # Track cost
        usage = getattr(response, "usage", None)
        tokens_used = getattr(usage, "total_tokens", 0) if usage else 0
        embed_info = self._embeddings.get(model, {}).get("model_info", {})
        cost = tokens_used * embed_info.get("cost_per_1k_tokens", 0.0) / 1000
        response_model = str(getattr(response, "model", None) or model)
        self._record_embedding_result(
            model=response_model,
            started_at=started_at,
            tokens_used=tokens_used,
            cost_usd=cost,
            provider=provider,
            provider_attempt=provider_attempt["current"],
        )

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
        *,
        deadline_at: Optional[float] = None,
        client_max_attempts: Optional[int] = None,
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
                started_at = time.monotonic()
                batch_provider_attempt = {"current": 1}
                batch_failures: set[int] = set()

                def record_ollama_batch_failure(
                    exc: BaseException,
                    attempt: int,
                    latency_sec: Optional[float] = None,
                    *,
                    from_retry_callback: bool = False,
                ) -> None:
                    if (
                        not from_retry_callback
                        and id(exc) in batch_failures
                    ):
                        return
                    if from_retry_callback:
                        batch_failures.add(id(exc))
                    batch_provider_attempt["current"] = max(
                        batch_provider_attempt["current"],
                        int(attempt) + 1,
                    )
                    self._record_embedding_result(
                        model=model,
                        started_at=started_at,
                        provider="ollama",
                        status="error",
                        error_type=type(exc).__name__,
                        input_count=len(batch),
                        provider_attempt=attempt,
                        latency_sec=latency_sec,
                    )

                def request_batch() -> Any:
                    response = requests.post(
                        f"{base}/api/embed",
                        json={"model": model_name, "input": batch},
                        timeout=remaining_seconds(
                            deadline_at, default=120
                        ),
                    )
                    response.raise_for_status()
                    return response

                try:
                    response = _call_with_retry(
                        request_batch,
                        operation=f"ollama_embed_batch:{model_name}",
                        max_attempts=client_max_attempts,
                        deadline_at=deadline_at,
                        on_attempt_failure=(
                            lambda exc, attempt, latency: (
                                record_ollama_batch_failure(
                                    exc,
                                    attempt,
                                    latency,
                                    from_retry_callback=True,
                                )
                            )
                        ),
                    )
                    body = response.json()
                except Exception as exc:
                    record_ollama_batch_failure(
                        exc, batch_provider_attempt["current"]
                    )
                    raise
                vectors.extend(body["embeddings"])
                self._record_embedding_result(
                    model=model,
                    started_at=started_at,
                    tokens_used=int(
                        body.get("prompt_eval_count", 0) or 0
                    ),
                    provider="ollama",
                    input_count=len(batch),
                    provider_attempt=batch_provider_attempt["current"],
                )
            return vectors

        kwargs = self._get_embedding_kwargs(model)
        all_embeddings: List[List[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            started_at = time.monotonic()
            batch_provider_attempt = {"current": 1}
            batch_failures: set[int] = set()

            def record_batch_failure(
                exc: BaseException,
                attempt: int,
                latency_sec: Optional[float] = None,
                *,
                from_retry_callback: bool = False,
            ) -> None:
                if (
                    not from_retry_callback
                    and id(exc) in batch_failures
                ):
                    return
                if from_retry_callback:
                    batch_failures.add(id(exc))
                batch_provider_attempt["current"] = max(
                    batch_provider_attempt["current"],
                    int(attempt) + 1,
                )
                self._record_embedding_result(
                    model=model,
                    started_at=started_at,
                    status="error",
                    error_type=type(exc).__name__,
                    input_count=len(batch),
                    provider_attempt=attempt,
                    latency_sec=latency_sec,
                )

            try:
                def request_litellm_batch() -> Any:
                    request_kwargs = dict(kwargs)
                    if deadline_at is not None:
                        request_kwargs["timeout"] = remaining_seconds(
                            deadline_at
                        )
                    return litellm.embedding(
                        model=model,
                        input=batch,
                        **request_kwargs,
                    )

                response = _call_with_retry(
                    request_litellm_batch,
                    operation=f"embed_batch:{model}:{batch_num}/{total_batches}",
                    max_attempts=client_max_attempts,
                    deadline_at=deadline_at,
                    on_attempt_failure=(
                        lambda exc, attempt, latency: (
                            record_batch_failure(
                                exc,
                                attempt,
                                latency,
                                from_retry_callback=True,
                            )
                        )
                    ),
                )
            except NotFoundError as e:
                record_batch_failure(
                    e, batch_provider_attempt["current"]
                )
                raise RuntimeError(
                    f"Embedding model '{model}' not found. "
                    f"Check models.yaml or DEFAULT_EMBEDDING_MODEL."
                ) from e
            except AuthenticationError as e:
                record_batch_failure(
                    e, batch_provider_attempt["current"]
                )
                raise RuntimeError(
                    f"Authentication failed for embedding model '{model}'. "
                    f"Check {_key_name_for_model(model)} in .env."
                ) from e
            except Exception as exc:
                record_batch_failure(
                    exc, batch_provider_attempt["current"]
                )
                logger.error(
                    "LLMClient.embed_batch failed at batch %d/%d for "
                    "model=%s error_type=%s",
                    batch_num,
                    total_batches,
                    model,
                    type(exc).__name__,
                )
                raise

            batch_vecs = [item["embedding"] for item in response.data]
            all_embeddings.extend(batch_vecs)

            usage = getattr(response, "usage", None)
            tokens_used = getattr(usage, "total_tokens", 0) if usage else 0
            embed_info = self._embeddings.get(model, {}).get("model_info", {})
            cost = tokens_used * embed_info.get("cost_per_1k_tokens", 0.0) / 1000
            response_model = str(
                getattr(response, "model", None) or model
            )
            self._record_embedding_result(
                model=response_model,
                started_at=started_at,
                tokens_used=tokens_used,
                cost_usd=cost,
                input_count=len(batch),
                provider_attempt=batch_provider_attempt["current"],
            )

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

