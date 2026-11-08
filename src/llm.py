"""Unified multi-provider language model client.

Supports Groq Cloud, OpenAI, Anthropic Claude, and Google Gemini with
automated Tenacity exponential backoff retries and rate pacing.
"""

import json
import os
from pathlib import Path
import time
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory entries in the sandboxed filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to inspect (e.g. '/' or '/workspace')."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file text contents from sandboxed storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read (e.g. '/workspace/data.txt')."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a sandboxed file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit final task answer to complete execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer to submit."}
                },
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "defer_to_human",
            "description": "Safely halt execution and defer to human operator when requirements cannot be met.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for requesting human intervention."}
                },
                "required": ["reason"],
            },
        },
    },
]

ANTHROPIC_TOOLS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in BENCHMARK_TOOLS
]



class APIFatalError(Exception):
    """Fatal client error (e.g. 400, 401, 404) that should not be retried."""


class APIRetryableError(Exception):
    """Transient server error (e.g. 429, 500, network timeout) that should be retried."""


def _log_retry(retry_state):
    """Print backoff notifications when transient API errors trigger retries."""
    exc = retry_state.outcome.exception()
    sleep_sec = getattr(retry_state.next_action, "sleep", 4.0)
    print(f"  [Retry] {exc} -> retrying in {sleep_sec:.1f}s...")


def load_dotenv(env_path: str = ".env") -> None:
    """Load key-value pairs from a .env file into os.environ if not already set.

    Args:
        env_path: Relative or absolute path to the .env file.
    """
    path = Path(env_path)
    if not path.is_file():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


load_dotenv()


PROVIDER_CONFIGS = {
    "groq": {
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_format": "openai",
        "small": "llama-3.1-8b-instant",
        "large": "llama-3.3-70b-versatile",
        "delay_small": 2.0,
        "delay_large": 10.0,
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_format": "openai",
        "small": "gpt-4o-mini",
        "large": "gpt-4o",
        "delay_small": 1.0,
        "delay_large": 2.0,
    },
    "claude": {
        "env_var": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages",
        "api_format": "anthropic",
        "small": "claude-3-5-haiku-20241022",
        "large": "claude-3-5-sonnet-20241022",
        "delay_small": 1.0,
        "delay_large": 2.0,
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_format": "openai",
        "small": "gemini-1.5-flash",
        "large": "gemini-1.5-pro",
        "delay_small": 1.0,
        "delay_large": 2.0,
    },
}

PROVIDER_ALIASES = {
    "groq": "groq",
    "groq cloud": "groq",
    "openai": "openai",
    "claude": "claude",
    "anthropic": "claude",
    "anthropic claude": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "google gemini": "gemini",
}


def normalize_provider(name: str) -> str:
    """Normalize provider name strings into canonical provider keys.

    Args:
        name: Raw provider identifier.

    Returns:
        Canonical provider key string ('groq', 'openai', 'claude', or 'gemini').

    Raises:
        ValueError: If provider is not supported.
    """
    key = name.strip().lower()
    if key in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[key]
    valid = list(PROVIDER_CONFIGS.keys())
    raise ValueError(f"Unsupported provider '{name}'. Must be one of: {valid}")


class LLMClient:
    """Unified client handling chat completion queries across major providers."""

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        delay: float | None = None,
        scale: str = "small",
    ):
        """Initialize provider endpoint, authentication, and rate pacing delay.

        Args:
            provider: Provider name ('groq', 'openai', 'claude', or 'gemini').
            model: Optional model identifier. Defaults to provider standard model.
            delay: Optional rate pacing sleep interval in seconds.
            scale: Model scale fallback ('small' or 'large') if model not specified.

        Raises:
            ValueError: If provider is unsupported or required API key is unset.
        """
        canonical_provider = normalize_provider(provider)
        config = PROVIDER_CONFIGS[canonical_provider]

        self.provider = canonical_provider
        self.api_format = config["api_format"]

        if model is not None:
            self.model = model
        else:
            self.model = config.get(scale, config["small"])

        if delay is not None:
            self.delay = delay
        else:
            self.delay = config.get(f"delay_{scale}", 2.0)

        legacy_env = config["env_var"]
        self.api_key = (
            os.environ.get("API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get(legacy_env, "")
        )

        if not self.api_key:
            raise ValueError(
                f"Provider '{self.provider}' configured, but API_KEY environment variable "
                f"is not set. Please set API_KEY (or {legacy_env}) or see .env.example."
            )

        self.base_url = os.environ.get("LLM_BASE_URL", config["base_url"])

    @retry(
        retry=retry_if_exception_type(APIRetryableError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(10),
        before_sleep=_log_retry,
    )
    def _call_api(self, messages: list[dict[str, Any]]) -> str:
        """Dispatch query formatted according to provider specification."""
        if self.api_format == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai_compatible(messages)

    def _call_openai_compatible(self, messages: list[dict[str, Any]]) -> str:
        """Query standard OpenAI-compatible completions endpoints (Groq, OpenAI, Gemini)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "tools": BENCHMARK_TOOLS,
        }
        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise APIRetryableError(f"Network error querying {self.provider}: {e}") from e

        if resp.status_code in (429, 500, 502, 503, 504):
            raise APIRetryableError(f"{self.provider.upper()} transient error ({resp.status_code}): {resp.text}")

        if not resp.ok:
            try:
                err_data = resp.json().get("error", {})
                err_msg = err_data.get("message", resp.text)
            except Exception:
                err_msg = resp.text
            raise APIFatalError(f"{self.provider.upper()} error ({resp.status_code}): {err_msg}")

        choice = resp.json()["choices"][0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    fn_args = json.loads(raw_args)
                except Exception:
                    fn_args = {}
            elif isinstance(raw_args, dict):
                fn_args = raw_args
            else:
                fn_args = {}
            return json.dumps({"action": fn_name, **fn_args})

        return msg.get("content") or ""

    def _call_anthropic(self, messages: list[dict[str, Any]]) -> str:
        """Query Anthropic Messages API, extracting system prompt to top-level attribute."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_prompt = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Execute task."}]

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": chat_messages,
            "tools": ANTHROPIC_TOOLS,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise APIRetryableError(f"Network error querying {self.provider}: {e}") from e

        if resp.status_code in (429, 500, 502, 503, 504):
            raise APIRetryableError(f"{self.provider.upper()} transient error ({resp.status_code}): {resp.text}")

        if not resp.ok:
            try:
                err_data = resp.json().get("error", {})
                err_msg = err_data.get("message", resp.text)
            except Exception:
                err_msg = resp.text
            raise APIFatalError(f"{self.provider.upper()} error ({resp.status_code}): {err_msg}")

        content_blocks = resp.json().get("content", [])
        for block in content_blocks:
            if block.get("type") == "tool_use":
                fn_name = block.get("name", "")
                fn_args = block.get("input", {})
                return json.dumps({"action": fn_name, **fn_args})
            if block.get("type") == "text":
                return block.get("text", "")

        return ""

    def __call__(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Execute chat completion and enforce rate pacing delay."""
        result = self._call_api(messages)
        time.sleep(self.delay)
        return result

    @property
    def name(self) -> str:
        """Model identifier tag."""
        return self.model
