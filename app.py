"""LangGraph orchestration example with Ollama and optional LLMLingua prompt compression.

Version A: Direct prompt to Ollama.
Version B: Compress the prompt with LLMLingua before calling Ollama.

Langfuse tracing is enabled via the @observe decorator. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
or LANGFUSE_TRACING_ENABLED=False to disable remote tracing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Any, Callable, Dict, Optional

import tiktoken
from langgraph.graph import START, StateGraph
from ollama import Client
from typing_extensions import TypedDict


def _is_langfuse_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
        return False

    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def _noop_observe(*args: Any, **kwargs: Any):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


if _is_langfuse_enabled():
    from langfuse import get_client, observe
else:
    get_client = None  # type: ignore[assignment]
    observe = _noop_observe

PromptCompressor = None  # type: ignore[misc]
LLMLINGUA_AVAILABLE = False


class AppState(TypedDict, total=False):
    user_prompt: str
    model: str
    compressed_prompt: str
    direct_response: str
    compressed_response: str
    direct_tokens: int
    compressed_tokens: int
    compression_ratio: float
    comparison: str


ENCODING = tiktoken.get_encoding("cl100k_base")
OLLAMA_CLIENT = Client(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))


def token_count(text: str) -> int:
    return len(ENCODING.encode(text or ""))


def call_ollama(prompt: str, model: str = "qwen2.5:7b-instruct") -> str:
    response = OLLAMA_CLIENT.generate(model=model, prompt=prompt)
    message = response.get("response", "")
    if not isinstance(message, str):
        raise RuntimeError(f"Unexpected Ollama response format: {response}")
    return message.strip()


@observe(name="orchestrator", as_type="span")
def compare_results(state: AppState, config=None) -> Dict[str, Any]:
    direct_tokens = state.get("direct_tokens", 0)
    compressed_tokens = state.get("compressed_tokens", 0)
    ratio = compressed_tokens / direct_tokens if direct_tokens else 0.0

    summary = (
        f"Direct prompt used {direct_tokens} tokens. "
        f"Compressed prompt used {compressed_tokens} tokens. "
        f"Compression ratio {ratio:.2f}."
    )

    return {
        "compression_ratio": ratio,
        "comparison": summary,
    }


@observe(name="ollama_direct_generation", as_type="generation")
def direct_agent(state: AppState, config=None) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "")
    direct_tokens = token_count(prompt)
    model = state.get("model") or (
        (config or {}).get("configurable", {}).get("model")
        if config
        else None
    ) or "qwen2.5:7b-instruct"
    response = call_ollama(prompt, model=model)

    return {
        "direct_response": response,
        "direct_tokens": direct_tokens,
    }


@observe(name="ollama_compressed_generation", as_type="generation")
def compressed_agent(state: AppState, config=None) -> Dict[str, Any]:
    prompt = state.get("compressed_prompt", state.get("user_prompt", ""))
    model = state.get("model") or (
        (config or {}).get("configurable", {}).get("model")
        if config
        else None
    ) or "qwen2.5:7b-instruct"
    response = call_ollama(prompt, model=model)
    return {"compressed_response": response}


def prepare_state(state: AppState, config=None) -> Dict[str, Any]:
    model = state.get("model") or (
        (config or {}).get("configurable", {}).get("model")
        if config
        else None
    ) or "qwen2.5:7b-instruct"
    return {"user_prompt": state.get("user_prompt", ""), "model": model}


def make_compression_node(
    use_llmlingua: bool,
    compressor: Optional[PromptCompressor],
    rate: float,
) -> Callable[[AppState, Any], Dict[str, Any]]:
    if use_llmlingua:

        @observe(name="llmlingua_compression", as_type="span")
        def llmlingua_compression(state: AppState, config=None) -> Dict[str, Any]:
            prompt = state.get("user_prompt", "")
            if compressor is None:
                raise RuntimeError("LLMLingua compressor was not initialized.")

            compressed = compressor.compress_prompt(
                [prompt],
                instruction="Compress this prompt for faster generation.",
                question="",
                rate=rate,
            )["compressed_prompt"]

            return {
                "compressed_prompt": compressed,
                "compressed_tokens": token_count(compressed),
            }

        return llmlingua_compression

    @observe(name="identity_compression", as_type="span")
    def identity_compression(state: AppState, config=None) -> Dict[str, Any]:
        prompt = state.get("user_prompt", "")
        return {
            "compressed_prompt": prompt,
            "compressed_tokens": token_count(prompt),
        }

    return identity_compression


def build_graph(
    use_llmlingua: bool = False,
    compressor: Optional[PromptCompressor] = None,
    compression_rate: float = 0.5,
    model: str = "qwen2.5:7b-instruct",
) -> Any:
    def prepare_state_with_model(state: AppState, config=None) -> Dict[str, Any]:
        selected_model = state.get("model") or (
            (config or {}).get("configurable", {}).get("model")
            if config
            else None
        ) or model
        return {"user_prompt": state.get("user_prompt", ""), "model": selected_model}

    builder = StateGraph(AppState)
    builder.add_node("prepare_state", prepare_state_with_model)
    builder.add_node("direct_agent", direct_agent)
    builder.add_node("compress_agent", make_compression_node(use_llmlingua, compressor, compression_rate))
    builder.add_node("compressed_agent", compressed_agent)
    builder.add_node("compare_results", compare_results)

    builder.add_edge(START, "prepare_state")
    builder.add_edge("prepare_state", "direct_agent")
    builder.add_edge("prepare_state", "compress_agent")
    builder.add_edge("compress_agent", "compressed_agent")
    builder.add_edge("direct_agent", "compare_results")
    builder.add_edge("compressed_agent", "compare_results")
    builder.set_finish_point("compare_results")

    return builder.compile()


def initialize_llmlingua(
    model_name: str = "gpt2",
    device_map: str = "cpu",
    use_llmlingua2: bool = False,
) -> Any:
    global LLMLINGUA_AVAILABLE, PromptCompressor

    if not LLMLINGUA_AVAILABLE:
        try:
            module = importlib.import_module("llmlingua.prompt_compressor")
            PromptCompressor = module.PromptCompressor
            LLMLINGUA_AVAILABLE = True
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LLMLingua is not installed in the current environment."
            ) from exc

    return PromptCompressor(
        model_name=model_name,
        device_map=device_map,
        use_llmlingua2=use_llmlingua2,
    )


def configure_langfuse() -> None:
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

    if not _is_langfuse_enabled():
        return

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return

    if get_client is None:
        from langfuse import get_client as langfuse_get_client

        globals()["get_client"] = langfuse_get_client

    get_client(public_key=public_key, secret_key=secret_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph orchestration example with Ollama and optional LLMLingua."
    )
    parser.add_argument(
        "--version",
        choices=["a", "b"],
        default="a",
        help="Use version 'a' for direct Ollama or 'b' for LLMLingua compression.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The user prompt to send to the agent graph.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b-instruct",
        help="Local Ollama model name to run.",
    )
    parser.add_argument(
        "--compression-rate",
        type=float,
        default=0.5,
        help="Compression rate for LLMLingua in version b.",
    )
    parser.add_argument(
        "--llmlingua-model",
        default="gpt2",
        help="Hugging Face model name used by LLMLingua for compression.",
    )
    parser.add_argument(
        "--llmlingua-device",
        default="cpu",
        help="Device map used by LLMLingua for prompt compression.",
    )
    parser.add_argument(
        "--llmlingua2",
        action="store_true",
        help="Enable LLMLingua-2 compression if supported.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_langfuse()

    if args.version == "b":
        compressor = initialize_llmlingua(
            model_name=args.llmlingua_model,
            device_map=args.llmlingua_device,
            use_llmlingua2=args.llmlingua2,
        )
        graph = build_graph(
            use_llmlingua=True,
            compressor=compressor,
            compression_rate=args.compression_rate,
        )
    else:
        graph = build_graph(use_llmlingua=False)

    result = graph.invoke(
        {"user_prompt": args.prompt, "model": args.model},
        config={"configurable": {"model": args.model}},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
