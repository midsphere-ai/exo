"""Use the tracing system — Exo quickstart.

Demonstrates the ``@traced`` decorator, manual span context managers,
and the ``PromptLogger`` for structured LLM execution logging.

Usage:
    uv run python examples/quickstart/use_trace.py
"""

import asyncio
import logging

from exo.trace.config import TraceBackend, TraceConfig  # pyright: ignore[reportMissingImports]
from exo.trace.decorator import span_async, traced  # pyright: ignore[reportMissingImports]
from exo.trace.prompt_logger import PromptLogger  # pyright: ignore[reportMissingImports]

logging.basicConfig(level=logging.INFO)


# --- @traced decorator: auto-creates an OTel span around the function ------


@traced("fetch-data", extract_args=True)
async def fetch_data(city: str) -> str:
    """Simulate fetching data — the call is automatically traced."""
    await asyncio.sleep(0.01)
    return f"Weather data for {city}: sunny, 22 °C"


# --- Manual span context manager -------------------------------------------


async def process_data(raw: str) -> str:
    """Process data inside a manual span."""
    async with span_async("process-data", attributes={"input_length": len(raw)}):
        return raw.upper()


# --- PromptLogger: structured LLM execution logging ------------------------


def demo_prompt_logger() -> None:
    """Show how PromptLogger computes token breakdowns."""
    logger = PromptLogger()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    entry = logger.log_execution(
        messages,
        agent_name="demo-agent",
        model_name="gpt-4o-mini",
        context_window=128_000,
        duration_s=0.42,
    )
    print(f"\n--- Prompt Logger Summary ---\n{entry.format_summary()}")


# --- Configuration ---------------------------------------------------------


def demo_trace_config() -> None:
    """Show TraceConfig for different backends."""
    console_cfg = TraceConfig(backend=TraceBackend.CONSOLE, service_name="my-app")
    print(f"\nConsole config: backend={console_cfg.backend}, service={console_cfg.service_name}")

    otlp_cfg = TraceConfig(
        backend=TraceBackend.OTLP,
        endpoint="http://localhost:4318",
        sample_rate=0.5,
    )
    print(f"OTLP config:    endpoint={otlp_cfg.endpoint}, sample_rate={otlp_cfg.sample_rate}")


# --- Main ------------------------------------------------------------------


async def main() -> None:
    raw = await fetch_data("Tokyo")
    result = await process_data(raw)
    print(f"Processed: {result[:40]}...")

    demo_prompt_logger()
    demo_trace_config()


if __name__ == "__main__":
    asyncio.run(main())
