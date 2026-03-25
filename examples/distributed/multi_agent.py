"""Multi-agent Swarm running via distributed execution.

A three-agent workflow pipeline (researcher -> drafter -> reviewer)
is submitted to the distributed queue.  Events from each agent are
streamed back with the originating agent's name.

Prerequisites:
    # Terminal 1 — start Redis
    docker run -p 6379:6379 redis:7

    # Terminal 2 — start a worker
    export EXO_REDIS_URL=redis://localhost:6379
    exo start worker

Usage:
    export OPENAI_API_KEY=sk-...
    export EXO_REDIS_URL=redis://localhost:6379
    uv run python examples/distributed/multi_agent.py
"""

import asyncio

from exo import Agent, Swarm
from exo.distributed import distributed  # pyright: ignore[reportMissingImports]
from exo.types import StatusEvent, StepEvent, TextEvent

# -- Define a three-stage workflow Swarm -----------------------------------

researcher = Agent(
    name="researcher",
    model="openai:gpt-4o-mini",
    instructions="You research a topic and return 3 concise bullet points.",
)

drafter = Agent(
    name="drafter",
    model="openai:gpt-4o-mini",
    instructions=(
        "You take bullet-point research and write a short, "
        "engaging paragraph (3-4 sentences)."
    ),
)

reviewer = Agent(
    name="reviewer",
    model="openai:gpt-4o-mini",
    instructions=(
        "You review text for clarity and grammar. "
        "Return the polished final version."
    ),
)

pipeline = Swarm(
    agents=[researcher, drafter, reviewer],
    flow="researcher >> drafter >> reviewer",
    mode="workflow",
)


# -- Submit to the distributed queue and stream events ---------------------

async def main() -> None:
    handle = await distributed(
        pipeline,
        "Explain the benefits of distributed agent execution.",
        detailed=True,
    )

    print(f"Task submitted: {handle.task_id}")
    print("Pipeline: researcher >> drafter >> reviewer\n")

    current_agent = ""

    async for event in handle.stream():
        match event:
            # Show which agent is active
            case StatusEvent() if event.status == "running":
                if event.agent_name != current_agent:
                    current_agent = event.agent_name
                    print(f"\n{'='*50}")
                    print(f"  Agent: {current_agent}")
                    print(f"{'='*50}")

            case StepEvent() if event.status == "started":
                print(f"  [step {event.step_number}]")

            case TextEvent():
                print(event.text, end="", flush=True)

            case StatusEvent() if event.status == "completed":
                print("\n\n[done] Pipeline completed.")

    # Retrieve the final result
    result = await handle.result()
    print(f"\nFinal output:\n{result.get('output', '')}")


if __name__ == "__main__":
    asyncio.run(main())
