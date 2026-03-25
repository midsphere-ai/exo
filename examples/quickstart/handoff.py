"""Handoff swarm — dynamic agent delegation.

Demonstrates ``Swarm(mode="handoff")`` where agents delegate
to each other dynamically.  The first agent runs and can hand off
to any agent listed in its ``handoffs``.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/handoff.py
"""

from exo import Agent, Swarm, run

# Specialist agents
billing = Agent(
    name="billing",
    model="openai:gpt-4o-mini",
    instructions="You are a billing specialist. Answer billing questions.",
)

tech_support = Agent(
    name="tech_support",
    model="openai:gpt-4o-mini",
    instructions="You are a tech-support specialist. Troubleshoot issues.",
)

# Triage agent — decides which specialist to hand off to
triage = Agent(
    name="triage",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a customer-service triage agent. "
        "For billing questions, respond with exactly: billing\n"
        "For technical issues, respond with exactly: tech_support\n"
        "For anything else, answer directly."
    ),
    handoffs=[billing, tech_support],
)

swarm = Swarm(
    agents=[triage, billing, tech_support],
    mode="handoff",
)

if __name__ == "__main__":
    result = run.sync(swarm, "I was charged twice on my last invoice.")
    print(result.output)
