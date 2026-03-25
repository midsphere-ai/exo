"""Deploy an agent as a web service via A2A protocol.

Demonstrates ``A2AServer`` with custom skills, streaming support,
and agent card discovery at ``/.well-known/agent-card``.

Usage:
    pip install fastapi uvicorn
    export OPENAI_API_KEY=sk-...
    uv run python examples/advanced/web_deploy.py

Then visit:
    http://localhost:8000/.well-known/agent-card   — agent metadata
    http://localhost:8000/docs                     — interactive API docs
"""

from exo import Agent, tool
from exo.a2a.server import A2AServer, AgentExecutor  # pyright: ignore[reportMissingImports]
from exo.a2a.types import AgentSkill, ServingConfig  # pyright: ignore[reportMissingImports]

# --- Define the agent and its tools ------------------------------------------


@tool
async def get_stock_price(symbol: str) -> str:
    """Look up a stock price by ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL').
    """
    # Stub — replace with a real API call.
    prices = {"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 415.60}
    price = prices.get(symbol.upper())
    if price is None:
        return f"Unknown ticker: {symbol}"
    return f"{symbol.upper()}: ${price:.2f}"


@tool
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g. 'USD').
        to_currency: Target currency code (e.g. 'EUR').
    """
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "USD_GBP": 0.79, "GBP_USD": 1.27}
    key = f"{from_currency.upper()}_{to_currency.upper()}"
    rate = rates.get(key)
    if rate is None:
        return f"No rate available for {from_currency} -> {to_currency}"
    result = amount * rate
    return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()}"


agent = Agent(
    name="finance-bot",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a helpful financial assistant. "
        "Use get_stock_price for stock lookups and convert_currency for FX."
    ),
    tools=[get_stock_price, convert_currency],
)

# --- Configure and launch the A2A server ------------------------------------

config = ServingConfig(
    host="0.0.0.0",
    port=8000,
    streaming=True,
    skills=(
        AgentSkill(
            id="stock-lookup",
            name="Stock Price Lookup",
            description="Look up current stock prices by ticker symbol.",
            tags=("finance", "stocks"),
        ),
        AgentSkill(
            id="currency-convert",
            name="Currency Conversion",
            description="Convert amounts between currencies.",
            tags=("finance", "fx"),
        ),
    ),
)

server = A2AServer(executor=AgentExecutor(agent, streaming=True), config=config)
app = server.build_app()

if __name__ == "__main__":
    import uvicorn

    print(f"Agent card: {server.agent_card.model_dump_json(indent=2)}")
    print("Starting server at http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
