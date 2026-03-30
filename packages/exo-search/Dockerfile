FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy full workspace (needed for workspace deps)
COPY . .

# Install all workspace packages
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "exo.search", "--serve", "--host", "0.0.0.0", "--port", "8000"]
