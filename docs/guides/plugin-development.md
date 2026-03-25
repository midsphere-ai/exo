# Plugin Development

Exo Web supports a plugin system that lets you extend the platform with custom model providers, tools, agent strategies, and more. Plugins are managed through the dashboard's marketplace UI and executed in isolated subprocesses for safety. This guide walks you through creating, testing, and publishing plugins.

## Plugin Types

Every plugin declares a `type` in its manifest. Exo Web supports five types:

| Type | Purpose | Example |
|---|---|---|
| `model` | Add LLM providers (OpenAI, Ollama, custom endpoints) | Local Llama 3 via Ollama |
| `tool` | Give agents new capabilities (search, file I/O, code execution) | Web search tool |
| `strategy` | Define agent reasoning strategies (ReAct, Tree of Thought) | Multi-path reasoning |
| `extension` | Modify core behavior (caching, observability, prompt transforms) | Response caching layer |
| `bundle` | Curated collection of plugins installed together | Starter pack |

## The `plugin.json` Manifest

Every plugin needs a `plugin.json` file in its root directory. This manifest tells Exo how to load and display the plugin.

```json
{
  "name": "My Custom Tool",
  "version": "1.0.0",
  "type": "tool",
  "entry_point": "main.py",
  "description": "A brief description of what the plugin does",
  "author": "Your Name",
  "permissions": ["network"]
}
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name (1–255 characters) |
| `version` | string | Semantic version (e.g., `1.0.0`, `0.3.2`) |
| `type` | string | One of: `model`, `tool`, `strategy`, `extension`, `bundle` |
| `entry_point` | string | Path to the Python entry point, relative to the plugin directory (default: `main.py`) |

### Optional Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `description` | string | `""` | Human-readable summary shown in the marketplace |
| `author` | string | `""` | Creator name or organization |
| `permissions` | array | `[]` | Permissions the plugin requires (see below) |

### Permissions

Plugins declare the permissions they need. Users see these before installation:

| Permission | Grants |
|---|---|
| `network` | Outbound HTTP/HTTPS requests |
| `api_keys` | Access to configured provider API keys |
| `sandbox` | Sandboxed code execution |
| `file_system` | Read/write access to the working directory |

```json
{
  "permissions": ["network", "api_keys"]
}
```

## Plugin Lifecycle

Plugins move through four status states:

```
install → installed → enable → enabled
                         ↕
                      disabled
                         ↓
                      uninstall (deleted)
```

| Status | Meaning |
|---|---|
| `installed` | Plugin is registered but not active |
| `enabled` | Plugin is active and available to agents |
| `disabled` | Plugin is paused — agents cannot use it |
| `error` | Plugin failed validation or execution |

### State Transitions

- **Install**: Validates `plugin.json`, runs the entry point with `--validate`, stores in database
- **Enable**: Sets status to `enabled`, makes plugin available to agents
- **Disable**: Sets status to `disabled`, plugin remains installed but inactive
- **Uninstall**: Removes plugin record from the database entirely

Use the REST API or dashboard UI to manage lifecycle:

```bash
# Enable a plugin
curl -X PUT /api/v1/plugins/{plugin_id}/status \
  -H "Content-Type: application/json" \
  -d '{"status": "enabled"}'

# Disable a plugin
curl -X PUT /api/v1/plugins/{plugin_id}/status \
  -H "Content-Type: application/json" \
  -d '{"status": "disabled"}'

# Uninstall a plugin
curl -X DELETE /api/v1/plugins/{plugin_id}
```

## Plugin Isolation Model

Plugins run in **isolated subprocesses**, not in the main Exo process. This provides safety guarantees:

- **Process isolation**: Each plugin executes in its own `asyncio.create_subprocess_exec()` call
- **Timeout enforcement**: Plugins have a 10-second execution timeout — exceeded processes are killed
- **Working directory**: The subprocess `cwd` is set to the plugin's directory
- **Output capture**: stdout and stderr are captured and returned for debugging
- **No shared memory**: Plugins cannot access the main process's memory or database directly

During installation, the entry point is called with a `--validate` flag. Your entry point must handle this flag and exit cleanly:

```python
# main.py
import sys

def validate():
    """Check that dependencies are available and config is valid."""
    print("Validation passed")
    return 0

def run():
    """Main plugin logic."""
    print("Plugin running")
    return 0

if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(validate())
    sys.exit(run())
```

## Developing Locally with "Load from Directory"

The fastest way to develop a plugin is to load it directly from a local directory — no packaging or upload needed.

### Step 1: Create Your Plugin Directory

```
my-plugin/
├── plugin.json
├── main.py
└── requirements.txt   # optional
```

### Step 2: Write the Manifest

```json
{
  "name": "Dev Tool",
  "version": "0.1.0",
  "type": "tool",
  "entry_point": "main.py",
  "description": "My tool in development",
  "author": "Dev",
  "permissions": []
}
```

### Step 3: Write the Entry Point

```python
# main.py
import sys
import json

def validate():
    print(json.dumps({"status": "ok", "message": "Validation passed"}))
    return 0

def run():
    # Your plugin logic here
    print(json.dumps({"status": "ok", "result": "Hello from my plugin"}))
    return 0

if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(validate())
    sys.exit(run())
```

### Step 4: Load via API

```bash
curl -X POST /api/v1/plugins/load-directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/absolute/path/to/my-plugin"}'
```

Or use the dashboard: navigate to **Plugins → Load from Directory** and enter the path.

### What Happens on Load

1. Exo reads `plugin.json` from the specified directory
2. Validates required manifest fields and plugin type
3. Checks that the entry point file exists
4. Runs the entry point with `--validate` in an isolated subprocess (10s timeout)
5. If validation passes, stores the plugin in the database with the directory path

You can re-load the directory after making changes — the plugin record is updated in place.

## Tutorial: Creating a Tool Plugin

This tutorial builds a web search tool plugin step by step.

### Directory Structure

```
exo-web-search/
├── plugin.json
├── main.py
└── search.py
```

### plugin.json

```json
{
  "name": "Web Search",
  "version": "1.0.0",
  "type": "tool",
  "entry_point": "main.py",
  "description": "Search the web using DuckDuckGo",
  "author": "Your Name",
  "permissions": ["network"]
}
```

### search.py

```python
"""Core search logic, separated from the entry point."""

import urllib.request
import urllib.parse
import json


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo instant answer API."""
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"https://api.duckduckgo.com/?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "ExoPlugin/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    results = []
    for topic in data.get("RelatedTopics", [])[:max_results]:
        if "Text" in topic:
            results.append({
                "title": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })
    return results
```

### main.py

```python
"""Entry point for the Web Search plugin."""

import sys
import json


def validate() -> int:
    """Validate that dependencies are available."""
    try:
        import urllib.request  # noqa: F401
        print(json.dumps({"status": "ok"}))
        return 0
    except ImportError as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


def run() -> int:
    """Execute a search (reads query from stdin or args)."""
    from search import search_web

    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "exo ai framework"
    try:
        results = search_web(query)
        print(json.dumps({"status": "ok", "results": results}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(validate())
    sys.exit(run())
```

### Test Locally

```bash
# Validate
python main.py --validate
# {"status": "ok"}

# Run a search
python main.py --run "python async patterns"
# {"status": "ok", "results": [...]}
```

### Load Into Exo

```bash
curl -X POST http://localhost:4321/api/v1/plugins/load-directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/home/you/exo-web-search"}'
```

## Tutorial: Creating a Model Provider Plugin

This tutorial builds a custom model provider plugin that wraps a local API.

### Directory Structure

```
exo-local-llm/
├── plugin.json
├── main.py
└── provider.py
```

### plugin.json

```json
{
  "name": "Local LLM Provider",
  "version": "1.0.0",
  "type": "model",
  "entry_point": "main.py",
  "description": "Connect to a local LLM server (llama.cpp, text-generation-webui, etc.)",
  "author": "Your Name",
  "permissions": ["network", "api_keys"]
}
```

### provider.py

```python
"""Model provider implementation for a local LLM endpoint."""

import urllib.request
import json
from typing import Any


class LocalLLMProvider:
    """Wraps a local OpenAI-compatible API (llama.cpp, vLLM, Ollama, etc.)."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        """Send a chat completion request to the local server."""
        payload = json.dumps({
            "model": kwargs.get("model", "default"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def list_models(self) -> list[str]:
        """List available models from the local server."""
        req = urllib.request.Request(f"{self.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return [m["id"] for m in data.get("data", [])]
```

### main.py

```python
"""Entry point for the Local LLM Provider plugin."""

import sys
import json


def validate() -> int:
    """Check that the provider module loads correctly."""
    try:
        from provider import LocalLLMProvider  # noqa: F401
        print(json.dumps({"status": "ok", "message": "Provider module loads correctly"}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


def run() -> int:
    """Test the provider with a simple request."""
    from provider import LocalLLMProvider

    base_url = "http://localhost:8080"
    if len(sys.argv) > 2:
        base_url = sys.argv[2]

    provider = LocalLLMProvider(base_url=base_url)
    try:
        models = provider.list_models()
        print(json.dumps({"status": "ok", "models": models}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(validate())
    sys.exit(run())
```

### Test Locally

```bash
# Validate (no server needed — just checks imports)
python main.py --validate

# Run with a local server
python main.py --run http://localhost:11434
```

## Testing Plugins Before Publishing

### 1. Validate the Manifest

Check that `plugin.json` is well-formed:

```python
import json

with open("plugin.json") as f:
    manifest = json.load(f)

required = {"name", "version", "type", "entry_point"}
missing = required - set(manifest.keys())
assert not missing, f"Missing fields: {missing}"

valid_types = {"model", "tool", "strategy", "extension", "bundle"}
assert manifest["type"] in valid_types, f"Invalid type: {manifest['type']}"

print("Manifest OK")
```

### 2. Run Validation Mode

Every plugin must pass validation — this is what Exo runs during installation:

```bash
python main.py --validate
```

Expected output: JSON with `"status": "ok"`. A non-zero exit code indicates failure.

### 3. Test with Load from Directory

Use the load-directory endpoint to install locally without packaging:

```bash
curl -X POST http://localhost:4321/api/v1/plugins/load-directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/your-plugin"}'
```

This runs the full installation flow: manifest validation, entry point check, subprocess validation.

### 4. Check Plugin Status

After loading, verify the plugin appears and is in the `installed` state:

```bash
curl http://localhost:4321/api/v1/plugins
```

### 5. Enable and Test

Enable the plugin and verify it works with an agent:

```bash
# Enable
curl -X PUT http://localhost:4321/api/v1/plugins/{id}/status \
  -H "Content-Type: application/json" \
  -d '{"status": "enabled"}'
```

### 6. Write Unit Tests

Test your plugin logic independently of Exo:

```python
# test_search.py
from search import search_web

def test_search_returns_results():
    results = search_web("python")
    assert isinstance(results, list)
    for r in results:
        assert "title" in r
        assert "url" in r

def test_validate_exits_zero():
    import subprocess
    result = subprocess.run(
        ["python", "main.py", "--validate"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    import json
    data = json.loads(result.stdout)
    assert data["status"] == "ok"
```

### Common Validation Failures

| Symptom | Cause | Fix |
|---|---|---|
| "Missing required fields" | `plugin.json` missing `name`, `version`, `type`, or `entry_point` | Add all required fields |
| "Invalid type" | `type` not in the allowed set | Use one of: `model`, `tool`, `strategy`, `extension`, `bundle` |
| "Entry point not found" | `entry_point` file doesn't exist | Check the path is relative to the plugin directory |
| "Validation timed out" | Entry point took longer than 10 seconds | Optimize validation — avoid network calls or heavy imports |
| "Non-zero exit code" | Validation function returned an error | Check stderr output for the error details |

## REST API Reference

All plugin endpoints are under `/api/v1/plugins`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/plugins` | List all installed plugins for the current user |
| `GET` | `/api/v1/plugins/marketplace` | Browse the plugin marketplace catalog |
| `GET` | `/api/v1/plugins/marketplace/{id}` | Get marketplace plugin details |
| `POST` | `/api/v1/plugins/install` | Install a plugin from a manifest package |
| `POST` | `/api/v1/plugins/load-directory` | Load a plugin from a local directory (development) |
| `GET` | `/api/v1/plugins/{id}` | Get details of an installed plugin |
| `DELETE` | `/api/v1/plugins/{id}` | Uninstall a plugin |
| `PUT` | `/api/v1/plugins/{id}/status` | Update plugin status (`enabled`, `disabled`) |

## Best Practices

- **Keep validation fast** — avoid network calls or heavy computation in `--validate` mode
- **Use JSON output** — print structured JSON to stdout for consistent parsing
- **Handle errors gracefully** — return non-zero exit codes and error messages, don't crash silently
- **Declare permissions honestly** — only request permissions your plugin actually needs
- **Pin your version** — use semantic versioning and bump on every change
- **Separate concerns** — keep your entry point thin, put logic in separate modules
- **Test the subprocess flow** — your plugin runs as a separate process, so test it that way
