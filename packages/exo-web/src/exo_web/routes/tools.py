"""Tools REST API.

Manages user-defined tools stored in the database.
Also exposes built-in tools merged into listings.
"""

from __future__ import annotations

import ast
import json
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/tools", tags=["tools"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"search", "code", "file", "data", "communication", "custom"}
VALID_TOOL_TYPES = {"function", "http", "schema", "mcp"}

# Built-in tools that are always available
_CODE_INTERPRETER_SCHEMA = json.dumps(
    {
        "type": "object",
        "description": "Execute Python code in a sandboxed environment with access to pandas, numpy, matplotlib, and other common libraries.",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["code"],
    }
)

BUILTIN_TOOLS: list[dict[str, Any]] = [
    {
        "id": "builtin:code_interpreter",
        "name": "code_interpreter",
        "description": "Execute Python code in a sandboxed environment. Has access to pandas, numpy, matplotlib, json, csv, math, and other common libraries. Can generate files (charts, CSVs) and capture stdout/stderr.",
        "category": "code",
        "schema_json": _CODE_INTERPRETER_SCHEMA,
        "code": "",
        "tool_type": "function",
        "usage_count": 0,
        "project_id": "",
        "user_id": "",
        "created_at": "2025-01-01 00:00:00",
    },
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    category: str = Field(description="Category label")
    schema_json: str = Field(description="JSON schema definition")
    code: str = Field(description="Source code")
    tool_type: str = Field(description="Tool type (function, api, etc.)")
    usage_count: int = Field(description="Number of times used")
    project_id: str = Field(description="Associated project identifier")
    user_id: str = Field(description="Owning user identifier")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    description: str = Field("", description="Human-readable description")
    category: str = Field("custom", description="Category label")
    schema_json: str = Field("{}", description="JSON schema definition")
    code: str = Field("", description="Source code")
    tool_type: str = Field("function", description="Tool type (function, api, etc.)")


class ToolUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")
    category: str | None = Field(None, description="Category label")
    schema_json: str | None = Field(None, description="JSON schema definition")
    code: str | None = Field(None, description="Source code")
    tool_type: str | None = Field(None, description="Tool type (function, api, etc.)")


class CustomToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    code: str = Field(..., min_length=1, description="Source code")
    description: str = Field("", description="Human-readable description")


class CustomToolUpdate(BaseModel):
    code: str = Field(..., min_length=1, description="Source code")
    name: str | None = Field(None, min_length=1, max_length=255, description="Display name")
    description: str | None = Field(None, description="Human-readable description")


class CustomToolTestRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict, description="Inputs")


class ParameterInfo(BaseModel):
    name: str = Field(description="Display name")
    type: str = Field(description="Type")
    default: str | None = Field(None, description="Default")
    description: str = Field("", description="Human-readable description")


class CustomToolSchemaResponse(BaseModel):
    tool: ToolResponse = Field(description="Tool")
    parameters: list[ParameterInfo] = Field(description="Parameters")
    schema: dict[str, Any] = Field(description="Schema")


class CustomToolTestResponse(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    output: Any = Field(None, description="Output text or data")
    error: str | None = Field(None, description="Error message if failed")
    traceback: str | None = Field(None, description="Traceback")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, tool_id: str, user_id: str) -> dict[str, Any]:
    """Verify tool exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM tools WHERE id = ? AND user_id = ?",
        (tool_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# AST-based parameter extraction and schema generation
# ---------------------------------------------------------------------------

_AST_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _annotation_to_json_type(node: ast.expr | None) -> str:
    """Convert an AST annotation node to a JSON Schema type string."""
    if node is None:
        return "string"
    if isinstance(node, ast.Name):
        return _AST_TYPE_MAP.get(node.id, "string")
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _AST_TYPE_MAP.get(node.value, "string")
    if isinstance(node, ast.Attribute):
        return "string"
    if isinstance(node, ast.Subscript):
        # Handle list[X], dict[K, V], Optional[X]
        if isinstance(node.value, ast.Name):
            if node.value.id in ("list", "List"):
                return "array"
            if node.value.id in ("dict", "Dict"):
                return "object"
            if node.value.id == "Optional":
                return _annotation_to_json_type(node.slice)
        return "string"
    return "string"


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Parse Google-style Args: section from a docstring into {param: description}."""
    if not docstring:
        return {}
    params: dict[str, str] = {}
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                # Left-aligned non-param line — end of Args section
                break
            if stripped.startswith("returns:") or stripped.startswith("raises:"):
                break
            if ":" in stripped:
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()
                # Handle "param_name (type)" or just "param_name"
                param_name = param_part.split("(")[0].strip()
                if param_name:
                    params[param_name] = parts[1].strip()
    return params


def _extract_function_info(
    code: str,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str | None]:
    """Parse code and find the first decorated function (with @tool) or the first function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    # Prefer the first @tool-decorated function, fall back to first function
    first_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if first_fn is None:
                first_fn = node
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "tool":
                    docstring = ast.get_docstring(node)
                    return node, docstring
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "tool"
                ):
                    docstring = ast.get_docstring(node)
                    return node, docstring

    if first_fn is not None:
        return first_fn, ast.get_docstring(first_fn)
    return None, None


def _generate_schema_from_code(code: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate JSON schema and parameter list from Python tool code.

    Returns (schema_dict, parameters_list).
    """
    func_node, docstring = _extract_function_info(code)
    if func_node is None:
        raise HTTPException(
            status_code=422,
            detail="No function found in code. Define a function (optionally decorated with @tool).",
        )

    doc_params = _parse_docstring_params(docstring)

    properties: dict[str, Any] = {}
    required: list[str] = []
    parameters: list[dict[str, Any]] = []

    for arg in func_node.args.args:
        if arg.arg == "self":
            continue
        param_name = arg.arg
        json_type = _annotation_to_json_type(arg.annotation)
        param_desc = doc_params.get(param_name, "")

        prop: dict[str, Any] = {"type": json_type}
        if param_desc:
            prop["description"] = param_desc

        properties[param_name] = prop

        # Check for default value
        default_val: str | None = None
        # In AST, defaults are stored right-aligned: last N defaults match last N args
        num_defaults = len(func_node.args.defaults)
        num_args = len(func_node.args.args)
        arg_index = func_node.args.args.index(arg)
        default_index = arg_index - (num_args - num_defaults)

        if default_index >= 0:
            default_node = func_node.args.defaults[default_index]
            if isinstance(default_node, ast.Constant):
                default_val = repr(default_node.value)
            elif isinstance(default_node, ast.Name) and default_node.id == "None":
                default_val = "None"
        else:
            required.append(param_name)

        parameters.append(
            {
                "name": param_name,
                "type": json_type,
                "default": default_val,
                "description": param_desc,
            }
        )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    if docstring:
        # Use first line of docstring as description
        first_line = docstring.strip().split("\n")[0].strip()
        if first_line:
            schema["description"] = first_line

    return schema, parameters


def _execute_tool_code(code: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute tool code in a restricted namespace and return result or error."""
    func_node, _ = _extract_function_info(code)
    if func_node is None:
        return {"success": False, "error": "No function found in code", "traceback": None}

    func_name = func_node.name

    # Build a restricted namespace with safe builtins
    safe_builtins = (
        {
            k: v
            for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k
            not in ("exec", "eval", "compile", "__import__", "open", "breakpoint", "exit", "quit")
        }
        if isinstance(__builtins__, dict)
        else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k
            not in ("exec", "eval", "compile", "__import__", "open", "breakpoint", "exit", "quit")
        }
    )

    # Provide a no-op @tool decorator so user code doesn't error
    def _noop_tool(fn: Any = None, **_kwargs: Any) -> Any:
        if fn is not None:
            return fn
        return lambda f: f

    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "tool": _noop_tool}

    try:
        exec(compile(code, "<custom_tool>", "exec"), namespace)
    except Exception:
        return {
            "success": False,
            "error": "Failed to compile tool code",
            "traceback": traceback.format_exc(),
        }

    if func_name not in namespace:
        return {
            "success": False,
            "error": f"Function '{func_name}' not found after execution",
            "traceback": None,
        }

    func = namespace[func_name]
    try:
        result = func(**inputs)
        return {"success": True, "output": result}
    except Exception:
        return {
            "success": False,
            "error": "Tool execution failed",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Endpoints — custom tool routes (must be before /{tool_id} param routes)
# ---------------------------------------------------------------------------


@router.post("/custom", response_model=CustomToolSchemaResponse, status_code=201)
async def create_custom_tool(
    body: CustomToolCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a custom tool from Python code with auto-generated schema."""
    schema, parameters = _generate_schema_from_code(body.code)
    schema_json_str = json.dumps(schema)

    # Use description from body, or fall back to schema description
    description = body.description or schema.get("description", "")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        tool_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO tools (
                id, name, description, category, schema_json, code,
                tool_type, usage_count, project_id, user_id, created_at
            ) VALUES (?, ?, ?, 'custom', ?, ?, 'function', 0, ?, ?, ?)
            """,
            (
                tool_id,
                body.name,
                description,
                schema_json_str,
                body.code,
                body.project_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return {
            "tool": _row_to_dict(row),
            "parameters": parameters,
            "schema": schema,
        }


@router.put("/custom/{tool_id}", response_model=CustomToolSchemaResponse)
async def update_custom_tool(
    tool_id: str,
    body: CustomToolUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a custom tool's code and regenerate schema."""
    schema, parameters = _generate_schema_from_code(body.code)
    schema_json_str = json.dumps(schema)

    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])

        updates: dict[str, str] = {
            "code": body.code,
            "schema_json": schema_json_str,
        }
        if body.name is not None:
            updates["name"] = body.name
        if body.description is not None:
            updates["description"] = body.description
        else:
            # Update description from schema if not explicitly provided
            desc = schema.get("description", "")
            if desc:
                updates["description"] = desc

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), tool_id]

        await db.execute(
            f"UPDATE tools SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return {
            "tool": _row_to_dict(row),
            "parameters": parameters,
            "schema": schema,
        }


@router.post("/custom/{tool_id}/test", response_model=CustomToolTestResponse)
async def test_custom_tool(
    tool_id: str,
    body: CustomToolTestRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Test a custom tool by executing it with sample inputs."""
    async with get_db() as db:
        tool_row = await _verify_ownership(db, tool_id, user["id"])

    code = tool_row["code"]
    if not code.strip():
        raise HTTPException(status_code=422, detail="Tool has no code to execute")

    return _execute_tool_code(code, body.inputs)


@router.post("/schema", response_model=dict[str, Any])
async def generate_schema(
    body: CustomToolUpdate,
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Generate JSON schema from Python code without saving."""
    schema, parameters = _generate_schema_from_code(body.code)
    return {"schema": schema, "parameters": parameters}


class VisualSchemaCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    project_id: str = Field(..., min_length=1, description="Associated project identifier")
    description: str = Field("", description="Human-readable description")
    schema: dict[str, Any] = Field(default_factory=dict, description="Schema")
    tool_type: str = Field("schema", description="Tool type (function, api, etc.)")
    http_config: dict[str, Any] | None = Field(None, description="Http config")


class WorkflowNodeToolResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="Human-readable description")
    node_id: str = Field(description="Associated node identifier")
    node_type: str = Field(description="Node type")
    workflow_id: str = Field(description="Associated workflow identifier")
    workflow_name: str = Field(description="Workflow name")
    schema_json: str = Field(description="JSON schema definition")
    source: str = Field("workflow_node", description="Source")


# Handle data type → JSON Schema type mapping (mirrors handleTypes.ts)
_HANDLE_TYPE_MAP: dict[str, str] = {
    "message": "string",
    "text": "string",
    "number": "number",
    "boolean": "boolean",
    "json": "object",
    "any": "string",
}


def _schema_from_node(node_data: dict[str, Any], node_type: str) -> dict[str, Any]:
    """Generate a JSON Schema for a workflow node based on its handle types.

    Input handles become properties, output handles define the return description.
    """
    # Replicate the handle spec logic from handleTypes.ts
    handle_specs = _NODE_HANDLE_MAP.get(
        node_type,
        [
            {"id": "input", "type": "target", "dataType": "any"},
            {"id": "output", "type": "source", "dataType": "any"},
        ],
    )

    inputs = [h for h in handle_specs if h["type"] == "target"]
    outputs = [h for h in handle_specs if h["type"] == "source"]

    properties: dict[str, Any] = {}
    required: list[str] = []
    for h in inputs:
        json_type = _HANDLE_TYPE_MAP.get(h["dataType"], "string")
        properties[h["id"]] = {
            "type": json_type,
            "description": f"{h['dataType']} input",
        }
        required.append(h["id"])

    output_desc = (
        ", ".join(f"{h.get('label', h['id'])} ({h['dataType']})" for h in outputs)
        if outputs
        else "no output"
    )

    label = node_data.get("label", node_type)
    schema: dict[str, Any] = {
        "type": "object",
        "description": f"Executes workflow node '{label}'. Returns: {output_desc}.",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


# Subset of handle specs from handleTypes.ts (server-side mirror)
_NODE_HANDLE_MAP: dict[str, list[dict[str, Any]]] = {
    "chat_input": [{"id": "output", "type": "source", "dataType": "message"}],
    "webhook": [{"id": "output", "type": "source", "dataType": "json"}],
    "schedule": [{"id": "output", "type": "source", "dataType": "json"}],
    "manual": [{"id": "output", "type": "source", "dataType": "message"}],
    "llm_call": [
        {"id": "input", "type": "target", "dataType": "message"},
        {"id": "output", "type": "source", "dataType": "text"},
    ],
    "prompt_template": [
        {"id": "input", "type": "target", "dataType": "any"},
        {"id": "output", "type": "source", "dataType": "text"},
    ],
    "agent_node": [
        {"id": "input", "type": "target", "dataType": "message"},
        {"id": "output", "type": "source", "dataType": "message"},
    ],
    "function_tool": [
        {"id": "input", "type": "target", "dataType": "json"},
        {"id": "output", "type": "source", "dataType": "json"},
    ],
    "http_request": [
        {"id": "input", "type": "target", "dataType": "any"},
        {"id": "output", "type": "source", "dataType": "json"},
    ],
    "code_python": [
        {"id": "input", "type": "target", "dataType": "any"},
        {"id": "output", "type": "source", "dataType": "any"},
    ],
    "code_javascript": [
        {"id": "input", "type": "target", "dataType": "any"},
        {"id": "output", "type": "source", "dataType": "any"},
    ],
    "conditional": [
        {"id": "input", "type": "target", "dataType": "any"},
        {"id": "output-true", "type": "source", "dataType": "any", "label": "True"},
        {"id": "output-false", "type": "source", "dataType": "any", "label": "False"},
    ],
    "knowledge_retrieval": [
        {"id": "input", "type": "target", "dataType": "text"},
        {"id": "output", "type": "source", "dataType": "json"},
    ],
    "approval_gate": [
        {"id": "input", "type": "target", "dataType": "any"},
        {"id": "output", "type": "source", "dataType": "any"},
    ],
    "chat_response": [{"id": "input", "type": "target", "dataType": "message"}],
    "api_response": [{"id": "input", "type": "target", "dataType": "json"}],
}


@router.get("/workflow-nodes", response_model=list[WorkflowNodeToolResponse])
async def list_workflow_node_tools(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return workflow nodes that have Tool Mode enabled as available tools."""
    async with get_db() as db:
        conditions = ["w.user_id = ?"]
        params: list[str] = [user["id"]]
        if project_id:
            conditions.append("w.project_id = ?")
            params.append(project_id)

        where = " AND ".join(conditions)
        cursor = await db.execute(
            f"SELECT w.id, w.name, w.nodes_json FROM workflows w WHERE {where}",
            params,
        )
        rows = await cursor.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        workflow = _row_to_dict(row)
        try:
            nodes = json.loads(workflow.get("nodes_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            continue

        for node in nodes:
            data = node.get("data", {})
            if not data.get("tool_mode"):
                continue

            node_id = node.get("id", "")
            node_type = data.get("nodeType", "default")
            label = data.get("label", node_type)
            schema = _schema_from_node(data, node_type)

            # Tool name: sanitize label to a valid identifier
            tool_name = label.lower().replace(" ", "_").replace("-", "_")

            results.append(
                {
                    "id": f"wfn_{workflow['id']}_{node_id}",
                    "name": tool_name,
                    "description": schema.get("description", f"Workflow node: {label}"),
                    "node_id": node_id,
                    "node_type": node_type,
                    "workflow_id": workflow["id"],
                    "workflow_name": workflow["name"],
                    "schema_json": json.dumps(schema),
                    "source": "workflow_node",
                }
            )

    return results


@router.post("/schema/save", response_model=ToolResponse, status_code=201)
async def save_visual_schema(
    body: VisualSchemaCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Save a tool defined by a visual schema editor."""
    tool_type = body.tool_type if body.tool_type in VALID_TOOL_TYPES else "schema"

    # Merge HTTP config into schema if provided
    schema = dict(body.schema)
    if body.http_config:
        schema["_http"] = body.http_config

    schema_json_str = json.dumps(schema)

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        tool_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO tools (
                id, name, description, category, schema_json, code,
                tool_type, usage_count, project_id, user_id, created_at
            ) VALUES (?, ?, ?, 'custom', ?, '', ?, 0, ?, ?, ?)
            """,
            (
                tool_id,
                body.name,
                body.description,
                schema_json_str,
                tool_type,
                body.project_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints — generic tool CRUD
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ToolResponse])
async def list_tools(
    category: str | None = Query(None),
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all tools for the current user, optionally filtered by category and/or project.

    Includes built-in tools (e.g. code_interpreter) merged with user-defined tools.
    """
    # Start with built-in tools
    results: list[dict[str, Any]] = []
    for bt in BUILTIN_TOOLS:
        if category and bt["category"] != category:
            continue
        results.append(bt)

    async with get_db() as db:
        conditions = ["user_id = ?"]
        params: list[str] = [user["id"]]

        if category:
            conditions.append("category = ?")
            params.append(category)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where = " AND ".join(conditions)
        cursor = await db.execute(
            f"SELECT * FROM tools WHERE {where} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()
        results.extend(_row_to_dict(r) for r in rows)

    return results


@router.post("", response_model=ToolResponse, status_code=201)
async def create_tool(
    body: ToolCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new tool."""
    if body.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )
    if body.tool_type not in VALID_TOOL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tool_type. Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}",
        )

    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        tool_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO tools (
                id, name, description, category, schema_json, code,
                tool_type, usage_count, project_id, user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                tool_id,
                body.name,
                body.description,
                body.category,
                body.schema_json,
                body.code,
                body.tool_type,
                body.project_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single tool by ID with full detail including schema."""
    async with get_db() as db:
        return await _verify_ownership(db, tool_id, user["id"])


@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: str,
    body: ToolUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a tool's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "category" in updates and updates["category"] not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )
    if "tool_type" in updates and updates["tool_type"] not in VALID_TOOL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tool_type. Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}",
        )

    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), tool_id]

        await db.execute(
            f"UPDATE tools SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{tool_id}", status_code=204)
async def delete_tool(
    tool_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a tool."""
    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])
        await db.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
        await db.commit()
