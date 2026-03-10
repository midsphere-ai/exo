"""Sandboxed Python code execution for computational problems."""
from __future__ import annotations
import logging
import subprocess
import tempfile
import os
from typing import Any

logger = logging.getLogger("deepsearch")


class CodeSandbox:
    def __init__(self, context: dict | None = None, generate_fn=None, schema_gen=None, max_attempts: int = 3):
        self.context = context or {}
        self.generate_fn = generate_fn
        self.schema_gen = schema_gen
        self.max_attempts = max_attempts

    async def solve(self, issue: str) -> dict:
        """Generate and execute Python code to solve computational problems.

        Returns dict with solution: {code, output} and attempts list.
        """
        attempts = []

        for i in range(self.max_attempts):
            code_response = await self._generate_code(issue, attempts)
            code = code_response.code if hasattr(code_response, 'code') else ""

            result = self._execute_code(code)

            if result["success"]:
                logger.info("Code sandbox solved: %s", issue[:50])
                return {
                    "solution": {"code": code, "output": result["output"]},
                    "attempts": attempts,
                }

            attempts.append({"code": code, "error": result.get("error", "")})
            logger.warning("Code attempt %d failed: %s", i + 1, result.get("error", ""))

        raise RuntimeError(f"Failed to generate working code after {self.max_attempts} attempts")

    async def _generate_code(self, problem: str, previous_attempts: list[dict]) -> Any:
        prev_context = ""
        if previous_attempts:
            prev_context = "\n".join(
                f"Attempt {i+1}:\nCode: {a['code']}\nError: {a.get('error', '')}"
                for i, a in enumerate(previous_attempts)
            )

        system = f"""You are an expert Python programmer. Generate Python code to solve the given problem.

Rules:
1. Generate plain Python code that prints the result
2. Must use print() to output the result
3. No third-party libraries that need installation (only stdlib)
4. Must be self-contained

{f'Previous failed attempts:{chr(10)}{prev_context}' if prev_context else ''}"""

        schema = self.schema_gen.get_code_generator_schema()
        return await self.generate_fn(schema=schema, system=system, prompt=problem)

    def _execute_code(self, code: str) -> dict:
        """Execute Python code in a subprocess sandbox."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            try:
                result = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True, text=True, timeout=30,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )

                if result.returncode == 0:
                    output = result.stdout.strip()
                    if not output:
                        return {"success": False, "error": "No output produced. Use print() to output results."}
                    return {"success": True, "output": output}
                else:
                    return {"success": False, "error": result.stderr.strip()}
            finally:
                os.unlink(tmp_path)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Code execution timed out (30s limit)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
