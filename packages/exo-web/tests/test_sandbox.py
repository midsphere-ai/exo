"""Tests for the sandboxed code execution service."""

from __future__ import annotations

from exo_web.services.sandbox import _build_runner_script, execute_code


class TestSandboxEscapeSequences:
    """Test that special characters in user code are safely serialized."""

    def test_sandbox_escape_sequences(self) -> None:
        """Code containing special characters executes safely without injection."""
        # The key requirement: code with \r, \t, \0, embedded quotes, etc.
        # must execute successfully (not inject, not crash the runner script).
        #
        # Note: subprocess.run(text=True) applies universal newline translation,
        # so \r is normalized to \n in captured stdout — we verify execution success
        # and correct output content, not the raw bytes of control characters.
        tricky_cases = [
            # carriage return — executes successfully, produces "hello" in output
            ("print('hello\\r')", "hello"),
            # tab character — tab is preserved in text output
            ("print('hello\\tworld')", "hello\tworld"),
            # embedded single quotes (was broken before repr() fix)
            ('print("it\'s fine")', "it's fine"),
        ]

        for code, expected_substr in tricky_cases:
            result = execute_code(code)
            assert result.success, (
                f"Code failed: {code!r}\nstderr: {result.stderr}\nerror: {result.error}"
            )
            assert expected_substr in result.stdout

    def test_sandbox_null_bytes(self) -> None:
        """Code containing null bytes is handled safely."""
        # A null byte in a string literal — repr() handles this correctly
        code = 'x = "hello\\x00world"\nprint(len(x))'
        result = execute_code(code)
        assert result.success, f"stderr: {result.stderr}"
        assert "11" in result.stdout

    def test_sandbox_unicode_escape(self) -> None:
        """Code containing unicode escape sequences (\\u0041 = 'A') executes correctly."""
        code = 'print("\\u0041")'
        result = execute_code(code)
        assert result.success, f"stderr: {result.stderr}"
        assert "A" in result.stdout

    def test_sandbox_embedded_triple_quotes(self) -> None:
        """Code containing embedded triple-quotes does not break the generated script."""
        code = "x = '''triple quoted'''\nprint(x)"
        result = execute_code(code)
        assert result.success, f"stderr: {result.stderr}"
        assert "triple quoted" in result.stdout

    def test_sandbox_no_injection_via_quote_escape(self) -> None:
        """Crafted payload with quote+newline cannot escape the string literal context."""
        # This was the original vulnerability: a payload like:
        #   '\nexec("import os; os.system(...)")\nx = '
        # would have closed the string and injected arbitrary code.
        malicious = "'; import os; os.system('echo INJECTED'); x = '"
        code = f"user_input = {malicious!r}\nprint('safe')"
        result = execute_code(code)
        # The script should either succeed safely or fail with a syntax/runtime error
        # but must NOT have executed the injected command (INJECTED must not be in stdout)
        assert "INJECTED" not in result.stdout
        assert "INJECTED" not in result.stderr

    def test_build_runner_script_uses_repr(self) -> None:
        """_build_runner_script uses repr() so special chars are safely embedded."""
        tricky_code = "x = 'it\\'s \\r\\n\\t\\x00 \\'tricky\\''\nprint(x)"
        script = _build_runner_script(tricky_code, "/tmp", ["json"])
        # The repr of the code string should appear in the generated script
        assert repr(tricky_code) in script
        # The old naive single-quote embedding pattern must NOT appear
        assert f"= '{tricky_code}'" not in script
