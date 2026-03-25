"""Tests for exo._internal.expression — safe AST expression evaluator."""

from __future__ import annotations

import pytest

from exo._internal.expression import (
    MAX_AST_DEPTH,
    ExpressionError,
    evaluate_expression,
)

# ---------------------------------------------------------------------------
# Valid expressions
# ---------------------------------------------------------------------------


class TestBasicExpressions:
    def test_simple_comparison(self) -> None:
        assert evaluate_expression("x > 5", {"x": 10}) is True

    def test_equality(self) -> None:
        assert evaluate_expression("x == 42", {"x": 42}) is True

    def test_string_equality(self) -> None:
        assert evaluate_expression('name == "alice"', {"name": "alice"}) is True

    def test_boolean_and(self) -> None:
        assert evaluate_expression("x > 0 and y > 0", {"x": 1, "y": 2}) is True

    def test_boolean_or(self) -> None:
        assert evaluate_expression("x > 0 or y > 0", {"x": -1, "y": 2}) is True

    def test_not_operator(self) -> None:
        assert evaluate_expression("not x", {"x": False}) is True

    def test_arithmetic_add(self) -> None:
        assert evaluate_expression("x + y", {"x": 3, "y": 4}) == 7

    def test_arithmetic_multiply(self) -> None:
        assert evaluate_expression("x * 2", {"x": 5}) == 10

    def test_arithmetic_divide(self) -> None:
        assert evaluate_expression("x / 2", {"x": 10}) == 5.0

    def test_arithmetic_modulo(self) -> None:
        assert evaluate_expression("x % 3", {"x": 10}) == 1

    def test_chained_comparison(self) -> None:
        assert evaluate_expression("0 < x < 10", {"x": 5}) is True
        assert evaluate_expression("0 < x < 10", {"x": 15}) is False

    def test_string_concatenation(self) -> None:
        assert evaluate_expression('a + " " + b', {"a": "hello", "b": "world"}) == "hello world"

    def test_in_operator(self) -> None:
        assert evaluate_expression('"a" in items', {"items": ["a", "b", "c"]}) is True

    def test_not_in_operator(self) -> None:
        assert evaluate_expression('"z" not in items', {"items": ["a", "b"]}) is True

    def test_dict_indexing(self) -> None:
        assert evaluate_expression('d["key"]', {"d": {"key": 42}}) == 42

    def test_list_indexing(self) -> None:
        assert evaluate_expression("items[0]", {"items": [10, 20, 30]}) == 10

    def test_negative_index(self) -> None:
        assert evaluate_expression("items[-1]", {"items": [10, 20, 30]}) == 30

    def test_nested_dict(self) -> None:
        assert evaluate_expression('d["a"]["b"]', {"d": {"a": {"b": 99}}}) == 99

    def test_ternary_expression(self) -> None:
        assert evaluate_expression('"yes" if x else "no"', {"x": True}) == "yes"
        assert evaluate_expression('"yes" if x else "no"', {"x": False}) == "no"

    def test_none_comparison(self) -> None:
        assert evaluate_expression("x is None", {"x": None}) is True
        assert evaluate_expression("x is not None", {"x": 42}) is True

    def test_literal_list(self) -> None:
        assert evaluate_expression("[1, 2, 3]") == [1, 2, 3]

    def test_literal_dict(self) -> None:
        assert evaluate_expression('{"a": 1}') == {"a": 1}

    def test_literal_tuple(self) -> None:
        assert evaluate_expression("(1, 2)") == (1, 2)

    def test_builtin_len(self) -> None:
        assert evaluate_expression("len(items)", {"items": [1, 2, 3]}) == 3

    def test_builtin_abs(self) -> None:
        assert evaluate_expression("abs(x)", {"x": -5}) == 5

    def test_builtin_min_max(self) -> None:
        assert evaluate_expression("min(a, b)", {"a": 3, "b": 7}) == 3
        assert evaluate_expression("max(a, b)", {"a": 3, "b": 7}) == 7

    def test_builtin_int_cast(self) -> None:
        assert evaluate_expression('int("42")') == 42

    def test_builtin_str_cast(self) -> None:
        assert evaluate_expression("str(42)") == "42"

    def test_string_method(self) -> None:
        assert evaluate_expression("name.upper()", {"name": "alice"}) == "ALICE"

    def test_string_startswith(self) -> None:
        assert evaluate_expression('name.startswith("al")', {"name": "alice"}) is True

    def test_empty_variables(self) -> None:
        assert evaluate_expression("1 + 2") == 3

    def test_boolean_literal(self) -> None:
        assert evaluate_expression("True") is True
        assert evaluate_expression("False") is False


# ---------------------------------------------------------------------------
# JS-style normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_js_and(self) -> None:
        assert evaluate_expression("x > 0 && y > 0", {"x": 1, "y": 2}) is True

    def test_js_or(self) -> None:
        assert evaluate_expression("x > 0 || y > 0", {"x": -1, "y": 2}) is True

    def test_js_strict_equal(self) -> None:
        assert evaluate_expression("x === 5", {"x": 5}) is True

    def test_js_strict_not_equal(self) -> None:
        assert evaluate_expression("x !== 5", {"x": 3}) is True

    def test_js_true_false(self) -> None:
        assert evaluate_expression("true") is True
        assert evaluate_expression("false") is False

    def test_js_null(self) -> None:
        assert evaluate_expression("null") is None

    def test_js_undefined(self) -> None:
        assert evaluate_expression("undefined") is None


# ---------------------------------------------------------------------------
# Attack vectors / security
# ---------------------------------------------------------------------------


class TestSecurityBlocks:
    def test_import_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Invalid expression syntax"):
            evaluate_expression("import os")

    def test_dunder_class(self) -> None:
        with pytest.raises(ExpressionError, match="Dunder attribute"):
            evaluate_expression("x.__class__", {"x": 1})

    def test_dunder_subclasses(self) -> None:
        with pytest.raises(ExpressionError, match="Dunder"):
            evaluate_expression("x.__class__.__subclasses__()", {"x": 1})

    def test_dunder_globals(self) -> None:
        with pytest.raises(ExpressionError, match="Dunder attribute"):
            evaluate_expression("x.__globals__", {"x": lambda: None})

    def test_dunder_builtins(self) -> None:
        with pytest.raises(ExpressionError, match=r"Dunder|not allowed"):
            evaluate_expression('x.__builtins__["__import__"]("os")', {"x": {}})

    def test_dunder_init(self) -> None:
        with pytest.raises(ExpressionError, match="Dunder attribute"):
            evaluate_expression("x.__init__.__globals__", {"x": object()})

    def test_exec_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Function call not allowed"):
            evaluate_expression('exec("print(1)")')

    def test_eval_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Function call not allowed"):
            evaluate_expression('eval("1+1")')

    def test_compile_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Function call not allowed"):
            evaluate_expression('compile("1", "", "eval")')

    def test_open_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Function call not allowed"):
            evaluate_expression('open("/etc/passwd")')

    def test_getattr_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Function call not allowed"):
            evaluate_expression('getattr(x, "__class__")', {"x": 1})

    def test_dunder_name_access(self) -> None:
        with pytest.raises(ExpressionError, match="Dunder name"):
            evaluate_expression("__import__")

    def test_lambda_blocked(self) -> None:
        with pytest.raises(ExpressionError, match=r"Lambda|not allowed"):
            evaluate_expression("(lambda: 1)()")

    def test_comprehension_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Comprehensions"):
            evaluate_expression("[x for x in range(10)]")

    def test_walrus_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Named expression"):
            evaluate_expression("(x := 5)")

    def test_starred_blocked(self) -> None:
        with pytest.raises(ExpressionError, match="Starred"):
            evaluate_expression("[*items]", {"items": [1, 2]})

    def test_complex_call_blocked(self) -> None:
        # Subscript call like func_list[0]()
        with pytest.raises(ExpressionError, match="Complex function calls"):
            evaluate_expression("funcs[0]()", {"funcs": [len]})


# ---------------------------------------------------------------------------
# DOS / limits
# ---------------------------------------------------------------------------


class TestLimits:
    def test_max_length(self) -> None:
        long_expr = "x " + "+ x " * 200  # way over 500 chars
        with pytest.raises(ExpressionError, match="maximum length"):
            evaluate_expression(long_expr, {"x": 1})

    def test_max_depth(self) -> None:
        # Build deeply nested ternary: 1 if (1 if (1 if ... else 0) else 0) else 0
        inner = "1"
        for _ in range(MAX_AST_DEPTH + 5):
            inner = f"(1 if {inner} else 0)"
        with pytest.raises(ExpressionError, match="maximum depth"):
            evaluate_expression(inner)

    def test_empty_expression(self) -> None:
        with pytest.raises(ExpressionError, match="Empty expression"):
            evaluate_expression("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ExpressionError, match="Empty expression"):
            evaluate_expression("   ")

    def test_syntax_error(self) -> None:
        with pytest.raises(ExpressionError, match="Invalid expression syntax"):
            evaluate_expression("x +")

    def test_undefined_variable(self) -> None:
        with pytest.raises(ExpressionError, match="Undefined variable"):
            evaluate_expression("unknown_var")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_float_comparison(self) -> None:
        assert evaluate_expression("x > 1.5", {"x": 2.0}) is True

    def test_negative_number(self) -> None:
        assert evaluate_expression("-x", {"x": 5}) == -5

    def test_power_operator(self) -> None:
        assert evaluate_expression("x ** 2", {"x": 3}) == 9

    def test_floor_division(self) -> None:
        assert evaluate_expression("x // 3", {"x": 10}) == 3

    def test_boolean_short_circuit_and(self) -> None:
        # Should not error on second operand due to short-circuit
        assert evaluate_expression("False and 1/0", {}) is False

    def test_boolean_short_circuit_or(self) -> None:
        assert evaluate_expression("True or 1/0", {}) is True

    def test_nested_ternary(self) -> None:
        expr = '"a" if x > 10 else "b" if x > 5 else "c"'
        assert evaluate_expression(expr, {"x": 15}) == "a"
        assert evaluate_expression(expr, {"x": 7}) == "b"
        assert evaluate_expression(expr, {"x": 1}) == "c"

    def test_isinstance_allowed(self) -> None:
        assert evaluate_expression("isinstance(x, int)", {"x": 5}) is True

    def test_mixed_js_and_python(self) -> None:
        assert evaluate_expression("x > 0 && x !== 5", {"x": 3}) is True
