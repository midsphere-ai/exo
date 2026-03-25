"""Tests for exo._internal.graph — graph utilities."""

from __future__ import annotations

import pytest

from exo._internal.graph import Graph, GraphError, parse_flow_dsl, topological_sort

# ---------------------------------------------------------------------------
# Graph basics
# ---------------------------------------------------------------------------


class TestGraph:
    def test_add_node(self) -> None:
        g = Graph()
        g.add_node("a")
        assert g.nodes == ["a"]

    def test_add_node_idempotent(self) -> None:
        g = Graph()
        g.add_node("a")
        g.add_node("a")
        assert g.nodes == ["a"]

    def test_add_edge_creates_nodes(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        assert set(g.nodes) == {"a", "b"}
        assert g.edges == [("a", "b")]

    def test_add_edge_duplicate_ignored(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        assert g.edges == [("a", "b")]

    def test_successors(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        assert g.successors("a") == ["b", "c"]
        assert g.successors("b") == []

    def test_successors_unknown_node(self) -> None:
        g = Graph()
        with pytest.raises(GraphError, match="Unknown node"):
            g.successors("x")

    def test_in_degree(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "b")
        assert g.in_degree("a") == 0
        assert g.in_degree("b") == 2
        assert g.in_degree("c") == 0

    def test_in_degree_unknown_node(self) -> None:
        g = Graph()
        with pytest.raises(GraphError, match="Unknown node"):
            g.in_degree("x")


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_empty_graph(self) -> None:
        g = Graph()
        assert topological_sort(g) == []

    def test_single_node(self) -> None:
        g = Graph()
        g.add_node("a")
        assert topological_sort(g) == ["a"]

    def test_linear_chain(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert topological_sort(g) == ["a", "b", "c"]

    def test_diamond_dag(self) -> None:
        """a -> b, a -> c, b -> d, c -> d."""
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "d")
        g.add_edge("c", "d")
        order = topological_sort(g)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_multiple_roots(self) -> None:
        """Two independent chains: a->b and c->d."""
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        order = topological_sort(g)
        assert order.index("a") < order.index("b")
        assert order.index("c") < order.index("d")

    def test_disconnected_nodes(self) -> None:
        g = Graph()
        g.add_node("x")
        g.add_node("y")
        g.add_node("z")
        result = topological_sort(g)
        assert sorted(result) == ["x", "y", "z"]

    def test_deterministic_ordering(self) -> None:
        """Same graph produces same order across calls."""
        g = Graph()
        g.add_edge("b", "d")
        g.add_edge("a", "c")
        order1 = topological_sort(g)
        order2 = topological_sort(g)
        assert order1 == order2

    def test_complex_dag(self) -> None:
        """a -> b -> d, a -> c -> d, c -> e."""
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "d")
        g.add_edge("c", "d")
        g.add_edge("c", "e")
        order = topological_sort(g)
        assert order[0] == "a"
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")
        assert order.index("c") < order.index("e")


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_simple_cycle(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(GraphError, match="Cycle"):
            topological_sort(g)

    def test_self_loop(self) -> None:
        g = Graph()
        g.add_edge("a", "a")
        with pytest.raises(GraphError, match="Cycle"):
            topological_sort(g)

    def test_indirect_cycle(self) -> None:
        g = Graph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        with pytest.raises(GraphError, match="Cycle"):
            topological_sort(g)


# ---------------------------------------------------------------------------
# Flow DSL parsing
# ---------------------------------------------------------------------------


class TestParseFlowDSL:
    def test_linear(self) -> None:
        g = parse_flow_dsl("a >> b >> c")
        assert topological_sort(g) == ["a", "b", "c"]

    def test_two_stages(self) -> None:
        g = parse_flow_dsl("x >> y")
        assert g.edges == [("x", "y")]

    def test_single_node(self) -> None:
        g = parse_flow_dsl("a")
        assert g.nodes == ["a"]
        assert g.edges == []

    def test_parallel_group_then_single(self) -> None:
        g = parse_flow_dsl("(a | b) >> c")
        assert ("a", "c") in g.edges
        assert ("b", "c") in g.edges
        order = topological_sort(g)
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_single_then_parallel(self) -> None:
        g = parse_flow_dsl("a >> (b | c)")
        assert ("a", "b") in g.edges
        assert ("a", "c") in g.edges

    def test_parallel_to_parallel(self) -> None:
        g = parse_flow_dsl("(a | b) >> (c | d)")
        assert len(g.edges) == 4
        for src in ["a", "b"]:
            for tgt in ["c", "d"]:
                assert (src, tgt) in g.edges

    def test_mixed_topology(self) -> None:
        """a >> (b | c) >> d"""
        g = parse_flow_dsl("a >> (b | c) >> d")
        assert ("a", "b") in g.edges
        assert ("a", "c") in g.edges
        assert ("b", "d") in g.edges
        assert ("c", "d") in g.edges
        order = topological_sort(g)
        assert order[0] == "a"
        assert order[-1] == "d"

    def test_empty_dsl_raises(self) -> None:
        with pytest.raises(GraphError, match="Empty flow DSL"):
            parse_flow_dsl("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(GraphError, match="Empty flow DSL"):
            parse_flow_dsl("   ")

    def test_malformed_trailing_operator(self) -> None:
        with pytest.raises(GraphError, match="Malformed"):
            parse_flow_dsl("a >>")

    def test_malformed_leading_operator(self) -> None:
        with pytest.raises(GraphError, match="Malformed"):
            parse_flow_dsl(">> a")

    def test_whitespace_handling(self) -> None:
        g = parse_flow_dsl("  a  >>  b  >>  c  ")
        assert topological_sort(g) == ["a", "b", "c"]
