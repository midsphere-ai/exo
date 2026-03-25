# exo._internal.graph

Graph utilities for agent execution ordering. Provides a simple directed graph, topological sort (Kahn's algorithm), cycle detection, and a flow DSL parser for defining agent pipelines.

> **Internal API** -- subject to change without notice.

**Module:** `exo._internal.graph`

```python
from exo._internal.graph import Graph, GraphError, parse_flow_dsl, topological_sort
```

---

## GraphError

```python
class GraphError(Exception)
```

Raised for graph-related errors (cycles, invalid DSL, unknown nodes, etc.). Note: this inherits from `Exception`, not `ExoError`.

---

## Graph

```python
@dataclass
class Graph
```

Simple directed graph using adjacency lists. Nodes are strings (typically agent names). Edges are directed from source to target.

### Methods

#### add_node()

```python
def add_node(self, name: str) -> None
```

Add a node (idempotent). If the node already exists, this is a no-op.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | The node name. |

#### add_edge()

```python
def add_edge(self, source: str, target: str) -> None
```

Add a directed edge from `source` to `target`. Both nodes are created implicitly if they don't exist. Duplicate edges are ignored.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `str` | *(required)* | Source node name. |
| `target` | `str` | *(required)* | Target node name. |

#### successors()

```python
def successors(self, name: str) -> list[str]
```

Return direct successors of a node.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | The node to query. |

**Returns:** List of successor node names.

**Raises:** `GraphError` if the node is unknown.

#### in_degree()

```python
def in_degree(self, name: str) -> int
```

Return the in-degree of a node (number of edges pointing to it).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | The node to query. |

**Returns:** The in-degree count.

**Raises:** `GraphError` if the node is unknown.

### Properties

#### nodes

```python
@property
def nodes(self) -> list[str]
```

Return all node names in insertion order.

#### edges

```python
@property
def edges(self) -> list[tuple[str, str]]
```

Return all edges as `(source, target)` tuples.

### Example

```python
from exo._internal.graph import Graph

g = Graph()
g.add_edge("a", "b")
g.add_edge("b", "c")
g.add_edge("a", "c")

print(g.nodes)       # ["a", "b", "c"]
print(g.edges)       # [("a", "b"), ("a", "c"), ("b", "c")]
print(g.successors("a"))  # ["b", "c"]
print(g.in_degree("c"))   # 2
```

---

## topological_sort()

```python
def topological_sort(graph: Graph) -> list[str]
```

Topological sort using Kahn's algorithm. Produces a deterministic ordering by sorting nodes with equal in-degree alphabetically at each step.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `graph` | `Graph` | *(required)* | The directed graph to sort. |

### Returns

Ordered list of node names. Nodes with no dependencies come first.

### Raises

`GraphError` -- if the graph contains a cycle.

### Example

```python
from exo._internal.graph import Graph, topological_sort

g = Graph()
g.add_edge("a", "b")
g.add_edge("b", "c")

order = topological_sort(g)
# ["a", "b", "c"]

# With parallel branches
g2 = Graph()
g2.add_edge("a", "c")
g2.add_edge("b", "c")
g2.add_node("a")
g2.add_node("b")

order = topological_sort(g2)
# ["a", "b", "c"]  (a before b because of alphabetical tie-breaking)
```

---

## parse_flow_dsl()

```python
def parse_flow_dsl(dsl: str) -> Graph
```

Parse a flow DSL string into a `Graph`.

### Syntax

```
"a >> b >> c"           # linear chain
"(a | b) >> c"          # parallel group then c
"a >> (b | c) >> d"     # a feeds into parallel b,c which feed into d
```

- `>>` denotes sequential dependency.
- `(x | y)` denotes a parallel group -- all members share the same predecessors and successors.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dsl` | `str` | *(required)* | The flow DSL string. |

### Returns

A `Graph` with appropriate nodes and edges.

### Raises

- `GraphError` -- if the DSL string is empty.
- `GraphError` -- if the DSL string is malformed (empty stages, empty parallel group members).

### Examples

```python
from exo._internal.graph import parse_flow_dsl, topological_sort

# Linear chain
g = parse_flow_dsl("a >> b >> c")
print(g.nodes)  # ["a", "b", "c"]
print(g.edges)  # [("a", "b"), ("b", "c")]

# Parallel group
g = parse_flow_dsl("(a | b) >> c")
print(g.nodes)  # ["a", "b", "c"]
print(g.edges)  # [("a", "c"), ("b", "c")]
order = topological_sort(g)
# ["a", "b", "c"]

# Mixed
g = parse_flow_dsl("a >> (b | c) >> d")
print(g.edges)
# [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
order = topological_sort(g)
# ["a", "b", "c", "d"]
```

### DSL Parsing Details

The parser works by:

1. Splitting the DSL string on `>>`
2. For each segment, checking if it matches the parallel group pattern `(x | y | ...)`
3. Creating nodes for all segment members
4. Adding directed edges from every node in stage N to every node in stage N+1
