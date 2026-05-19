"""Tests for paper_distiller.agents.dag — topology validation + topo_levels."""
import pytest

from paper_distiller.agents.dag import DAG, DAGError


class _FakeAgent:
    def __init__(self, name: str, deps: list[str] | None = None):
        self.name = name
        self.deps = deps or []

    async def run(self, ctx):
        return {}


def test_dag_constructs_with_valid_agents():
    a = _FakeAgent("a")
    b = _FakeAgent("b", deps=["a"])
    dag = DAG([a, b])
    assert set(dag.agents.keys()) == {"a", "b"}


def test_dag_rejects_duplicate_names():
    a1 = _FakeAgent("a")
    a2 = _FakeAgent("a")
    with pytest.raises(DAGError, match="duplicate"):
        DAG([a1, a2])


def test_dag_rejects_missing_dep():
    a = _FakeAgent("a", deps=["nonexistent"])
    with pytest.raises(DAGError, match="missing dependency"):
        DAG([a])


def test_dag_rejects_cycle():
    a = _FakeAgent("a", deps=["b"])
    b = _FakeAgent("b", deps=["a"])
    with pytest.raises(DAGError, match="cycle"):
        DAG([a, b])


def test_topo_levels_groups_parallel_agents():
    """Agents with no deps OR all deps in earlier levels go in the same level."""
    a = _FakeAgent("a")          # level 0
    b = _FakeAgent("b")          # level 0
    c = _FakeAgent("c", ["a", "b"])  # level 1
    d = _FakeAgent("d", ["c"])   # level 2
    dag = DAG([a, b, c, d])
    levels = dag.topo_levels()
    assert set(levels[0]) == {"a", "b"}
    assert levels[1] == ["c"]
    assert levels[2] == ["d"]


def test_topo_levels_linear_chain():
    a = _FakeAgent("a")
    b = _FakeAgent("b", ["a"])
    c = _FakeAgent("c", ["b"])
    dag = DAG([a, b, c])
    levels = dag.topo_levels()
    assert levels == [["a"], ["b"], ["c"]]
