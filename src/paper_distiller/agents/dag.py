"""Static DAG of agents — topology validation + topological level grouping."""

from __future__ import annotations

from .base import Agent


class DAGError(ValueError):
    pass


class DAG:
    """Topology of agents. Validates at construction time.

    Provides topo_levels() which returns a list of lists — each inner list
    is a set of agent names that can run concurrently (no deps on each other).
    """

    def __init__(self, agents: list[Agent]):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            seen = set()
            dupes = [n for n in names if n in seen or seen.add(n)]
            raise DAGError(f"duplicate agent names: {dupes}")
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self._validate_deps()

    def _validate_deps(self) -> None:
        for a in self.agents.values():
            for dep in a.deps:
                if dep not in self.agents:
                    raise DAGError(
                        f"agent {a.name!r} has missing dependency {dep!r}"
                    )
        # cycle detection via DFS
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(name: str, path: list[str]) -> None:
            if name in visiting:
                cycle = path[path.index(name):] + [name]
                raise DAGError(f"cycle detected: {' -> '.join(cycle)}")
            if name in visited:
                return
            visiting.add(name)
            for dep in self.agents[name].deps:
                dfs(dep, path + [name])
            visiting.discard(name)
            visited.add(name)

        for name in self.agents:
            dfs(name, [])

    def topo_levels(self) -> list[list[str]]:
        """Group agents into parallel-executable levels.

        Level k contains all agents whose deps are all in levels < k.
        Returns a list of level groups, in execution order.
        """
        levels: list[list[str]] = []
        placed: set[str] = set()
        remaining = set(self.agents.keys())

        while remaining:
            this_level = [
                name for name in remaining
                if all(dep in placed for dep in self.agents[name].deps)
            ]
            if not this_level:
                raise DAGError("topo_levels: stuck (should never happen if validation passed)")
            this_level.sort()  # stable ordering
            levels.append(this_level)
            placed.update(this_level)
            remaining -= set(this_level)
        return levels
