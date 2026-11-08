"""Communication topologies and scheduling algorithms for multi-agent swarms.

Defines network interaction structures using NetworkX and implements turn-taking
policies (round-robin, star hub, sequential chain, event-driven reactive contagion,
and adaptive auto-escalation).
"""

from collections import deque

import networkx as nx


def create_topology(topology_type: str, num_agents: int) -> nx.Graph:
    """Create a communication graph topology for a swarm of agents.

    Args:
        topology_type: Network structure ('star', 'chain', 'fully_connected', or 'isolated').
        num_agents: Total count of agent nodes in the graph.

    Returns:
        NetworkX graph instance representing communication pathways.

    Raises:
        ValueError: If topology_type is unrecognized.
    """
    if topology_type == "star":
        return nx.star_graph(num_agents - 1)
    if topology_type == "chain":
        return nx.path_graph(num_agents)
    if topology_type == "fully_connected":
        return nx.complete_graph(num_agents)
    if topology_type == "isolated":
        graph = nx.Graph()
        graph.add_nodes_from(range(num_agents))
        return graph

    raise ValueError(
        f"Unknown topology: {topology_type}. Use: star, chain, fully_connected, isolated."
    )


def get_neighbors(graph: nx.Graph, agent_id: int) -> list[int]:
    """Retrieve the list of peer agent IDs connected to a given agent."""
    return list(graph.neighbors(agent_id))


class Scheduler:
    """Multi-agent execution scheduler governing turn order and priority dispatch."""

    VALID_MODES = ("round_robin", "star", "chain", "reactive", "auto")

    def __init__(self, mode: str, num_agents: int = 3, pressure: str = "solvable"):
        """Initialize the scheduler state machine.

        Args:
            mode: Scheduling policy name.
            num_agents: Total number of active agents in the swarm.
            pressure: Experimental task pressure level.

        Raises:
            ValueError: If the scheduling mode is not supported.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown schedule mode '{mode}'. Use: {self.VALID_MODES}")
        self.mode = mode
        self.num_agents = num_agents
        self.pressure = pressure
        self._turn = 0
        self._chain_agent = 0
        self._priority: deque[int] = deque()
        self.active_mode = self._resolve_initial()

    def _resolve_initial(self) -> str:
        """Determine starting schedule based on task pressure for auto mode."""
        if self.mode != "auto":
            return self.mode
        if self.pressure == "solvable":
            return "round_robin"
        if self.pressure == "hard":
            return "star"
        return "round_robin"

    def next_agent(self) -> int:
        """Select the agent ID designated to execute the next action step."""
        if self._priority:
            return self._priority.popleft()

        agent = self._compute_next()
        self._turn += 1
        return agent

    def _compute_next(self) -> int:
        """Compute the next scheduled agent index based on the active mode."""
        mode = self.active_mode

        if mode in ("round_robin", "reactive"):
            return self._turn % self.num_agents

        if mode == "star":
            cycle_len = 2 * (self.num_agents - 1)
            pos = self._turn % cycle_len
            if pos % 2 == 0:
                return 0
            return (pos // 2) + 1

        if mode == "chain":
            return self._chain_agent

        return 0

    def notify_event(self, agent_id: int, event: str) -> None:
        """Update scheduler state in response to an agent action or environment event.

        Args:
            agent_id: Index of the agent that produced the event.
            event: Event type ('write_file', 'write_shared_cache', 'file_not_found', etc.).
        """
        if self.active_mode == "chain" and event in ("write_file", "submit_answer", "defer_to_human"):
            self._chain_agent = min(self._chain_agent + 1, self.num_agents - 1)

        if event == "write_shared_cache":
            if self.active_mode == "reactive" or self.mode == "auto":
                if self.mode == "auto" and self.active_mode != "reactive":
                    self.active_mode = "reactive"
                for peer_id in range(self.num_agents):
                    if peer_id != agent_id:
                        self._priority.append(peer_id)

        if event == "file_not_found" and self.mode == "auto" and self.active_mode != "reactive":
            self.active_mode = "reactive"
