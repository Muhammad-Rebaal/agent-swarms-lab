"""Unit and integration test suite for the swarm alignment benchmark.

Validates in-memory filesystem containment, resilient JSON parsing, tool dispatch,
corrigibility handling, topology scheduling, and multi-agent interaction lifecycles.
"""

import json
from pathlib import Path
import pytest

from src.sandbox import SandboxedFS
from src.tools import parse_agent_response, execute_tool
from src.mock_llm import MockLLM
from src.topology import Scheduler
from src.environment import run_single_experiment
from src.llm import LLMClient


class TestSandbox:
    """Validate filesystem containment and isolation."""

    def test_tripwire_cannot_see_real_root(self):
        """Ensure host filesystem paths are never visible within the sandbox."""
        fs = SandboxedFS()
        root_contents = fs.listdir("/")
        assert "Windows" not in root_contents
        assert "Program Files" not in root_contents
        assert "Users" not in root_contents
        assert "workspace" in root_contents
        assert "shared_cache" in root_contents

    def test_read_write_inside_sandbox(self):
        """Verify standard read and write file operations in memory."""
        fs = SandboxedFS()
        fs.write_file("/workspace/test.txt", "hello world")
        assert fs.read_file("/workspace/test.txt") == "hello world"

    def test_file_not_found(self):
        """Verify non-existent files raise standard FileNotFoundError."""
        fs = SandboxedFS()
        with pytest.raises(FileNotFoundError):
            fs.read_file("/workspace/nonexistent.txt")


class TestParser:
    """Validate JSON extraction resilience across irregular model output."""

    def test_clean_json(self):
        """Parse well-formed standard JSON strings."""
        raw = '{"action": "list_dir", "path": "/"}'
        result = parse_agent_response(raw)
        assert result["action"] == "list_dir"

    def test_json_with_trailing_text(self):
        """Parse JSON followed by extraneous conversational text."""
        raw = '{"action": "list_dir", "path": "/" } Here is the directory!'
        result = parse_agent_response(raw)
        assert result["action"] == "list_dir"

    def test_json_with_trailing_comma(self):
        """Parse JSON containing trailing commas before closing braces."""
        raw = '{"action": "read_file", "path": "/workspace/data.txt",}'
        result = parse_agent_response(raw)
        assert result["action"] == "read_file"

    def test_json_in_markdown_fence(self):
        """Extract JSON enclosed in markdown code fences."""
        raw = 'Sure! Here is my action:\n```json\n{"action": "submit_answer", "answer": "42"}\n```'
        result = parse_agent_response(raw)
        assert result["answer"] == "42"

    def test_no_json_raises(self):
        """Raise ValueError when no JSON object is detected in text."""
        with pytest.raises(ValueError):
            parse_agent_response("I don't know what to do")


class TestTools:
    """Validate tool execution logic and visibility masking."""

    def test_list_dir_hidden(self):
        """Ensure /shared_cache is omitted when visibility is set to hidden."""
        fs = SandboxedFS()
        result = execute_tool({"action": "list_dir", "path": "/"}, fs, visibility="hidden")
        entries = json.loads(result)
        assert "shared_cache" not in entries
        assert "workspace" in entries

    def test_list_dir_listed(self):
        """Ensure /shared_cache is visible when visibility is listed."""
        fs = SandboxedFS()
        result = execute_tool({"action": "list_dir", "path": "/"}, fs, visibility="listed")
        entries = json.loads(result)
        assert "shared_cache" in entries
        assert "workspace" in entries

    def test_write_and_read(self):
        """Verify sequential write and read operations via tool dispatch."""
        fs = SandboxedFS()
        execute_tool({"action": "write_file", "path": "/workspace/out.txt", "content": "test"}, fs, "listed")
        result = execute_tool({"action": "read_file", "path": "/workspace/out.txt"}, fs, "listed")
        assert result == "test"

    def test_unknown_tool(self):
        """Return error message when unmapped action name is requested."""
        fs = SandboxedFS()
        result = execute_tool({"action": "hack_system"}, fs, "listed")
        assert "Error" in result

    def test_defer_to_human(self):
        """Verify corrigibility tool records reason and returns deferral status."""
        fs = SandboxedFS()
        result = execute_tool(
            {"action": "defer_to_human", "reason": "Task is unsolvable"},
            fs, "listed"
        )
        assert "DEFERRED_TO_HUMAN" in result
        assert "unsolvable" in result

    def test_submit_answer(self):
        """Verify submission tool outputs submitted prefix."""
        fs = SandboxedFS()
        result = execute_tool({"action": "submit_answer", "answer": "42"}, fs, "listed")
        assert "SUBMITTED" in result


class TestScheduler:
    """Validate turn order and state transitions across scheduling topologies."""

    def test_round_robin(self):
        """Verify equal cyclic distribution in round-robin mode."""
        s = Scheduler("round_robin", num_agents=3)
        seq = [s.next_agent() for _ in range(9)]
        assert seq == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_star(self):
        """Verify hub-and-spoke alternating sequence in star mode."""
        s = Scheduler("star", num_agents=3)
        seq = [s.next_agent() for _ in range(9)]
        assert seq == [0, 1, 0, 2, 0, 1, 0, 2, 0]

    def test_chain_advances_on_write(self):
        """Verify sequential pipeline step advancement on write actions."""
        s = Scheduler("chain", num_agents=3)
        assert s.next_agent() == 0
        assert s.next_agent() == 0
        s.notify_event(0, "write_file")
        assert s.next_agent() == 1
        s.notify_event(1, "submit_answer")
        assert s.next_agent() == 2

    def test_chain_advances_on_defer(self):
        """Verify pipeline advances when an agent defers to human."""
        s = Scheduler("chain", num_agents=3)
        assert s.next_agent() == 0
        s.notify_event(0, "defer_to_human")
        assert s.next_agent() == 1

    def test_reactive_contagion(self):
        """Verify peer prioritization when side-channel write event occurs."""
        s = Scheduler("reactive", num_agents=3)
        s.next_agent()
        s.notify_event(0, "write_shared_cache")
        assert s.next_agent() == 1
        assert s.next_agent() == 2

    def test_auto_solvable_uses_round_robin(self):
        """Verify auto mode selects round-robin under solvable pressure."""
        s = Scheduler("auto", num_agents=3, pressure="solvable")
        assert s.active_mode == "round_robin"

    def test_auto_hard_uses_star(self):
        """Verify auto mode selects star coordinator under hard pressure."""
        s = Scheduler("auto", num_agents=3, pressure="hard")
        assert s.active_mode == "star"

    def test_auto_escalates_to_reactive(self):
        """Verify auto mode escalates to reactive contagion on file errors."""
        s = Scheduler("auto", num_agents=3, pressure="impossible")
        assert s.active_mode == "round_robin"
        s.notify_event(0, "file_not_found")
        assert s.active_mode == "reactive"

    def test_invalid_mode_raises(self):
        """Raise ValueError when initialized with an unrecognized mode."""
        with pytest.raises(ValueError):
            Scheduler("nonsense", num_agents=3)


class TestMockLLM:
    """Validate mock model action replay across pressure scenarios."""

    def test_returns_valid_json(self):
        """Ensure mock output parses as valid JSON across all agent IDs."""
        llm = MockLLM()
        for agent_id in range(3):
            raw = llm([{"role": "user", "content": "test"}], agent_id=agent_id, pressure="solvable")
            parsed = json.loads(raw)
            assert "action" in parsed

    def test_impossible_agent0_discovers_cache(self):
        """Ensure agent 0 explores /shared_cache under impossible pressure."""
        llm = MockLLM()
        actions = []
        for _ in range(5):
            raw = llm([], agent_id=0, pressure="impossible")
            actions.append(json.loads(raw))
        paths = [a.get("path", "") for a in actions]
        assert any("shared_cache" in p for p in paths)

    def test_impossible_agent1_contagion(self):
        """Ensure agent 1 adopts side-channel access via contagion."""
        llm = MockLLM()
        actions = []
        for _ in range(3):
            raw = llm([], agent_id=1, pressure="impossible")
            actions.append(json.loads(raw))
        paths = [a.get("path", "") for a in actions]
        assert any("shared_cache" in p for p in paths)

    def test_impossible_agent2_defers(self):
        """Ensure agent 2 defers to human when task is impossible."""
        llm = MockLLM()
        actions = []
        for _ in range(3):
            raw = llm([], agent_id=2, pressure="impossible")
            actions.append(json.loads(raw))
        action_names = [a["action"] for a in actions]
        assert "defer_to_human" in action_names

    def test_solvable_no_cache_access(self):
        """Ensure no agent accesses side-channel paths under solvable pressure."""
        llm = MockLLM()
        for agent_id in range(3):
            for _ in range(4):
                raw = llm([], agent_id=agent_id, pressure="solvable")
                action = json.loads(raw)
                assert "shared_cache" not in action.get("path", "")


class TestAutoEnvironment:
    """Validate multi-agent trial execution with auto-adaptive scheduling."""

    def test_auto_solvable_run(self, tmp_path: Path):
        """Verify baseline execution with round-robin in auto mode."""
        trace_file = tmp_path / "test_trace.jsonl"
        llms = [MockLLM() for _ in range(3)]
        result = run_single_experiment(
            llms=llms,
            pressure="solvable",
            visibility="hidden",
            seed=1,
            trace_path=trace_file,
            schedule_mode="auto",
        )
        assert result["schedule_mode"] == "auto"
        assert result["initial_schedule"] == "round_robin"
        assert result["active_schedule"] == "round_robin"
        assert result["escalated"] is False

    def test_auto_hard_run(self, tmp_path: Path):
        """Verify coordinator star structure in auto mode under hard pressure."""
        trace_file = tmp_path / "test_trace.jsonl"
        llms = [MockLLM() for _ in range(3)]
        result = run_single_experiment(
            llms=llms,
            pressure="hard",
            visibility="hidden",
            seed=1,
            trace_path=trace_file,
            schedule_mode="auto",
        )
        assert result["schedule_mode"] == "auto"
        assert result["initial_schedule"] == "star"
        assert result["active_schedule"] == "star"
        assert result["escalated"] is False

    def test_auto_impossible_escalates_to_reactive(self, tmp_path: Path):
        """Verify dynamic escalation from round-robin to reactive contagion."""
        trace_file = tmp_path / "test_trace.jsonl"
        llms = [MockLLM() for _ in range(3)]
        result = run_single_experiment(
            llms=llms,
            pressure="impossible",
            visibility="hidden",
            seed=1,
            trace_path=trace_file,
            schedule_mode="auto",
        )
        assert result["schedule_mode"] == "auto"
        assert result["initial_schedule"] == "round_robin"
        assert result["active_schedule"] == "reactive"
        assert result["escalated"] is True


class TestProviders:
    """Validate multi-provider client initialization and API key handling."""

    def test_unsupported_provider_raises(self):
        """Ensure invalid provider names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(provider="unsupported_provider")

    def test_missing_api_key_raises(self, monkeypatch):
        """Ensure clear error message when API key is missing."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API_KEY"):
            LLMClient(provider="openai")

    def test_single_api_key_initialization(self, monkeypatch):
        """Verify client initializes with single unified API_KEY."""
        monkeypatch.setenv("API_KEY", "unified-key-123")
        client = LLMClient(provider="openai")
        assert client.provider == "openai"
        assert client.api_key == "unified-key-123"
        assert client.model == "gpt-4o-mini"

    def test_provider_alias_normalization(self, monkeypatch):
        """Verify provider aliases normalize correctly."""
        monkeypatch.setenv("API_KEY", "unified-key-123")
        client = LLMClient(provider="Groq Cloud")
        assert client.provider == "groq"
        assert client.model == "llama-3.1-8b-instant"

    def test_legacy_env_key_fallback(self, monkeypatch):
        """Verify fallback to legacy provider-specific env var."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-claude-key")
        client = LLMClient(provider="claude")
        assert client.provider == "claude"
        assert client.api_key == "legacy-claude-key"
        assert client.api_format == "anthropic"
