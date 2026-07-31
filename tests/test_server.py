"""Tests for the MCP server entry point (get_status, run_server)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from torchtalk import indexer
from torchtalk.server import get_status, run_server


@pytest.fixture
def server_state(mock_state):
    s = mock_state
    s.bindings = [{"python_name": "add", "dispatch_key": "CPU"}]
    s.native_functions = {"add": {"base_name": "add"}}
    s.pytorch_source = "/fake/pytorch"
    s.cpp_extractor = None
    s.cpp_building = False
    s.py_modules = {}
    s.py_classes = {}
    s.nn_modules = []
    s.test_files = {}
    s.test_classes = {}
    s.test_functions = {}
    s.opinfo_registry = {}
    s.derivatives = {}
    s.registrations = {}
    indexer._build_indexes(s)
    return s


class TestGetStatus:
    def test_shows_pytorch_source(self, server_state):
        out = asyncio.run(get_status())
        assert "/fake/pytorch" in out

    def test_bindings_loaded(self, server_state):
        out = asyncio.run(get_status())
        assert "1 loaded" in out

    def test_bindings_not_loaded(self, server_state):
        server_state.bindings = []
        indexer._build_indexes(server_state)
        out = asyncio.run(get_status())
        assert "Not loaded" in out

    def test_cpp_building(self, server_state):
        server_state.cpp_building = True
        out = asyncio.run(get_status())
        assert "Building" in out

    def test_cpp_ready(self, server_state):
        ext = MagicMock()
        ext.get_call_graph_data.return_value = {
            "stats": {"total_functions": 51000, "total_call_edges": 51000}
        }
        server_state.cpp_extractor = ext
        out = asyncio.run(get_status())
        assert "Ready" in out
        assert "Functions: 51,000" in out

    def test_cpp_unavailable_no_source(self, server_state):
        server_state.pytorch_source = None
        out = asyncio.run(get_status())
        assert "Not available" in out

    def test_python_modules_loaded(self, server_state):
        server_state.py_modules = {"torch.nn.linear": MagicMock()}
        server_state.py_classes = {"Linear": [MagicMock()]}
        server_state.nn_modules = [MagicMock()]
        out = asyncio.run(get_status())
        assert "Modules: 1" in out

    def test_test_infra_loaded(self, server_state):
        server_state.test_files = {"test/test_x.py": {}}
        server_state.test_classes = {"TestX": [{}]}
        server_state.test_functions = {"test_foo": [{}]}
        server_state.opinfo_registry = {"softmax": {}}
        out = asyncio.run(get_status())
        assert "Test files: 1" in out

    def test_tools_table_ready_states(self, server_state):
        out = asyncio.run(get_status())
        assert "Ready" in out
        assert "Not ready" in out


class TestRunServer:
    def test_starts_daemon_thread(self, monkeypatch):
        captured = {}

        class FakeThread:
            def __init__(self, target, daemon):
                captured["target"] = target
                captured["daemon"] = daemon

            def start(self):
                pass

        monkeypatch.setattr("torchtalk.server.mcp.run", lambda **kw: None)
        monkeypatch.setattr("threading.Thread", FakeThread)

        run_server(pytorch_source="/fake", transport="stdio")

        assert captured["daemon"] is True
