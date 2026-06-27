import pytest

from zyn.tools import Sandbox, execute, tool_schemas


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(tmp_path, allow_write=False, allow_shell=False)


@pytest.fixture
def writable(tmp_path):
    return Sandbox(tmp_path, allow_write=True, allow_shell=False)


def test_calculator(sandbox):
    assert execute("calculator", {"expr": "2 + 3 * 4"}, sandbox) == "14"
    assert execute("calculator", {"expr": "(1 + 1) ** 5"}, sandbox) == "32"
    assert execute("calculator", {"expr": "7 / 2"}, sandbox) == "3.5"


def test_calculator_rejects_code(sandbox):
    out = execute("calculator", {"expr": "__import__('os').getcwd()"}, sandbox)
    assert out.startswith("Error")


def test_read_and_list(writable):
    execute("write_file", {"path": "a.txt", "content": "hello"}, writable)
    assert execute("read_file", {"path": "a.txt"}, writable) == "hello"
    assert "a.txt" in execute("list_dir", {"path": "."}, writable)


def test_write_blocked_without_permission(sandbox):
    out = execute("write_file", {"path": "x.txt", "content": "y"}, sandbox)
    assert "Error" in out and "disabled" in out


def test_edit_file(writable):
    execute("write_file", {"path": "f.py", "content": "x = 1\n"}, writable)
    execute("edit_file", {"path": "f.py", "old": "x = 1", "new": "x = 2"}, writable)
    assert "x = 2" in execute("read_file", {"path": "f.py"}, writable)


def test_glob_and_grep(writable):
    execute("write_file", {"path": "src/main.py", "content": "def run():\n    pass\n"}, writable)
    execute("write_file", {"path": "src/util.py", "content": "x = 0\n"}, writable)
    matches = execute("glob", {"pattern": "src/*.py"}, writable)
    assert "src/main.py" in matches and "src/util.py" in matches
    hits = execute("grep", {"pattern": "def ", "path": "."}, writable)
    assert "main.py" in hits and "def run" in hits


def test_path_escape_blocked(sandbox):
    out = execute("read_file", {"path": "../../etc/passwd"}, sandbox)
    assert "Error" in out and "escapes sandbox" in out


def test_bash_blocked_without_permission(sandbox):
    out = execute("bash", {"command": "echo hi"}, sandbox)
    assert "Error" in out and "disabled" in out


def test_bash_runs_when_allowed(tmp_path):
    sb = Sandbox(tmp_path, allow_shell=True)
    assert execute("bash", {"command": "echo hi"}, sb) == "hi"


def test_unknown_tool(sandbox):
    assert "unknown tool" in execute("nope", {}, sandbox)


def test_tool_schemas_have_required_fields():
    schemas = tool_schemas()
    names = {s["name"] for s in schemas}
    assert {"calculator", "read_file", "write_file", "bash", "grep"} <= names
    for s in schemas:
        assert "description" in s
        assert s["parameters"]["type"] == "object"
