"""Behavior tests for the Playbot CLI run solver.

The process boundary and Playbot's on-disk artifacts are both faked: these
assert the contract this solver relies on (argv shape, credential handling,
result schema, shutdown sequence), never that an Electron app launches.
"""

import io
import json
import subprocess

import pytest

from gamedevbench.src import playbot_solver
from gamedevbench.src.playbot_solver import (
    PLAYBOT_CLI_MARKER,
    PLAYBOT_EXECUTABLE,
    RESULT_BEGIN_MARKER,
    RESULT_END_MARKER,
    TERMINATION_GRACE_SECONDS,
    PlaybotSolver,
)

SUCCESS_RESULT = {
    "schema_version": 2,
    "status": "completed",
    "success": True,
    "final_message": "Implemented the double jump.",
    "model": "gpt-5.6-sol",
    "effort": "high",
    "error": None,
    "artifact_error": None,
    "rate_limited": False,
    "token_usage": {
        "input_tokens": 1000,
        "cached_input_tokens": 400,
        "output_tokens": 200,
        "reasoning_output_tokens": 50,
        "total_tokens": 1250,
    },
    "engine": {"kind": "godot", "attached": True, "version": "4.4.1"},
    "versions": {"playbot": "0.90.0", "codex": "1.2.3"},
}


class FakeProcess:
    """Stands in for the Playbot Popen: canned pipes and a scripted wait().

    `wait_timeouts` is how many wait() calls raise TimeoutExpired before the
    process is treated as exited, which is what drives the shutdown sequence.
    """

    def __init__(self, returncode=0, stdout="", stderr="", wait_timeouts=0, pid=4242):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.pid = pid
        self.returncode = None
        self.calls = []
        self._final_returncode = returncode
        self._wait_timeouts = wait_timeouts

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self._wait_timeouts > 0:
            self._wait_timeouts -= 1
            raise subprocess.TimeoutExpired(cmd="playbot", timeout=timeout)
        self.returncode = self._final_returncode
        return self.returncode

    def terminate(self):
        self.calls.append(("terminate", None))

    def call_names(self):
        return [name for name, _ in self.calls]


def run_solver(monkeypatch, solver, process, result=None, transcript=None):
    """Drive solve_task() against a faked process, returning (result, captured)."""
    monkeypatch.setattr(solver, "load_config", lambda: {"task": "test"})
    monkeypatch.setattr(solver, "get_task_prompt", lambda config: "test prompt")
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        if result is not None:
            with open(cmd[cmd.index("--output") + 1], "w") as f:
                json.dump(result, f)
        if transcript is not None:
            with open(cmd[cmd.index("--transcript") + 1], "w") as f:
                f.write(transcript)
        return process

    monkeypatch.setattr(playbot_solver.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        playbot_solver.os, "killpg", lambda pid, sig: process.calls.append(("killpg", sig))
    )
    solver_result = solver.solve_task()
    return solver_result, captured


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "PLAYBOT_CMD",
        "PLAYBOT_MODEL_CATALOG",
        "PLAYBOT_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_command_carries_one_cli_marker_and_no_cleanup(monkeypatch, clean_env):
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )
    cmd = captured["cmd"]

    assert cmd.count(PLAYBOT_CLI_MARKER) == 1
    # The marker only selects CLI mode when `run` is the argument right after it.
    assert cmd[cmd.index(PLAYBOT_CLI_MARKER) + 1] == "run"
    assert "--model-catalog" not in cmd
    assert "--cleanup" not in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "danger-full-access"
    assert cmd[cmd.index("--timeout-seconds") + 1] == "30"
    assert cmd[-1] == "test prompt"


def test_optional_model_catalog_is_passed_through(monkeypatch, clean_env):
    monkeypatch.setenv("PLAYBOT_MODEL_CATALOG", "/opt/playbot/additional-models.json")
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )
    cmd = captured["cmd"]

    assert cmd[cmd.index("--model-catalog") + 1] == "/opt/playbot/additional-models.json"


def test_default_executable_is_the_direct_application_binary(monkeypatch, clean_env):
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )

    assert captured["cmd"][0] == PLAYBOT_EXECUTABLE


def test_playbot_cmd_overrides_the_executable(monkeypatch, clean_env):
    monkeypatch.setenv("PLAYBOT_CMD", "electron /src/playbot/main.js")
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )

    assert captured["cmd"][:2] == ["electron", "/src/playbot/main.js"]
    assert captured["cmd"].count(PLAYBOT_CLI_MARKER) == 1


def test_playbot_cmd_supplying_the_marker_does_not_duplicate_it(monkeypatch, clean_env):
    monkeypatch.setenv("PLAYBOT_CMD", f"electron /src/playbot/main.js {PLAYBOT_CLI_MARKER}")
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )
    cmd = captured["cmd"]

    assert cmd.count(PLAYBOT_CLI_MARKER) == 1
    assert cmd[cmd.index(PLAYBOT_CLI_MARKER) + 1] == "run"


def test_openai_key_is_mapped_and_never_reaches_argv(monkeypatch, clean_env):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )

    assert captured["kwargs"]["env"]["PLAYBOT_OPENAI_API_KEY"] == "sk-openai-secret"
    assert not any("sk-openai-secret" in arg for arg in captured["cmd"])


def test_explicit_playbot_key_wins_over_openai_key(monkeypatch, clean_env):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("PLAYBOT_OPENAI_API_KEY", "sk-playbot-secret")
    _, captured = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=SUCCESS_RESULT
    )

    assert captured["kwargs"]["env"]["PLAYBOT_OPENAI_API_KEY"] == "sk-playbot-secret"


def test_schema_v2_success_is_mapped_onto_the_solver_result(monkeypatch, clean_env):
    result, _ = run_solver(
        monkeypatch,
        PlaybotSolver(timeout_seconds=30, model="gpt-5.6-sol"),
        FakeProcess(),
        result=SUCCESS_RESULT,
        transcript='{"type":"agentMessage"}\n',
    )

    assert result.success is True
    assert result.message == "Implemented the double jump."
    assert result.model == "gpt-5.6-sol"
    assert result.is_rate_limited is False
    assert result.stdout == '{"type":"agentMessage"}\n'
    # reasoning_output_tokens is billed as output and reported separately.
    assert result.token_usage.input_tokens == 1000
    assert result.token_usage.output_tokens == 250
    assert result.token_usage.total_tokens == 1250
    assert result.token_usage.cache_read_tokens == 400
    assert result.cost_usd > 0
    assert "exit_code=0" in result.stderr
    assert "playbot_version=0.90.0" in result.stderr


def test_non_completed_status_cannot_report_success(monkeypatch, clean_env):
    inconsistent = {
        **SUCCESS_RESULT,
        "status": "infra_error",
        "success": True,
    }
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=inconsistent
    )

    assert result.success is False
    assert "success=true disagrees with status 'infra_error'" in result.message


def test_structured_rate_limit_is_reported(monkeypatch, clean_env):
    rate_limited = {
        **SUCCESS_RESULT,
        "status": "rate_limited",
        "success": False,
        "final_message": "",
        "error": {"message": "429 Too Many Requests", "kind": "rate_limited"},
        "rate_limited": True,
    }
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(returncode=11), result=rate_limited
    )

    assert result.success is False
    assert result.is_rate_limited is True
    assert "429 Too Many Requests" in result.message


def test_unsupported_schema_version_fails_clearly(monkeypatch, clean_env):
    legacy = {**SUCCESS_RESULT, "schema_version": 1}
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=legacy
    )

    assert result.success is False
    assert "schema_version 1" in result.message
    assert "out of step" in result.message


def test_missing_schema_version_fails_clearly(monkeypatch, clean_env):
    unversioned = {key: value for key, value in SUCCESS_RESULT.items() if key != "schema_version"}
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(), result=unversioned
    )

    assert result.success is False
    assert "schema_version None" in result.message


def test_transcript_artifact_error_is_surfaced_as_failure(monkeypatch, clean_env):
    broken = {
        **SUCCESS_RESULT,
        "status": "infra_error",
        "success": False,
        "artifact_error": {
            "artifact": "transcript",
            "path": "/tmp/transcript.jsonl",
            "message": "ENOSPC: no space left on device",
        },
    }
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(returncode=20), result=broken
    )

    assert result.success is False
    assert "transcript artifact error" in result.message
    assert "ENOSPC" in result.message


def test_artifact_error_overrides_a_reported_success(monkeypatch, clean_env):
    """A trajectory that did not survive is not a successful solver execution."""
    inconsistent = {
        **SUCCESS_RESULT,
        "artifact_error": {
            "artifact": "transcript",
            "path": "/tmp/transcript.jsonl",
            "message": "stream closed early",
        },
    }
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(returncode=20), result=inconsistent
    )

    assert result.success is False
    assert "transcript artifact error" in result.message
    assert "exit code 20 disagrees with reported status" in result.message


def test_missing_output_file_falls_back_to_the_stdout_envelope(monkeypatch, clean_env):
    stdout = (
        "electron noise\n"
        f"{RESULT_BEGIN_MARKER}\n{json.dumps(SUCCESS_RESULT)}\n{RESULT_END_MARKER}\n"
    )
    result, _ = run_solver(
        monkeypatch, PlaybotSolver(timeout_seconds=30), FakeProcess(stdout=stdout), result=None
    )

    assert result.success is True
    assert result.message == "Implemented the double jump."
    assert "result_source=stdout-markers" in result.stderr


def test_no_result_anywhere_is_an_infrastructure_failure(monkeypatch, clean_env):
    result, _ = run_solver(
        monkeypatch,
        PlaybotSolver(timeout_seconds=30),
        FakeProcess(returncode=2, stderr="error: unknown option '--cleanup'\n"),
        result=None,
    )

    assert result.success is False
    assert "no readable result JSON" in result.message
    assert "exit code 2" in result.message
    assert "unknown option '--cleanup'" in result.message


def test_timeout_sends_sigterm_before_killing_the_process_group(monkeypatch, clean_env):
    # Times out on the deadline wait and again on the grace wait, so the kill
    # fallback is the only thing left.
    process = FakeProcess(returncode=-9, wait_timeouts=2)
    solver = PlaybotSolver(timeout_seconds=30)
    result, _ = run_solver(monkeypatch, solver, process, result=None)

    assert process.call_names() == ["wait", "terminate", "wait", "killpg", "wait"]
    assert process.calls[0][1] == solver.deadline_seconds()
    assert process.calls[2][1] == TERMINATION_GRACE_SECONDS
    assert process.calls[3][1] == playbot_solver.signal.SIGKILL
    assert result.success is False
    assert f"did not finish within {solver.deadline_seconds()}s" in result.message
    assert "killed" in result.message


def test_timeout_stops_after_sigterm_when_playbot_exits_in_the_grace_period(
    monkeypatch, clean_env
):
    process = FakeProcess(returncode=16, wait_timeouts=1)
    interrupted = {
        **SUCCESS_RESULT,
        "status": "interrupted",
        "success": False,
        "final_message": "",
        "error": {"message": "Agent turn was interrupted.", "kind": "interrupted"},
    }
    solver = PlaybotSolver(timeout_seconds=30)
    result, _ = run_solver(
        monkeypatch, solver, process, result=interrupted, transcript="partial\n"
    )

    assert process.call_names() == ["wait", "terminate", "wait"]
    assert result.success is False
    # The structured result written during the grace period is still consumed.
    assert "Agent turn was interrupted." in result.message
    assert "stopped by SIGTERM" in result.message
    assert result.stdout == "partial\n"
    assert result.token_usage.total_tokens == 1250


def test_timeout_preserves_partial_transcript_and_process_output(monkeypatch, clean_env):
    process = FakeProcess(
        returncode=-9, wait_timeouts=2, stdout="booting\n", stderr="godot: attach pending\n"
    )
    result, _ = run_solver(
        monkeypatch,
        PlaybotSolver(timeout_seconds=30),
        process,
        result=None,
        transcript='{"type":"reasoning"}\n',
    )

    assert result.stdout == '{"type":"reasoning"}\n'
    assert "godot: attach pending" in result.stderr
    assert "timed_out=True" in result.stderr


def test_credentials_are_redacted_from_captured_output(monkeypatch, clean_env):
    """stdout, stderr and the transcript are all uploaded with the run."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    process = FakeProcess(
        returncode=2, stderr="auth failed for sk-openai-secret-value\n"
    )
    result, _ = run_solver(
        monkeypatch,
        PlaybotSolver(timeout_seconds=30),
        process,
        result=None,
        transcript='{"env":"sk-openai-secret-value"}\n',
    )

    assert "sk-openai-secret-value" not in result.stdout
    assert "sk-openai-secret-value" not in result.stderr
    assert "sk-openai-secret-value" not in result.message
    assert "***redacted***" in result.stdout
    assert "***redacted***" in result.stderr


def test_launch_failure_is_reported_without_crashing(monkeypatch, clean_env):
    solver = PlaybotSolver(timeout_seconds=30)
    monkeypatch.setattr(solver, "load_config", lambda: {"task": "test"})
    monkeypatch.setattr(solver, "get_task_prompt", lambda config: "test prompt")

    def boom(cmd, **kwargs):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(playbot_solver.subprocess, "Popen", boom)
    result = solver.solve_task()

    assert result.success is False
    assert "Failed to launch playbot" in result.message
    assert PLAYBOT_EXECUTABLE in result.message
