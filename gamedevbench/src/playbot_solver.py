#!/usr/bin/env python3
"""
Playbot solver implementation using the Playbot CLI run mode.

Playbot is an Electron IDE whose agent runs against the local Godot project
through an engine addon (execute_engine_code over WebSocket). This solver
drives one non-interactive CLI run:

    <playbot executable> --playbot-cli run -C <cwd> [flags] "<prompt>"

The run writes a machine-readable result JSON to --output and an agent
trajectory JSONL to --transcript, and always prints the same result envelope
between marker lines on stdout.

The installed `playbot` shell command is a wrapper that inserts
`--playbot-cli` itself, so this solver invokes the application binary
directly and supplies the marker exactly once. PLAYBOT_CMD overrides the
invocation for development and must likewise name a direct application
invocation (e.g. "electron /path/to/main.js"), never the shell wrapper.

The API key is taken from PLAYBOT_OPENAI_API_KEY, falling back to
OPENAI_API_KEY, and is passed only through the environment.

The working directory is GameDevBench's disposable sandbox copy. Playbot
installs its engine integration there and does not roll it back; the harness
deletes the sandbox after validation, so the solver does not undo it either.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from gamedevbench.src.base_solver import BaseSolver
from gamedevbench.src.utils.data_types import SolverResult, TokenUsage

# The packaged Linux app inside the benchmark image. This is the raw Electron
# binary, not the `playbot` shell wrapper a desktop install puts on PATH.
PLAYBOT_EXECUTABLE = "/opt/playbot/Playbot"
PLAYBOT_CLI_MARKER = "--playbot-cli"

# Result envelope contract (docs/cli-run.md in playbot-ide). Schema 2 renamed
# engine.godot_version to engine.version and added artifact_error.
SUPPORTED_RESULT_SCHEMA_VERSION = 2
RESULT_BEGIN_MARKER = "PLAYBOT_RESULT_JSON_BEGIN"
RESULT_END_MARKER = "PLAYBOT_RESULT_JSON_END"

DEFAULT_TIMEOUT_SECONDS = 600

# Wall-clock allowance on top of the agent-turn budget we hand Playbot via
# --timeout-seconds. It covers engine attach (Godot's budget is 180s), addon
# install, and teardown. Kept separate from the turn timeout on purpose: this
# is the harness's patience, not the agent's.
ENGINE_ATTACH_MARGIN_SECONDS = 300

# SIGTERM makes Playbot cancel the turn, stop the Godot editor it started, and
# still write its result and transcript. This is how long we let that finish
# before killing the process group outright.
TERMINATION_GRACE_SECONDS = 15

# Cap on waiting for the output pipes to close after the process is gone; a
# leaked grandchild holding the pipe must not stall the harness.
OUTPUT_DRAIN_TIMEOUT_SECONDS = 30

MAX_CAPTURED_OUTPUT = 2_000_000
MAX_MESSAGE_STDERR = 2_000


def _redact_secrets(text: str) -> str:
    """Strip provider credentials out of anything that becomes an artifact.

    The solver never writes the key anywhere, but stdout, stderr and the
    transcript all come from a process that holds it in its environment and an
    agent that can read that environment, and all three are uploaded with the
    run.
    """
    if not text:
        return text
    for name in ("PLAYBOT_OPENAI_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name)
        # Short values are not credentials and would corrupt ordinary output.
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***redacted***")
    return text


@dataclass
class _PlaybotRun:
    """Outcome of the Playbot process itself, before the result is interpreted."""

    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    deadline_seconds: int
    terminated_gracefully: bool = False

    def shutdown_note(self) -> str:
        return "stopped by SIGTERM" if self.terminated_gracefully else "killed"


class _PipeDrainer(threading.Thread):
    """Drains a pipe on its own thread into a bounded buffer.

    Reading both pipes only after the process exits would deadlock as soon as
    either fills, and an unbounded buffer would let a chatty Electron build
    exhaust the container's memory.
    """

    def __init__(self, stream, limit: int = MAX_CAPTURED_OUTPUT):
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._lock = threading.Lock()
        self._chunks: deque = deque()
        self._size = 0

    def run(self) -> None:
        try:
            for line in self._stream:
                with self._lock:
                    self._chunks.append(line)
                    self._size += len(line)
                    while self._size > self._limit and len(self._chunks) > 1:
                        self._size -= len(self._chunks.popleft())
        except (OSError, ValueError):
            pass
        finally:
            try:
                self._stream.close()
            except OSError:
                pass

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)[-self._limit:]


class PlaybotSolver(BaseSolver):
    """Solver that runs tasks through one Playbot CLI run."""

    SUPPORTS_MCP = False
    SUPPORTS_SYSTEM_PROMPT = False
    SUPPORTS_EFFORT = True

    DEFAULT_MODEL = "gpt-5.6-sol"

    def __init__(
        self,
        timeout_seconds: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
        debug: bool = False,
        use_mcp: bool = False,
        use_runtime_video: bool = False,
        model: Optional[str] = None,
        effort: Optional[str] = None,
    ):
        super().__init__(timeout_seconds, debug, use_mcp, use_runtime_video)
        self.model = model or self.DEFAULT_MODEL
        self.effort = effort

    @staticmethod
    def is_rate_limit_error(error_message: str) -> bool:
        """Check if the error message indicates API rate limit."""
        error_lower = error_message.lower()
        rate_limit_keywords = [
            "rate limit", "rate_limit", "ratelimit",
            "quota exceeded", "429", "too many requests",
            "usage limit", "usagelimitexceeded", "credits depleted",
        ]
        return any(keyword in error_lower for keyword in rate_limit_keywords)

    def solve_task(self) -> SolverResult:
        """Solve the task with a single Playbot CLI run."""
        config = self.load_config()
        if not config:
            return SolverResult(
                success=False,
                message="Could not load task configuration",
                duration_seconds=0.0,
            )

        start_time = time.time()
        prompt = self.get_task_prompt(config)

        if self.debug:
            print("=" * 60)
            print("SENDING PROMPT TO PLAYBOT:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)

        run_dir = tempfile.mkdtemp(prefix="playbot_run_")
        output_path = os.path.join(run_dir, "result.json")
        transcript_path = os.path.join(run_dir, "transcript.jsonl")
        cmd = self.build_command(prompt, run_dir, output_path, transcript_path)

        try:
            try:
                run = self._run_playbot(cmd)
            except OSError as e:
                return SolverResult(
                    success=False,
                    message=f"Failed to launch playbot ({cmd[0]}): {e}",
                    duration_seconds=time.time() - start_time,
                    model=self.model,
                )
            return self._interpret_run(
                run, output_path, transcript_path, time.time() - start_time
            )
        finally:
            # The profile directories under run_dir are per-task by design;
            # Playbot state is never carried between benchmark tasks.
            shutil.rmtree(run_dir, ignore_errors=True)

    def build_command(
        self, prompt: str, run_dir: str, output_path: str, transcript_path: str
    ) -> list:
        """Build the CLI run argv, with exactly one --playbot-cli marker.

        The marker must be immediately followed by `run`, so a PLAYBOT_CMD that
        already supplies it has to end with it.
        """
        cmd = shlex.split(os.environ.get("PLAYBOT_CMD") or PLAYBOT_EXECUTABLE)
        if PLAYBOT_CLI_MARKER not in cmd:
            cmd.append(PLAYBOT_CLI_MARKER)
        cmd.append("run")
        model_catalog = os.environ.get("PLAYBOT_MODEL_CATALOG")
        if model_catalog:
            cmd += ["--model-catalog", model_catalog]
        cmd += [
            "-C", os.getcwd(),
            "--model", self.model,
            "--sandbox", "danger-full-access",
            "--timeout-seconds", str(self.agent_timeout_seconds()),
            "--output", output_path,
            "--transcript", transcript_path,
            # Unique per task so concurrent Batch jobs never share a profile.
            "--user-data-path", os.path.join(run_dir, "profile", "user-data"),
            "--playbot-data-dir", os.path.join(run_dir, "profile", "data"),
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        cmd.append(prompt)
        return cmd

    def agent_timeout_seconds(self) -> int:
        return self.timeout_seconds or DEFAULT_TIMEOUT_SECONDS

    def deadline_seconds(self) -> int:
        return self.agent_timeout_seconds() + ENGINE_ATTACH_MARGIN_SECONDS

    @staticmethod
    def build_env() -> dict:
        """Provider credential goes through the environment and nowhere else.

        It must never reach argv, the result JSON, the transcript, or any
        uploaded artifact.
        """
        env = os.environ.copy()
        if not env.get("PLAYBOT_OPENAI_API_KEY"):
            env["PLAYBOT_OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "")
        return env

    def _run_playbot(self, cmd: list) -> _PlaybotRun:
        deadline = self.deadline_seconds()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.build_env(),
            # Own session/process group, so the fallback kill reaches the Godot
            # editor and Codex helpers Playbot spawned.
            start_new_session=True,
        )
        drainers = [_PipeDrainer(process.stdout), _PipeDrainer(process.stderr)]
        for drainer in drainers:
            drainer.start()

        timed_out = False
        terminated_gracefully = False
        try:
            process.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminated_gracefully = self._stop_playbot(process)

        for drainer in drainers:
            drainer.join(timeout=OUTPUT_DRAIN_TIMEOUT_SECONDS)

        return _PlaybotRun(
            returncode=process.returncode,
            stdout=_redact_secrets(drainers[0].text()),
            stderr=_redact_secrets(drainers[1].text()),
            timed_out=timed_out,
            deadline_seconds=deadline,
            terminated_gracefully=terminated_gracefully,
        )

    @staticmethod
    def _stop_playbot(process: subprocess.Popen) -> bool:
        """Stop a run that outlived the deadline; returns True if SIGTERM sufficed.

        Playbot treats SIGTERM as "cancel this turn": it aborts the agent, stops
        the Godot editor it owns, and still writes the result and transcript.
        Killing the process group is the fallback, and also reaps engine
        processes if Playbot died without cleaning up after itself.
        """
        try:
            process.terminate()
        except OSError:
            return False

        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
            return True
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        return False

    def _interpret_run(
        self,
        run: _PlaybotRun,
        output_path: str,
        transcript_path: str,
        duration: float,
    ) -> SolverResult:
        transcript = _redact_secrets(self._read_capped(transcript_path))
        result_data, result_source = self._load_result(output_path, run.stdout)

        if result_data is None:
            return self._infrastructure_failure(
                run, transcript, duration, "Playbot wrote no readable result JSON"
            )

        schema_version = result_data.get("schema_version")
        if schema_version != SUPPORTED_RESULT_SCHEMA_VERSION:
            return self._infrastructure_failure(
                run,
                transcript,
                duration,
                f"Playbot result schema_version {schema_version!r} is not supported "
                f"(this solver reads schema {SUPPORTED_RESULT_SCHEMA_VERSION}); "
                "the packaged Playbot build and the harness are out of step",
            )

        status = result_data.get("status", "failed")
        reported_success = bool(result_data.get("success", False))
        success = reported_success and status == "completed"
        error = result_data.get("error") or {}
        error_message = error.get("message", "")
        artifact_error = result_data.get("artifact_error") or {}

        notes = []
        if reported_success and status != "completed":
            notes.append(f"success=true disagrees with status {status!r}")
        elif not reported_success and status == "completed":
            notes.append("success=false disagrees with status 'completed'")
        if artifact_error:
            # A run whose trajectory did not survive is not a usable result,
            # whatever the turn itself reported.
            success = False
            notes.append(
                "{} artifact error at {}: {}".format(
                    artifact_error.get("artifact", "artifact"),
                    artifact_error.get("path", "?"),
                    artifact_error.get("message", "unknown"),
                )
            )
        if run.timed_out:
            success = False
            notes.append(
                f"solver deadline of {run.deadline_seconds}s expired; "
                f"playbot was {run.shutdown_note()}"
            )
        if (run.returncode == 0) != reported_success:
            notes.append(
                f"exit code {run.returncode} disagrees with reported status {status!r}"
            )

        message = (
            result_data.get("final_message") or error_message or f"status: {status}"
        )
        if notes:
            message = f"{message} [{'; '.join(notes)}]"

        is_rate_limited = bool(result_data.get("rate_limited")) or (
            not success and self.is_rate_limit_error(error_message)
        )
        token_usage = self._parse_token_usage(result_data.get("token_usage"))
        reported_model = result_data.get("model") or self.model

        solver_result = SolverResult(
            success=success,
            message=message,
            duration_seconds=duration,
            stdout=transcript or run.stdout,
            stderr=f"{self._diagnostics(run, result_data, result_source)}\n{run.stderr}",
            is_rate_limited=is_rate_limited,
            token_usage=token_usage,
            model=reported_model,
        )
        solver_result.calculate_cost()

        if self.debug and token_usage:
            print(
                f"Tokens: input={token_usage.input_tokens}, "
                f"output={token_usage.output_tokens}, "
                f"total={token_usage.total_tokens}"
            )

        return solver_result

    def _infrastructure_failure(
        self, run: _PlaybotRun, transcript: str, duration: float, reason: str
    ) -> SolverResult:
        """Failures of the Playbot process itself, as opposed to the agent turn."""
        if run.timed_out:
            reason = (
                f"Playbot did not finish within {run.deadline_seconds}s and was "
                f"{run.shutdown_note()}; {reason}"
            )
        stderr_tail = run.stderr[-MAX_MESSAGE_STDERR:].strip()
        message = f"{reason} (exit code {run.returncode})"
        if stderr_tail:
            message = f"{message}: {stderr_tail}"

        return SolverResult(
            success=False,
            message=message,
            duration_seconds=duration,
            stdout=transcript or run.stdout,
            stderr=(
                f"[playbot] exit_code={run.returncode} timed_out={run.timed_out} "
                f"graceful_shutdown={run.terminated_gracefully}\n{run.stderr}"
            ),
            is_rate_limited=False,
            model=self.model,
        )

    @staticmethod
    def _diagnostics(run: _PlaybotRun, result_data: dict, result_source: str) -> str:
        """One-line provenance header, kept at the top of the trajectory's stderr."""
        engine = result_data.get("engine") or {}
        versions = result_data.get("versions") or {}
        fields = [
            ("exit_code", run.returncode),
            ("status", result_data.get("status")),
            ("schema_version", result_data.get("schema_version")),
            ("result_source", result_source),
            ("timed_out", run.timed_out),
            ("playbot_version", versions.get("playbot")),
            ("codex_version", versions.get("codex")),
            ("engine_kind", engine.get("kind")),
            ("engine_attached", engine.get("attached")),
            ("engine_version", engine.get("version")),
        ]
        return "[playbot] " + " ".join(f"{key}={value}" for key, value in fields)

    @classmethod
    def _load_result(cls, output_path: str, stdout: str) -> Tuple[Optional[dict], str]:
        """The requested file is authoritative; the stdout envelope is the fallback."""
        result = cls._read_json(output_path)
        if result is not None:
            return result, "output-file"
        result = cls._parse_marked_result(stdout)
        if result is not None:
            return result, "stdout-markers"
        return None, "missing"

    @staticmethod
    def _read_json(path: str) -> Optional[dict]:
        try:
            with open(path, "r") as f:
                parsed = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _parse_marked_result(stdout: str) -> Optional[dict]:
        """Recover the result envelope Playbot prints between marker lines.

        The last envelope wins: a run that fails to write its --output file
        prints the original result and then the rewritten failure envelope.
        """
        begin = stdout.rfind(RESULT_BEGIN_MARKER)
        if begin < 0:
            return None
        end = stdout.find(RESULT_END_MARKER, begin)
        if end < 0:
            return None
        try:
            parsed = json.loads(stdout[begin + len(RESULT_BEGIN_MARKER) : end])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _read_capped(path: str) -> str:
        try:
            with open(path, "r", errors="replace") as f:
                return f.read(MAX_CAPTURED_OUTPUT)
        except OSError:
            return ""

    @staticmethod
    def _parse_token_usage(usage: Optional[dict]) -> Optional[TokenUsage]:
        """Map playbot's token_usage block onto the harness TokenUsage.

        Playbot reports {input_tokens, cached_input_tokens, output_tokens,
        reasoning_output_tokens, total_tokens}. cached_input_tokens is a
        subset of input_tokens; reasoning tokens are billed as output, so
        fold them in when the totals show they are reported separately.
        """
        if not usage:
            return None

        def _int(key: str) -> int:
            value = usage.get(key)
            return int(value) if isinstance(value, (int, float)) else 0

        input_tokens = _int("input_tokens")
        output_tokens = _int("output_tokens")
        reasoning_tokens = _int("reasoning_output_tokens")
        total_tokens = _int("total_tokens")

        if reasoning_tokens and input_tokens + output_tokens + reasoning_tokens <= total_tokens:
            output_tokens += reasoning_tokens

        if total_tokens == 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=_int("cached_input_tokens"),
        )
