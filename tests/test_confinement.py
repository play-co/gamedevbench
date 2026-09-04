import errno
import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from gamedevbench.src import benchmark_runner, confinement, provider_proxy
from gamedevbench.src.benchmark_runner import GodotBenchmarkRunner
from gamedevbench.src.confinement import (
    ConfinementError,
    _safe_environment,
    _secret_environment,
    build_bwrap_command,
    provider_hosts_for,
    run_confined_godot,
)
from gamedevbench.src.provider_proxy import (
    NonPublicAddressError,
    ProviderProxy,
    _connect_public_host,
    _parse_client_hello_sni,
    host_is_allowed,
)
from gamedevbench.src.utils.data_types import ValidationResult


@pytest.fixture(scope="session")
def bwrap_available():
    if platform.system() != "Linux" or shutil.which("bwrap") is None:
        pytest.skip("Bubblewrap integration test requires Linux and bwrap")
    completed = subprocess.run(
        [
            "bwrap", "--unshare-all", "--ro-bind", "/", "/",
            "--proc", "/proc", "/bin/true",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode:
        pytest.skip(f"Linux namespaces unavailable: {completed.stderr.strip()}")


@pytest.fixture(params=["local", "namespace"])
def proxy_probe(request, monkeypatch, tmp_path):
    """Run the same HTTP probes locally and inside the production namespace."""
    confined = request.param == "namespace"
    if confined:
        request.getfixturevalue("bwrap_available")

    workspace = tmp_path / "workspace"
    private_home = tmp_path / "home"
    output_dir = tmp_path / "output"
    proxy_dir = tmp_path / "proxy"
    for directory in (workspace, private_home, output_dir, proxy_dir):
        directory.mkdir()

    socket_path = proxy_dir / "provider.sock"
    with ProviderProxy(socket_path, ["api.meta.ai"]) as proxy:
        relay = None
        if not confined:
            relay = provider_proxy._ThreadingTcpServer(
                ("127.0.0.1", 0), provider_proxy._UnixRelayHandler
            )
            relay.unix_socket_path = str(socket_path)
            monkeypatch.setattr(confinement, "PROXY_PORT", relay.server_address[1])
            thread = threading.Thread(target=relay.serve_forever, daemon=True)
            thread.start()

        def run_probe(probe):
            command = [sys.executable, "-c", probe]
            environment = _safe_environment()
            if confined:
                command = build_bwrap_command(
                    agent="muse",
                    workspace=workspace,
                    private_home=private_home,
                    output_dir=output_dir,
                    proxy_dir=proxy_dir,
                    worker_config=output_dir / "config.json",
                    worker_output=output_dir / "result.json",
                    use_private_display=False,
                    # These probes do not run Godot; satisfy its required mount.
                    godot_path="/usr/bin/python3",
                    inner_command=command,
                )
                environment = None  # Bubblewrap installs the safe environment.
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            return json.loads(completed.stdout.strip().splitlines()[-1])

        try:
            yield run_probe, proxy
        finally:
            if relay is not None:
                relay.shutdown()
                relay.server_close()
                thread.join(timeout=5)


@contextmanager
def serve_http(host, tls_context=None):
    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"test-origin")

        def log_message(self, format, *args):
            pass

    try:
        server = Server((host, 0), Handler)
    except OSError as error:
        if host == "::1" and error.errno in (
            errno.EAFNOSUPPORT,
            errno.EADDRNOTAVAIL,
            errno.EPROTONOSUPPORT,
        ):
            pytest.skip(f"IPv6 loopback unavailable: {error}")
        raise
    with server:
        if tls_context is not None:
            server.socket = tls_context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_loopback_http_post_bypasses_provider_proxy(proxy_probe, host):
    run_probe, proxy = proxy_probe
    bind_host = "127.0.0.1" if host == "localhost" else host
    url_host = "[::1]" if host == "::1" else host
    result = run_probe(f"""
import errno, httpx, json, socket, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6 if {host!r} == "::1" else socket.AF_INET

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append([self.path, body.decode()])
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"namespace-service")

    def log_message(self, format, *args):
        pass

try:
    server = Server(({bind_host!r}, 0), Handler)
except OSError as error:
    if {host!r} == "::1" and error.errno in (
        errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EPROTONOSUPPORT
    ):
        print(json.dumps({{"skip": f"IPv6 loopback unavailable: {{error}}"}}))
        raise SystemExit(0)
    raise

with server:
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(timeout=3) as client:
            response = client.post(
                "http://{url_host}:" + str(server.server_port) + "/mcp",
                content=b"loopback-probe",
            )
        print(json.dumps({{
            "status": response.status_code,
            "body": response.text,
            "requests": server.requests,
        }}))
    finally:
        server.shutdown()
        thread.join(timeout=5)
""")
    if "skip" in result:
        pytest.skip(result["skip"])
    assert proxy.audit.to_dict() == {"allowed_connects": [], "denied_connects": []}
    assert result == {
        "status": 200,
        "body": "namespace-service",
        "requests": [["/mcp", "loopback-probe"]],
    }


def test_allowed_https_still_uses_provider_proxy(proxy_probe, monkeypatch, tmp_path):
    run_probe, proxy = proxy_probe
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("Local HTTPS provider fixture requires openssl")
    certificate = tmp_path / "workspace" / "provider.pem"
    private_key = tmp_path / "provider-key.pem"
    subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(private_key), "-out", str(certificate), "-days", "1",
            "-subj", "/CN=api.meta.ai", "-addext", "subjectAltName=DNS:api.meta.ai",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, private_key)
    with serve_http("127.0.0.1", context) as server:
        def connect_provider(host, port, timeout):
            assert (host, port) == ("api.meta.ai", 443)
            return socket.create_connection(server.server_address, timeout=timeout)

        # Only the upstream destination is replaced. CONNECT, SNI validation,
        # TLS, HTTP, and proxy audit all run through their real implementations.
        monkeypatch.setattr(provider_proxy, "_connect_public_host", connect_provider)
        result = run_probe("""
import httpx, json, ssl

context = ssl.create_default_context(cafile="provider.pem")
with httpx.Client(verify=context, timeout=3) as client:
    response = client.get("https://api.meta.ai/proxy-proof")
print(json.dumps({"status": response.status_code, "body": response.text}))
""")

    assert result == {"status": 200, "body": "test-origin"}
    assert proxy.audit.to_dict() == {
        "allowed_connects": ["api.meta.ai:443"],
        "denied_connects": [],
    }


@pytest.mark.parametrize(
    "url, destination, status",
    [
        ("https://raw.githubusercontent.com/", "raw.githubusercontent.com:443", 403),
        ("https://10.0.0.1/", "10.0.0.1:443", 403),
        ("https://api.meta.ai:8443/", "api.meta.ai:8443", 403),
        ("http://api.meta.ai/", "invalid-request", 400),
    ],
)
def test_prohibited_http_traffic_is_still_denied(proxy_probe, url, destination, status):
    run_probe, proxy = proxy_probe
    result = run_probe(f"""
import httpx, json

with httpx.Client(timeout=3) as client:
    try:
        response = client.get({url!r})
        result = {{"status": response.status_code}}
    except httpx.ProxyError as error:
        result = {{"proxy_error": str(error)}}
print(json.dumps(result))
""")
    if status == 403:
        assert "403" in result["proxy_error"]
    else:
        assert result == {"status": status}
    assert proxy.audit.to_dict() == {
        "allowed_connects": [],
        "denied_connects": [destination],
    }


@pytest.mark.parametrize("proxy_probe", ["namespace"], indirect=True)
@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_bwrap_loopback_does_not_expose_host_service(proxy_probe, host):
    run_probe, proxy = proxy_probe
    url_host = "[::1]" if host == "::1" else host
    with serve_http(host) as server:
        url = f"http://{url_host}:{server.server_port}/host-only"
        # Establish that the service really is reachable in the host namespace.
        with httpx.Client(trust_env=False, timeout=3) as client:
            assert client.get(url).text == "test-origin"
        result = run_probe(f"""
import httpx, json

with httpx.Client(timeout=3) as client:
    try:
        response = client.get({url!r})
        result = {{"status": response.status_code, "body": response.text}}
    except httpx.ConnectError:
        result = {{"host_service": "unreachable"}}
print(json.dumps(result))
""")
    assert result == {"host_service": "unreachable"}
    assert proxy.audit.to_dict() == {"allowed_connects": [], "denied_connects": []}


def test_provider_suffix_matching_does_not_allow_lookalikes():
    assert host_is_allowed("api.meta.ai", ["meta.ai"])
    assert host_is_allowed("meta.ai", ["meta.ai"])
    assert not host_is_allowed("meta.ai.evil.example", ["meta.ai"])
    assert not host_is_allowed("notmeta.ai", ["meta.ai"])
    assert not host_is_allowed("127.0.0.1", ["meta.ai"])


def test_muse_default_egress_allows_api_and_godot_documentation():
    hosts = provider_hosts_for("muse", "muse-spark-1.2")
    assert hosts == ("api.meta.ai", "docs.godotengine.org")
    assert host_is_allowed("api.meta.ai", hosts)
    assert host_is_allowed("docs.godotengine.org", hosts)
    assert host_is_allowed("preview.docs.godotengine.org", hosts)
    assert not host_is_allowed("dev.meta.ai", hosts)
    assert not host_is_allowed("godotengine.org", hosts)
    assert not host_is_allowed("facebook.com", hosts)


def test_claude_code_gateway_base_url_is_allowlisted_and_forwarded(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-token")
    hosts = provider_hosts_for("claude-code", None)
    assert "gateway.example.com" in hosts
    assert "api.anthropic.com" in hosts
    secrets = _secret_environment("claude-code", None)
    assert secrets["ANTHROPIC_AUTH_TOKEN"] == "gateway-token"
    assert secrets["ANTHROPIC_BASE_URL"] == "https://gateway.example.com/v1"
    # Other agents neither allowlist the gateway nor receive its token.
    assert "gateway.example.com" not in provider_hosts_for("muse", None)
    assert "ANTHROPIC_AUTH_TOKEN" not in _secret_environment("muse", None)


def test_claude_code_gateway_base_url_must_be_https_port_443(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gateway.example.com")
    with pytest.raises(ConfinementError):
        provider_hosts_for("claude-code", None)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com:8443")
    with pytest.raises(ConfinementError):
        provider_hosts_for("claude-code", None)


def test_secret_environment_is_not_placed_in_bubblewrap_arguments(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "canary-secret-must-not-enter-argv")
    assert "OPENAI_API_KEY" not in _safe_environment()
    assert "canary-secret-must-not-enter-argv" not in _safe_environment().values()


def test_safe_environment_limits_bypass_and_preserves_proxy_variables():
    environment = _safe_environment()
    assert environment["NO_PROXY"] == "localhost,127.0.0.1,::1"
    assert environment["no_proxy"] == environment["NO_PROXY"]
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        assert environment[key] == f"http://127.0.0.1:{confinement.PROXY_PORT}"


def test_custom_codex_provider_does_not_receive_unrelated_openai_key(
    monkeypatch, tmp_path
):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model_provider = "arena"\n', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-secret")
    assert "OPENAI_API_KEY" not in _secret_environment("codex", "gpt-5.6-sol")


def test_provider_proxy_rejects_non_allowlisted_connect(tmp_path):
    socket_path = tmp_path / "provider.sock"
    with ProviderProxy(socket_path, ["meta.ai"]) as proxy:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with client:
            client.connect(str(socket_path))
            client.sendall(
                b"CONNECT raw.githubusercontent.com:443 HTTP/1.1\r\n\r\n"
            )
            response = client.recv(1024)

    assert response.startswith(b"HTTP/1.1 403")
    assert proxy.audit.denied == ["raw.githubusercontent.com:443"]
    assert proxy.audit.allowed == []


def test_provider_proxy_rejects_private_dns_resolution(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(NonPublicAddressError):
        _connect_public_host("api.meta.ai", 443, 1.0)


def test_tls_client_hello_sni_cannot_front_another_domain():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    incoming = ssl.MemoryBIO()
    outgoing = ssl.MemoryBIO()
    tls = context.wrap_bio(
        incoming,
        outgoing,
        server_side=False,
        server_hostname="raw.githubusercontent.com",
    )
    with pytest.raises(ssl.SSLWantReadError):
        tls.do_handshake()

    assert _parse_client_hello_sni(outgoing.read()) == "raw.githubusercontent.com"


@pytest.mark.skipif(
    platform.system() != "Linux" or shutil.which("bwrap") is None,
    reason="Bubblewrap integration test requires Linux",
)
@pytest.mark.parametrize("proxy_probe", ["namespace"], indirect=True)
def test_bwrap_hides_ground_truth_siblings_and_public_network(proxy_probe, tmp_path):
    run_probe, proxy = proxy_probe
    workspace = tmp_path / "workspace"
    (workspace / "visible.txt").write_text("workspace-only", encoding="utf-8")
    forbidden_tmp = tmp_path / "forbidden.txt"
    forbidden_tmp.write_text("secret", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    probe = f"""
import httpx, json, os, socket
from pathlib import Path

result = {{
    'workspace': Path('/workspace/visible.txt').read_text(),
    'ground_truth_visible': Path({str(project_root / 'tasks_gt')!r}).exists(),
    'source_tasks_visible': Path({str(project_root / 'tasks')!r}).exists(),
    'sibling_tmp_visible': Path({str(forbidden_tmp)!r}).exists(),
    'host_ssh_visible': Path('/home/waynechi/.ssh').exists(),
    'hosts': Path('/etc/hosts').read_text(),
    'hosts_writable': os.access('/etc/hosts', os.W_OK),
}}
direct = socket.socket()
direct.settimeout(1)
try:
    direct.connect(('1.1.1.1', 443))
    result['direct_network'] = 'connected'
except OSError:
    result['direct_network'] = 'blocked'
finally:
    direct.close()

with httpx.Client(timeout=3) as client:
    try:
        client.get('https://raw.githubusercontent.com/')
        result['github_proxy'] = 'connected'
    except httpx.ProxyError as error:
        result['github_proxy'] = str(error)
print(json.dumps(result))
"""

    result = run_probe(probe)
    assert "403" in result.pop("github_proxy")
    assert result == {
        "workspace": "workspace-only",
        "ground_truth_visible": False,
        "source_tasks_visible": False,
        "sibling_tmp_visible": False,
        "host_ssh_visible": False,
        "hosts": "127.0.0.1 localhost\n::1 localhost\n",
        "hosts_writable": False,
        "direct_network": "blocked",
    }
    assert proxy.audit.denied == ["raw.githubusercontent.com:443"]


@pytest.mark.skipif(
    platform.system() != "Linux"
    or shutil.which("bwrap") is None
    or shutil.which("godot") is None,
    reason="Confined Godot integration test requires Linux, Bubblewrap, and Godot",
)
@pytest.mark.usefixtures("bwrap_available")
def test_validation_godot_has_no_host_files_credentials_or_network(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    forbidden = tmp_path / "host-secret.txt"
    forbidden.write_text("secret", encoding="utf-8")
    (workspace / "project.godot").write_text(
        """[application]
run/main_scene="res://main.tscn"

[display]
window/size/viewport_width=320
window/size/viewport_height=240

[rendering]
renderer/rendering_method="gl_compatibility"
""",
        encoding="utf-8",
    )
    (workspace / "main.tscn").write_text(
        """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://probe.gd" id="1"]

[node name="Probe" type="Node"]
script = ExtResource("1")
""",
        encoding="utf-8",
    )
    (workspace / "probe.gd").write_text(
        f"""extends Node

func _ready():
    var peer = StreamPeerTCP.new()
    peer.connect_to_host("1.1.1.1", 443)
    for step in range(20):
        peer.poll()
        OS.delay_msec(10)
    print("GDB_PROBE:", JSON.stringify({{
        "host_visible": FileAccess.file_exists({str(forbidden)!r}),
        "credentials_visible": OS.has_environment("OPENAI_API_KEY"),
        "network_connected": peer.get_status() == StreamPeerTCP.STATUS_CONNECTED,
    }}))
    get_tree().quit()
""",
        encoding="utf-8",
    )

    completed = run_confined_godot(
        workspace=workspace,
        godot_path="godot",
        godot_args=["--headless", "--path", "/workspace"],
        use_private_display=False,
        timeout_seconds=30,
    )

    probe_line = next(
        line.split("GDB_PROBE:", 1)[1]
        for line in (completed.stdout + completed.stderr).splitlines()
        if "GDB_PROBE:" in line
    )
    assert json.loads(probe_line) == {
        "host_visible": False,
        "credentials_visible": False,
        "network_connected": False,
    }


def test_runner_skips_validation_when_confinement_fails_closed(
    monkeypatch, tmp_path
):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task_0001"
    task_dir.mkdir(parents=True)
    (task_dir / "project.godot").write_text("config_version=5", encoding="utf-8")
    (task_dir / "task_config.json").write_text(
        json.dumps({"instruction": "test"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        benchmark_runner, "validate_confinement_available", lambda: "test-bwrap"
    )
    runner = GodotBenchmarkRunner(
        use_gt=False,
        agent="muse",
        model="muse-spark-1.2",
        confinement="strict",
    )
    runner.tasks_dir = tasks_dir
    runner.test_results_dir = tasks_dir / "test_result"
    monkeypatch.setattr(
        runner,
        "_run_godot_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    monkeypatch.setattr(
        benchmark_runner,
        "run_confined_solver",
        lambda **kwargs: (_ for _ in ()).throw(ConfinementError("probe failed")),
    )
    validation_called = False

    def unexpected_validation(*args, **kwargs):
        nonlocal validation_called
        validation_called = True
        raise AssertionError("validation must not run after confinement failure")

    monkeypatch.setattr(runner, "_validate_in_directory", unexpected_validation)
    result_dir = tmp_path / "saved-result"
    result_dir.mkdir()
    monkeypatch.setattr(runner, "_save_test_result", lambda *args: result_dir)

    result = runner._run_benchmark_with_agent("task_0001")

    assert not validation_called
    assert not result["success"]
    assert result["confinement"]["status"] == "failed-closed"


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_solver_workspace_rejects_links_and_special_files(tmp_path, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if kind == "symlink":
        (workspace / "escape").symlink_to("/etc/passwd")
    else:
        os.mkfifo(workspace / "escape")

    with pytest.raises(ConfinementError, match="forbidden"):
        GodotBenchmarkRunner._assert_safe_solver_workspace(workspace)


def test_unsafe_result_tree_is_not_copied(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "escape").symlink_to("/etc/passwd")
    runner = GodotBenchmarkRunner(use_gt=False, confinement="strict")
    runner.test_results_dir = tmp_path / "results"

    saved = runner._save_test_result(
        source,
        "task_0001",
        validation_result=ValidationResult(False, "failed closed"),
        confinement_metadata={"status": "failed-closed"},
        copy_task_files=False,
    )

    assert not (saved / "escape").exists()
    assert json.loads((saved / "result.json").read_text())["confinement"] == {
        "status": "failed-closed"
    }
