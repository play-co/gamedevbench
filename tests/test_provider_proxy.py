import socket
import threading
import time

from gamedevbench.src.provider_proxy import _relay_bidirectional


def test_provider_stream_survives_idle_handshake_timeout():
    client, left = socket.socketpair()
    right, upstream = socket.socketpair()
    left.settimeout(0.05)
    right.settimeout(0.05)
    client.settimeout(2)
    upstream.settimeout(2)
    relay = threading.Thread(target=_relay_bidirectional, args=(left, right))
    relay.start()
    try:
        # A reasoning pause must not inherit the connection handshake deadline.
        time.sleep(0.15)
        upstream.sendall(b"model response")
        assert client.recv(1024) == b"model response"
        client.sendall(b"next request")
        assert upstream.recv(1024) == b"next request"
    finally:
        client.close()
        upstream.close()
        relay.join(timeout=3)
        left.close()
        right.close()
    assert not relay.is_alive()
