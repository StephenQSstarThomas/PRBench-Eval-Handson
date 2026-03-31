"""Dynamic port allocation utilities for parallel evaluation."""

import socket


def find_free_port() -> int:
    """Find a free TCP port by binding to port 0 and reading the assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_free_port_pair() -> tuple[int, int]:
    """Find two free TCP ports for green and white agents."""
    p1 = find_free_port()
    p2 = find_free_port()
    # Ensure they're different
    while p2 == p1:
        p2 = find_free_port()
    return p1, p2
