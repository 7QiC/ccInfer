#!/usr/bin/env python3
"""ccInfer server lifecycle management."""

import subprocess
import time
import signal
import requests
from typing import Optional
from pathlib import Path


class ServerHarness:
    """Manages ccInfer server process lifecycle."""

    def __init__(self, binary: str, model_path: str, port: int = 8080,
                 extra_args: Optional[list[str]] = None):
        self.binary = binary
        self.model_path = model_path
        self.port = port
        self.extra_args = extra_args or []
        self._process: Optional[subprocess.Popen] = None
        self._base_url = f"http://localhost:{port}"

    @property
    def base_url(self) -> str:
        return self._base_url

    def start(self, timeout: float = 180.0) -> None:
        """Start server and wait until healthy."""
        cmd = [self.binary, "--port", str(self.port),
               "--model-path", self.model_path] + self.extra_args
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(f"Server exited early: {stderr}")
            try:
                r = requests.get(f"{self._base_url}/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.5)
        self.stop(timeout=2.0)
        raise TimeoutError(f"Server did not become healthy within {timeout}s")

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stop the server."""
        if self._process is None:
            return
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
