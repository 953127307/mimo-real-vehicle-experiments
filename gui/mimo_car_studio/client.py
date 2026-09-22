from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any

from .protocol import JsonLineDecoder, encode_message


class CarClient:
    def __init__(self) -> None:
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10_000)
        self._socket: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._connected = False
        self._last_tx = 0.0

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connected

    def connect(self, host: str, port: int, timeout: float = 4.0) -> None:
        self.disconnect()
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(0.20)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = sock
        self._stop.clear()
        with self._state_lock:
            self._connected = True
        self._thread = threading.Thread(target=self._reader_loop, name="car-client", daemon=True)
        self._thread.start()
        self.send("hello")

    def disconnect(self) -> None:
        self._stop.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=0.8)
        self._thread = None
        with self._state_lock:
            self._connected = False

    def send(self, command: str, **parameters: Any) -> int:
        with self._send_lock:
            if self._socket is None or not self.connected:
                raise ConnectionError("not connected to car")
            self._sequence += 1
            frame = {"v": 1, "seq": self._sequence, "cmd": command, **parameters}
            self._socket.sendall(encode_message(frame))
            self._last_tx = time.monotonic()
            return self._sequence

    def drain(self, limit: int = 1000) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for _ in range(limit):
            try:
                result.append(self.messages.get_nowait())
            except queue.Empty:
                break
        return result

    def _post(self, message: dict[str, Any]) -> None:
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            self.messages.put_nowait(message)

    def _reader_loop(self) -> None:
        decoder = JsonLineDecoder()
        reason = "connection_closed"
        try:
            while not self._stop.is_set():
                if time.monotonic() - self._last_tx > 0.45:
                    self.send("hello")
                sock = self._socket
                if sock is None:
                    break
                try:
                    data = sock.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    break
                for message in decoder.feed(data):
                    self._post(message)
        except Exception as exc:  # Delivered to the GUI instead of killing the worker silently.
            reason = str(exc)
        finally:
            with self._state_lock:
                self._connected = False
            self._post({"type": "connection_closed", "message": reason})
