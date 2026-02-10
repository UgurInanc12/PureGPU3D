from __future__ import annotations

from threading import Event


def make_cancel_event() -> Event:
    return Event()
