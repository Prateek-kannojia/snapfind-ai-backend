from __future__ import annotations


class AppError(Exception):
    """Base for every exception allowed to cross into an HTTP response.

    Any subclass raised inside a route (or inside a service function a route
    calls) is caught by the single handler registered in main.py and turned
    into a JSON response — no route needs its own try/except for it.

    status_code: the HTTP status main.py's handler sends back.
    error_code: a short, stable, machine-readable identifier (e.g.
    "job_not_found") so a client (the Android app) can branch on a fixed
    string instead of parsing the human-readable message, which is free to
    change wording without breaking callers.
    """

    def __init__(self, message: str, *, status_code: int = 400, error_code: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
