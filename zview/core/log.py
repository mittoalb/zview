"""Tiny ANSI-coloured logger used across zview.

Honours the `NO_COLOR` env var and auto-disables when stderr is not a TTY, so
piping to a file or running under a dumb terminal gets plain text. Everything
goes to stderr, which keeps application output cleanly separable from logs.
"""

import os
import sys

_USE_COLOR = (
    sys.stderr.isatty()
    and os.environ.get("NO_COLOR", "") == ""
    and os.environ.get("TERM", "") != "dumb"
)

_RESET = "\033[0m"
_LEVELS = {
    "debug":   ("\033[90m", "DBG"),   # grey
    "info":    ("\033[36m", "INF"),   # cyan
    "success": ("\033[32m", "OK "),   # green
    "warn":    ("\033[33m", "WRN"),   # yellow
    "error":   ("\033[31m", "ERR"),   # red
}


def _emit(level, msg):
    color, tag = _LEVELS[level]
    if _USE_COLOR:
        print(f"{color}[{tag}]{_RESET} {msg}", file=sys.stderr)
    else:
        print(f"[{tag}] {msg}", file=sys.stderr)


def debug(msg):   _emit("debug", msg)
def info(msg):    _emit("info", msg)
def success(msg): _emit("success", msg)
def warn(msg):    _emit("warn", msg)
def error(msg):   _emit("error", msg)
