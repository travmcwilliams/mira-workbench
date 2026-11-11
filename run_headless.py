"""Backward-compatible headless runner — delegates to mira_cli."""

from __future__ import annotations

import sys

from mira_cli import main


def legacy_main() -> int:
    """Translate old flags into `mira run` invocation."""
    if len(sys.argv) > 1 and sys.argv[1] in ("acquire", "run"):
        return main()
    argv = ["run", *sys.argv[1:]]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(legacy_main())