"""Entry point: python -m aegis <command>. Phase 0 provides `doctor`."""
from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("usage: python -m aegis doctor")
        return 0
    if args[0] == "doctor":
        from .doctor import run
        return run()
    print(f"unknown command: {args[0]} (Phase 0 provides: doctor)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
