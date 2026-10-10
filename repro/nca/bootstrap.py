"""Fetch an immutable upstream checkout without executing upstream code."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
from .protocol import LOCK, verify_checkout


def bootstrap(destination: Path) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        return verify_checkout(destination)
    destination.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", LOCK["repository"]], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth=1", "origin", LOCK["commit"]], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
    return verify_checkout(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    bootstrap(args.out)
    print(f"Verified {LOCK['commit']} at {args.out.resolve()}")


if __name__ == "__main__":
    main()
