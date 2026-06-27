#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--prompt", default="def add(a, b):")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    try:
        health = _get(f"{base}/health")
        print("health:", json.dumps(health))
        if not health.get("model_loaded"):
            print("model not loaded; set MODEL_DIR and restart the server")
            return 1
        gen = _post(
            f"{base}/generate",
            {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens, "temperature": 0.8},
        )
        print("generate.text:", gen["text"][:200])
        print("smoke test passed")
        return 0
    except urllib.error.URLError as exc:
        print(f"smoke test failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
