"""Verify packaged bytes and undo the documented import-only transformation."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/sources.json").read_text())
    for entry in manifest["files"]:
        raw = (root / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["packaged_sha256"]:
            raise SystemExit("Packaged source changed: " + entry["file"])
        original = raw.replace(b"from .torch_module import ", b"from torch_module import ")
        if hashlib.sha256(original).hexdigest() != entry["upstream_sha256"]:
            raise SystemExit("Undocumented source transformation: " + entry["file"])
    print(json.dumps({"verified_core_modules": len(manifest["files"]),
                      "changes": "package-relative imports only"}))


if __name__ == "__main__":
    main()
