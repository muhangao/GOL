"""Mechanical execution repairs, never undisclosed scientific recipe changes.

The pristine checkout remains unchanged. These transformations are applied to a
per-run snapshot, checked for exact matches, and recorded with before/after hashes.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

REPAIRS = (
    ("R01", "src/openwebtext_pt.py", "    DownstreamLanguageModel,\n", "",
     "Remove an unavailable, unused GPT-2 import; this replay only allows Llama."),
    ("R02", "src/openwebtext_pt.py", "compiled_model._freeze_unfreeze_modules()",
     "model._freeze_unfreeze_modules()",
     "Call the same module method through the original model, not the DataParallel wrapper."),
    ("R03", "utils/dataset_utils.py", "return len(self.data) // self.block_size",
     "return max(0, (len(self.data) - 1) // self.block_size)",
     "Exclude a final block without its next-token target; no padding or synthetic token is added."),
)


def repaired_text(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Repair expected exactly one match for {old!r}")
    return text.replace(old, new, 1)


def apply_repairs(root: Path, stage: str) -> list[dict]:
    if stage == "source":
        return []
    results = []
    for key, rel, old, new, reason in REPAIRS:
        path = root / rel
        before = path.read_bytes()
        after = repaired_text(before.decode(), old, new).encode()
        path.write_bytes(after)
        results.append(dict(id=key, path=rel, reason=reason,
                            before_sha256=hashlib.sha256(before).hexdigest(),
                            after_sha256=hashlib.sha256(after).hexdigest()))
    return results
