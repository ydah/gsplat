#!/usr/bin/env python3
"""Generate deterministic Python-gsplat golden data for the Ruby implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from golden_cases import all_cases, generate, load_runtime, to_numpy


MAX_OUTPUT_BYTES = 50 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("test/golden"))
    parser.add_argument("--only", action="append", default=[], help="case name or family (repeatable)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="list cases without importing NumPy, torch, or gsplat")
    parser.add_argument("--require-all", action="store_true", help="fail instead of skipping CUDA-only cases")
    return parser.parse_args()


def selected_cases(filters: list[str]):
    cases = all_cases()
    if not filters:
        return cases
    selected = [case for case in cases if case.name in filters or case.family in filters]
    missing = sorted(set(filters) - {case.name for case in selected} - {case.family for case in selected})
    if missing:
        raise SystemExit(f"unknown case or family: {', '.join(missing)}")
    return selected


def main() -> int:
    args = parse_args()
    cases = selected_cases(args.only)
    if args.dry_run:
        for case in cases:
            requirement = " [CUDA]" if case.requires_cuda else ""
            print(f"{case.name}{requirement}")
        return 0

    runtime = load_runtime(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"gsplat_version": "1.5.3", "seed": 42, "device": str(runtime["device"]), "cases": {}}
    for case in cases:
        if case.requires_cuda and runtime["device"].type != "cuda":
            if args.require_all:
                raise SystemExit(f"{case.name} requires CUDA")
            manifest["cases"][case.name] = {"status": "skipped", "reason": "CUDA required"}
            print(f"skip {case.name}: CUDA required")
            continue

        output = args.out / f"{case.name}.npz"
        payload = to_numpy(generate(case, runtime), runtime)
        runtime["np"].savez_compressed(output, **payload)
        manifest["cases"][case.name] = {"status": "generated", "file": output.name}
        print(f"wrote {output}")

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_size = sum(path.stat().st_size for path in args.out.glob("*") if path.is_file())
    if output_size > MAX_OUTPUT_BYTES:
        raise SystemExit(f"golden output is {output_size} bytes; maximum is {MAX_OUTPUT_BYTES}")
    print(f"total: {output_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
