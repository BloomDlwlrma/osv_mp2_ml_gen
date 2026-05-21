#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build passed molecule IDs from failed-only validation JSON.")
    parser.add_argument("--start-mol", type=int, required=True)
    parser.add_argument("--end-mol", type=int, required=True)
    parser.add_argument("--failed-json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def normalize_failed_ids(raw_failed):
    failed = set()
    for item in raw_failed:
        try:
            failed.add(int(item))
        except (TypeError, ValueError):
            continue
    return failed


def main():
    args = parse_args()
    failed_path = Path(args.failed_json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not failed_path.exists():
        raise FileNotFoundError(f"Missing failed-json file: {failed_path}")

    with failed_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    failed = normalize_failed_ids(payload.get("failed_mol_ids", []))
    all_ids = range(args.start_mol, args.end_mol + 1)
    passed = [mid for mid in all_ids if mid not in failed]

    with output_path.open("w", encoding="utf-8") as f:
        for mid in passed:
            f.write(f"{mid}\n")

    print(
        f"Wrote passed IDs: total={len(passed)} failed={len(failed)} "
        f"range=[{args.start_mol},{args.end_mol}] -> {output_path}"
    )


if __name__ == "__main__":
    main()
