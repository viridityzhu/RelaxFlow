#!/usr/bin/env python3
"""Attach user-supplied prior images to a RelaxFlow dataset manifest.

The released Hugging Face datasets ship manifests with input observations and
``prior_text`` prompts. Generate or provide one or more prior images per sample,
arrange them as ``<priors_root>/<sample_id>/<file>``, then run this helper to
produce a manifest with ``prior_images`` that is ready for
``demo_relaxflow_batch.py``.

Required prior layout:

    <priors_root>/
    |-- <sample_id_1>/
    |   |-- prior_0.png
    |   |-- prior_1.png
    |   `-- ...
    `-- <sample_id_2>/
        `-- prior_0.png

Sample IDs must match the ``id`` field in ``manifest.json``. If a sample ID
contains slashes, keep the same nested directory structure under
``priors_root``. Accepted extensions are .png, .jpg, .jpeg, and .webp.

Example:

    python scripts/prepare_manifest_with_priors.py \\
        --manifest data/AmbiSem-3D/original/manifest.json \\
        --priors-root data/AmbiSem-3D/original/priors \\
        --output data/AmbiSem-3D/original/manifest_with_priors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIOR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def find_priors_for_sample(priors_root: Path, sample_id: str) -> list[Path]:
    sample_dir = priors_root / sample_id
    if not sample_dir.is_dir():
        return []
    return [
        entry
        for entry in sorted(sample_dir.iterdir())
        if entry.is_file() and entry.suffix.lower() in PRIOR_EXTENSIONS
    ]


def _load_manifest(manifest_path: Path) -> tuple[list[dict[str, Any]], bool]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return list(payload["samples"]), True
    if isinstance(payload, list):
        return payload, False
    raise ValueError(
        f"Expected a JSON list or a JSON object with 'samples' at {manifest_path}, "
        f"got {type(payload).__name__}."
    )


def _dump_manifest(entries: list[dict[str, Any]], output_path: Path, wrapped: bool) -> None:
    payload: Any = {"samples": entries} if wrapped else entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def attach_priors(
    manifest_path: Path,
    priors_root: Path,
    output_path: Path,
    *,
    require_all: bool = True,
    prior_text_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    entries, wrapped = _load_manifest(manifest_path)

    output_path = output_path.resolve()
    output_dir = output_path.parent
    priors_root = priors_root.resolve()
    overrides = prior_text_override or {}

    out_entries: list[dict[str, Any]] = []
    missing: list[str] = []
    with_priors = 0
    total_prior_files = 0

    for row in entries:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError(f"Malformed manifest entry (missing 'id'): {row!r}")
        sample_id = str(row["id"])
        new_entry = dict(row)

        prior_files = find_priors_for_sample(priors_root, sample_id)
        if prior_files:
            with_priors += 1
            total_prior_files += len(prior_files)
            try:
                rel_priors = [p.relative_to(output_dir).as_posix() for p in prior_files]
            except ValueError:
                rel_priors = [p.as_posix() for p in prior_files]
            new_entry["prior_images"] = rel_priors
        else:
            missing.append(sample_id)

        if sample_id in overrides:
            new_entry["prior_text"] = overrides[sample_id]
        out_entries.append(new_entry)

    stats = {
        "total": len(entries),
        "with_priors": with_priors,
        "total_prior_files": total_prior_files,
        "missing": missing,
        "output": output_path.as_posix(),
    }

    if require_all and missing:
        preview = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise SystemExit(
            f"Missing priors for {len(missing)}/{len(entries)} samples under "
            f"{priors_root}: {preview}\n"
            "Either add the missing priors, or rerun with --allow-missing."
        )

    _dump_manifest(out_entries, output_path, wrapped)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a released dataset manifest.json.",
    )
    parser.add_argument(
        "--priors-root",
        type=Path,
        required=True,
        help="Root directory containing <sample_id>/ subfolders of prior images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the manifest with prior_images attached.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the output even if some samples have no priors yet.",
    )
    parser.add_argument(
        "--prior-text-overrides",
        type=Path,
        default=None,
        help="Optional JSON file mapping sample_id to replacement prior_text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, str] | None = None
    if args.prior_text_overrides is not None:
        with args.prior_text_overrides.open("r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise SystemExit("--prior-text-overrides must contain a JSON object.")
        overrides = {str(key): str(value) for key, value in overrides.items()}

    stats = attach_priors(
        manifest_path=args.manifest,
        priors_root=args.priors_root,
        output_path=args.output,
        require_all=not args.allow_missing,
        prior_text_override=overrides,
    )
    print(
        f"Attached priors for {stats['with_priors']}/{stats['total']} samples "
        f"({stats['total_prior_files']} files total)."
    )
    if stats["missing"]:
        preview = ", ".join(stats["missing"][:10]) + (
            " ..." if len(stats["missing"]) > 10 else ""
        )
        print(f"Missing priors for {len(stats['missing'])} samples: {preview}")
    print(f"Wrote: {stats['output']}")


if __name__ == "__main__":
    main()
