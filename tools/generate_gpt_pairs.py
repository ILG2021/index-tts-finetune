#!/usr/bin/env python3
"""
Helper to batch-generate prompt/target pair manifests for IndexTTS2 GPT fine-tuning.

This script wraps `tools/build_gpt_prompt_pairs.py` so multiple processed datasets can
be updated in a single call without repeating boilerplate commands. It expects each
dataset directory to contain `train_manifest.jsonl` / `val_manifest.jsonl` produced by
the preprocessing pipeline and writes `gpt_pairs_train.jsonl` / `gpt_pairs_val.jsonl`
beside them (configurable via CLI flags).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from omegaconf import OmegaConf
    _USE_OMEGACONF = True
except ImportError:  # fall back to plain yaml
    import yaml as _yaml
    _USE_OMEGACONF = False

from build_gpt_prompt_pairs import (
    Sample,
    build_pairs,
    group_by_speaker,
    read_manifest,
)


def _load_gpt_limits(config_path: Path) -> Dict[str, int]:
    """Return {'max_mel_tokens': ..., 'max_text_tokens': ...} from a config YAML."""
    if _USE_OMEGACONF:
        cfg = OmegaConf.load(config_path)
        gpt = cfg.get("gpt", {})
        return {
            "max_mel_tokens": int(OmegaConf.select(cfg, "gpt.max_mel_tokens") or 0),
            "max_text_tokens": int(OmegaConf.select(cfg, "gpt.max_text_tokens") or 0),
        }
    else:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh)
        gpt = cfg.get("gpt", {}) if cfg else {}
        return {
            "max_mel_tokens": int(gpt.get("max_mel_tokens", 0)),
            "max_text_tokens": int(gpt.get("max_text_tokens", 0)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GPT prompt-target pair manifests for one or more processed datasets."
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Path to a processed dataset directory (repeat for multiple).",
    )
    parser.add_argument(
        "--train-manifest-name",
        type=str,
        default="train_manifest.jsonl",
        help="Filename of the single-utterance training manifest inside each dataset directory.",
    )
    parser.add_argument(
        "--val-manifest-name",
        type=str,
        default="val_manifest.jsonl",
        help="Filename of the single-utterance validation manifest inside each dataset directory.",
    )
    parser.add_argument(
        "--train-output-name",
        type=str,
        default="gpt_pairs_train.jsonl",
        help="Filename for the paired training manifest to write.",
    )
    parser.add_argument(
        "--val-output-name",
        type=str,
        default="gpt_pairs_val.jsonl",
        help="Filename for the paired validation manifest to write.",
    )
    parser.add_argument(
        "--pairs-per-target",
        type=int,
        default=2,
        help="Number of prompt samples to pair with each target utterance.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional cap on the number of pairs generated per manifest (0 = unlimited).",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=1,
        help="Skip targets with tokenised text length below this threshold.",
    )
    parser.add_argument(
        "--min-code-len",
        type=int,
        default=1,
        help="Skip targets with semantic code length below this threshold.",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=0,
        help="Skip targets whose text length exceeds this value. "
             "0 = read from --config (gpt.max_text_tokens); -1 = no limit.",
    )
    parser.add_argument(
        "--max-code-len",
        type=int,
        default=0,
        help="Skip targets whose semantic code length exceeds this value. "
             "0 = read from --config (gpt.max_mel_tokens); -1 = no limit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="Path to the IndexTTS config YAML (default: checkpoints/config.yaml). "
             "gpt.max_mel_tokens / gpt.max_text_tokens are used as upper-bound defaults "
             "for --max-code-len / --max-text-len unless those flags are set explicitly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed used when sampling prompt partners.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing paired manifests if present.",
    )
    return parser.parse_args()


def ensure_manifest(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"Manifest is not a file: {path}")


def write_pairs(pairs: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in pairs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_for_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    pairs_per_target: int,
    min_text_len: int,
    min_code_len: int,
    max_text_len: int = 0,
    max_code_len: int = 0,
    max_pairs: Optional[int],
) -> int:
    samples: List[Sample] = read_manifest(manifest_path)
    if not samples:
        print(f"[Generate] No entries in {manifest_path}; skipping.")
        if output_path.exists():
            output_path.unlink()
        return 0
    grouped = group_by_speaker(samples)
    pairs = build_pairs(
        grouped,
        pairs_per_target=pairs_per_target,
        min_text_len=min_text_len,
        min_code_len=min_code_len,
        max_text_len=max_text_len,
        max_code_len=max_code_len,
        max_pairs=max_pairs,
    )
    if not pairs:
        print(f"[Generate] No valid pairs generated for {manifest_path}; skipping.")
        if output_path.exists():
            output_path.unlink()
        return 0
    write_pairs(pairs, output_path)
    return len(pairs)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # --- Resolve length limits -------------------------------------------
    # Explicit CLI values of -1 mean "no limit"; 0 means "read from config".
    max_code_len: int = args.max_code_len
    max_text_len: int = args.max_text_len

    config_path: Optional[Path] = None
    raw_config = Path(args.config).expanduser()
    # Try as-is (works when called from project root or with an absolute path),
    # then try relative to the tools/ directory's parent (project root) when the
    # script is invoked from inside tools/.
    _this_dir = Path(__file__).resolve().parent
    for candidate in (raw_config.resolve(), (_this_dir.parent / raw_config).resolve()):
        if candidate.exists():
            config_path = candidate
            break

    if config_path is not None:
        limits = _load_gpt_limits(config_path)
        if max_code_len == 0 and limits["max_mel_tokens"] > 0:
            max_code_len = limits["max_mel_tokens"]
            print(f"[Generate] max_code_len set to {max_code_len} (from {config_path.name} gpt.max_mel_tokens)")
        if max_text_len == 0 and limits["max_text_tokens"] > 0:
            max_text_len = limits["max_text_tokens"]
            print(f"[Generate] max_text_len set to {max_text_len} (from {config_path.name} gpt.max_text_tokens)")
    elif args.max_code_len == 0 and args.max_text_len == 0:
        print(f"[Generate] Warning: config not found at '{args.config}'; length filters disabled.")

    # Normalise: -1 sentinel → 0 (build_pairs treats 0 as "no limit")
    if max_code_len < 0:
        max_code_len = 0
    if max_text_len < 0:
        max_text_len = 0
    # -----------------------------------------------------------------------

    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    for dataset in args.datasets:
        dataset_dir = Path(dataset).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not dataset_dir.is_dir():
            raise RuntimeError(f"Dataset path is not a directory: {dataset_dir}")

        train_manifest = dataset_dir / args.train_manifest_name
        val_manifest = dataset_dir / args.val_manifest_name
        train_output = dataset_dir / args.train_output_name
        val_output = dataset_dir / args.val_output_name

        ensure_manifest(train_manifest)
        ensure_manifest(val_manifest)

        if train_output.exists() and not args.force:
            raise FileExistsError(
                f"{train_output} already exists. Use --force to overwrite, or delete it manually."
            )
        if val_output.exists() and not args.force:
            raise FileExistsError(
                f"{val_output} already exists. Use --force to overwrite, or delete it manually."
            )

        print(f"[Generate] Dataset: {dataset_dir}")
        train_count = generate_for_manifest(
            train_manifest,
            train_output,
            pairs_per_target=args.pairs_per_target,
            min_text_len=args.min_text_len,
            min_code_len=args.min_code_len,
            max_text_len=max_text_len,
            max_code_len=max_code_len,
            max_pairs=max_pairs,
        )
        print(f"  - Wrote {train_count} train pairs -> {train_output.name}")

        val_count = generate_for_manifest(
            val_manifest,
            val_output,
            pairs_per_target=args.pairs_per_target,
            min_text_len=args.min_text_len,
            min_code_len=args.min_code_len,
            max_text_len=max_text_len,
            max_code_len=max_code_len,
            max_pairs=max_pairs,
        )
        print(f"  - Wrote {val_count} val pairs -> {val_output.name}")


if __name__ == "__main__":
    main()
