"""Command-line helper to build Pikalytics compendium CSVs.

This wraps the utilities in `pikalytics_util.py` so you can run:

    python generate_pikalytics_compendium.py gen9vgc2025regh --min-usage 0.5

It will reuse cached HTML in `pikalytics_cache/` or fetch from the web if
available, then emit either the multi-file CSV set or a single CSV depending on
flags.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Iterable, Dict, Set

import pandas as pd

from pikalytics_util import (
    build_and_save_compendium_csv,
    build_and_save_compendium_single_csv,
    fetch_overview,
    fetch_details,
    CACHE_DIR,
)


def _unique(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in seq:
        key = str(item)
        if key not in seen:
            seen.add(key)
            out.append(str(item))
    return out


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def generate_compendium(
    format_slug: str,
    min_usage: float,
    top_moves: int,
    top_items: int,
    top_abilities: int,
    out_dir: str | None,
    single_file: bool,
    use_cache: bool,
) -> List[str]:
    """Run the appropriate builder and return the written file paths."""

    if single_file:
        path = build_and_save_compendium_single_csv(
            format_slug=format_slug,
            min_usage=min_usage,
            top_moves=top_moves,
            top_items=top_items,
            top_abilities=top_abilities,
            out_dir=out_dir,
            use_cache=use_cache,
        )
        return [path]

    return build_and_save_compendium_csv(
        format_slug=format_slug,
        min_usage=min_usage,
        top_moves=top_moves,
        top_items=top_items,
        top_abilities=top_abilities,
        out_dir=out_dir,
        use_cache=use_cache,
    )


def load_stats_names(path: str) -> List[str]:
    """Return unique Pokémon names from a stats CSV."""
    df = pd.read_csv(path)
    col = None
    for candidate in ("pokemon", "name", "Name"):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(f"Stats CSV {path} must contain a 'pokemon' or 'name' column.")
    names = []
    for val in df[col].dropna().unique():
        text = str(val).strip()
        if text:
            names.append(text)
    names = _unique(names)
    names.sort()
    return names


def write_usage_report(
    names: Iterable[str],
    usage_map: Dict[str, float],
    min_usage: float,
    out_path: str,
) -> str:
    """Write a CSV summarizing usage for the provided names."""
    rows = []
    lower_map = {k.lower(): v for k, v in usage_map.items()}
    slug_map = {_normalize_name(k): v for k, v in usage_map.items()}
    for name in names:
        usage = lower_map.get(name.lower())
        if usage is None:
            usage = slug_map.get(_normalize_name(name))
        rows.append({
            "pokemon": name,
            "usage": float(usage) if usage is not None else 0.0,
            "meets_threshold": bool(usage is not None and usage >= min_usage),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip().lower()).strip("_")


def fetch_details_for_names(
    names: Iterable[str],
    format_slug: str,
    use_cache: bool,
) -> None:
    """Ensure detail pages are cached by fetching each Pokémon once."""
    seen: Set[str] = set()
    for name in names:
        slug = _slugify(name)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        try:
            fetch_details(name, format_slug=format_slug, use_cache=use_cache)
        except Exception as exc:
            print(f"Warning: failed to fetch details for {name} ({slug}): {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Pikalytics usage compendium CSV files.",
    )
    parser.add_argument(
        "format_slug",
        nargs="?",
        default="gen9vgc2025regh",
        help="Pikalytics format slug (default: %(default)s).",
    )
    parser.add_argument(
        "--min-usage",
        type=float,
        default=0.05,
        help="Minimum usage percentage to include (default: %(default)s).",
    )
    parser.add_argument(
        "--top-moves",
        type=int,
        default=8,
        help="Number of top moves to keep per Pokémon (default: %(default)s).",
    )
    parser.add_argument(
        "--top-items",
        type=int,
        default=6,
        help="Number of top items to keep per Pokémon (default: %(default)s).",
    )
    parser.add_argument(
        "--top-abilities",
        type=int,
        default=4,
        help="Number of top abilities to keep per Pokémon (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write CSVs (default: use pikalytics_util cache).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Write a single combined CSV instead of four separate files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached HTML and refetch from Pikalytics if possible.",
    )
    parser.add_argument(
        "--stats-csv",
        default="pokemon.csv",
        help="Optional stats CSV to cross-check usage (default: pokemon.csv if present).",
    )
    parser.add_argument(
        "--skip-usage-report",
        action="store_true",
        help="Do not generate the auxiliary usage summary CSV.",
    )
    parser.add_argument(
        "--skip-details",
        action="store_true",
        help="Skip fetching per-Pokémon detail pages.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    usage_pairs = fetch_overview(args.format_slug, use_cache=not args.no_cache)
    usage_map = {name: float(usage) for name, usage in usage_pairs if isinstance(usage, (int, float))}
    if not usage_map:
        print("Warning: No usage data retrieved; resulting CSVs may be empty. Ensure cache HTML exists or rerun with --no-cache.")

    stats_names: List[str] | None = None
    if args.stats_csv and os.path.exists(args.stats_csv):
        try:
            stats_names = load_stats_names(args.stats_csv)
        except Exception as exc:
            print(f"Warning: Could not read stats CSV '{args.stats_csv}': {exc}")
            stats_names = None
    elif args.stats_csv and args.stats_csv != "pokemon.csv":
        print(f"Warning: stats CSV '{args.stats_csv}' not found; skipping usage report.")

    if stats_names:
        slugged_stats: Set[str] = {_normalize_name(n) for n in stats_names}
        added = False
        for usage_name in usage_map.keys():
            slug = _normalize_name(usage_name)
            if slug not in slugged_stats:
                stats_names.append(usage_name)
                slugged_stats.add(slug)
                added = True
        if added:
            stats_names = _unique(stats_names)
            stats_names.sort()

    detail_names = stats_names or list(usage_map.keys())
    if detail_names and not args.skip_details:
        print(f"Fetching detail pages for {len(detail_names)} Pokémon (use_cache={not args.no_cache})...")
        fetch_details_for_names(detail_names, args.format_slug, use_cache=not args.no_cache)

    if stats_names and not args.skip_usage_report:
        output_dir = args.out_dir or CACHE_DIR
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"usage_{args.format_slug}.csv")
        write_usage_report(stats_names, usage_map, args.min_usage, report_path)
        print(f"Wrote usage summary: {report_path}")

    paths = generate_compendium(
        format_slug=args.format_slug,
        min_usage=args.min_usage,
        top_moves=args.top_moves,
        top_items=args.top_items,
        top_abilities=args.top_abilities,
        out_dir=args.out_dir,
        single_file=args.single,
        use_cache=not args.no_cache,
    )

    if not paths:
        print("No files were generated.")
        return 1

    print("Generated the following files:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
