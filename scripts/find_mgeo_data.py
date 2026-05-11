from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


MGOE_CORE_FILES = ("global_info.json", "node_set.json", "link_set.json")
MGOE_OPTIONAL_FILES = (
    "lane_node_set.json",
    "lane_marking_set.json",
    "surface_marking_set.json",
    "traffic_sign_set.json",
    "traffic_light_set.json",
    "synced_traffic_light_set.json",
    "intersection_controller_set.json",
    "crosswalk_set.json",
    "singlecrosswalk_set.json",
    "road_polygon_set.json",
    "lane_boundary_set.json",
)
DEFAULT_BUNDLE_SUFFIXES = (
    ".scene",
    ".vehicle",
    ".obstacle",
    ".pedestrian",
    ".function",
    ".material",
)
DEFAULT_BUNDLE_KEYWORDS = (
    "mgeo",
    "node_set",
    "link_set",
    "lane_marking",
    "traffic_light",
    "surface_marking",
    "road_mesh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find MGeo JSON folders and optionally scan Unity bundles with UnityPy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to search from. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--scan-bundles",
        action="store_true",
        help="Scan Unity asset bundles with UnityPy for MGeo-related names/text.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("MoraiLauncher_Lin_Data/Bundle"),
        help="Bundle directory used with --scan-bundles.",
    )
    parser.add_argument(
        "--bundle-filter",
        default="",
        help="Only scan bundle files whose path contains this substring.",
    )
    parser.add_argument(
        "--bundle-limit",
        type=int,
        default=30,
        help="Maximum number of bundle hits to print.",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        dest="keywords",
        default=[],
        help="Additional keyword to search for in bundle scans. Can be repeated.",
    )
    return parser.parse_args()


def looks_like_mgeo_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in MGOE_CORE_FILES)


def count_json_items(path: Path) -> int | str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return f"ERR:{exc}"

    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return "N/A"


def summarize_mgeo_dir(path: Path) -> dict[str, int | str]:
    summary: dict[str, int | str] = {}
    for name in MGOE_CORE_FILES + MGOE_OPTIONAL_FILES:
        file_path = path / name
        if file_path.exists():
            summary[name] = count_json_items(file_path)
    return summary


def find_mgeo_dirs(root: Path) -> list[Path]:
    results: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if looks_like_mgeo_dir(candidate):
            results.append(candidate)
    return results


def print_mgeo_report(root: Path, directories: Iterable[Path]) -> None:
    directories = list(directories)
    print(f"MGeo directories under: {root.resolve()}")
    print(f"Found: {len(directories)}")
    for directory in directories:
        rel = directory.resolve().relative_to(root.resolve())
        summary = summarize_mgeo_dir(directory)
        print(f"\n[{rel}]")
        for name, count in summary.items():
            print(f"  {name}: {count}")


def load_unitypy():
    try:
        import UnityPy  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "UnityPy is not installed. Run without --scan-bundles or install UnityPy."
        ) from exc
    return UnityPy


def bundle_candidates(bundle_root: Path, bundle_filter: str) -> list[Path]:
    candidates = [
        path
        for path in sorted(bundle_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in DEFAULT_BUNDLE_SUFFIXES
    ]
    if bundle_filter:
        lowered = bundle_filter.lower()
        candidates = [path for path in candidates if lowered in str(path).lower()]
    return candidates


def object_text_candidate(obj, data) -> str:
    parts = [obj.type.name]
    name = getattr(data, "name", None)
    container = getattr(obj, "container", None)
    if name:
        parts.append(str(name))
    if container:
        parts.append(str(container))

    if obj.type.name == "TextAsset":
        script = getattr(data, "script", b"")
        if isinstance(script, str):
            text = script
        else:
            text = bytes(script).decode("utf-8", errors="ignore")
        if text:
            parts.append(text[:4000])

    return "\n".join(parts)


def scan_bundles(
    bundle_root: Path,
    bundle_filter: str,
    keywords: list[str],
    limit: int,
) -> list[str]:
    UnityPy = load_unitypy()
    lowered_keywords = [keyword.lower() for keyword in keywords]
    hits: list[str] = []

    for bundle_path in bundle_candidates(bundle_root, bundle_filter):
        try:
            env = UnityPy.load(str(bundle_path))
        except Exception as exc:
            hits.append(f"{bundle_path}: failed to load bundle ({exc})")
            if len(hits) >= limit:
                break
            continue

        for obj in env.objects:
            try:
                data = obj.read()
                text = object_text_candidate(obj, data).lower()
            except Exception:
                continue

            if any(keyword in text for keyword in lowered_keywords):
                hits.append(f"{bundle_path}: {obj.type.name}")
                if len(hits) >= limit:
                    return hits

    return hits


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    directories = find_mgeo_dirs(root)
    print_mgeo_report(root, directories)

    if args.scan_bundles:
        keywords = list(DEFAULT_BUNDLE_KEYWORDS)
        for keyword in args.keywords:
            if keyword not in keywords:
                keywords.append(keyword)

        print("\nBundle scan")
        print(f"Bundle root: {args.bundle_root.resolve()}")
        print(f"Keywords: {', '.join(keywords)}")
        hits = scan_bundles(
            bundle_root=args.bundle_root.resolve(),
            bundle_filter=args.bundle_filter,
            keywords=keywords,
            limit=args.bundle_limit,
        )
        if hits:
            for hit in hits:
                print(f"  {hit}")
        else:
            print("  No MGeo-like bundle hits found with the current keywords.")


if __name__ == "__main__":
    main()
