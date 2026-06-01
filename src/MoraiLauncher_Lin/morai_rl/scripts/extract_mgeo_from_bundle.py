from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import UnityPy


FIELD_TO_FILENAME = {
    "txNodeSetPath": "node_set.json",
    "txLinkSetPath": "link_set.json",
    "txV2XSetPath": "v2x_set.json",
    "txLaneNodeSetPath": "lane_node_set.json",
    "txLaneMarkingSetPath": "lane_marking_set.json",
    "txGlobalInfoPath": "global_info.json",
    "crosswalkDataSet": "crosswalk_set.json",
    "singlecrosswalkDataSet": "singlecrosswalk_set.json",
    "txSurfaceData": "surface_marking_set.json",
    "m_trafficSignSet": "traffic_sign_set.json",
    "m_trafficLightSet": "traffic_light_set.json",
    "m_intscnCtrllrSet": "intersection_controller_set.json",
    "m_intersectionCtrlData": "intersection_controller_data.json",
    "m_laneBoundarySet": "lane_boundary_set.json",
    "m_roadPolygonSet": "road_polygon_set.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MGeo-style TextAssets from a MORAI Unity scene bundle."
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        help=(
            "Path to the scene bundle or a map name such as daegumap. "
            "If a map name is provided, the script searches under --bundle-root."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("extracted_mgeo"),
        help="Directory where the extracted dataset will be written.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("MoraiLauncher_Lin_Data/Bundle"),
        help="Root directory that contains MORAI bundle version folders.",
    )
    parser.add_argument(
        "--bundle-version",
        default="latest",
        help=(
            "Bundle version folder to search, for example v.R2.241031.H2. "
            "Defaults to latest."
        ),
    )
    parser.add_argument(
        "--admodule",
        action="store_true",
        help=(
            "Write the extracted dataset into MORAI-ADModule/mgeo/lib/mgeo_data "
            "instead of --out-dir."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Override output folder name. Defaults to createMapName or bundle stem.",
    )
    parser.add_argument(
        "--list-maps",
        action="store_true",
        help="Print discoverable scene bundles under --bundle-root and exit.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON when possible.",
    )
    return parser.parse_args()


def read_textasset_string(obj) -> str:
    data = obj.read()
    raw = getattr(data, "m_Script", "")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def normalize_dataset_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "_")


def bundle_version_dirs(bundle_root: Path) -> list[Path]:
    if not bundle_root.exists():
        return []
    return [path for path in sorted(bundle_root.iterdir()) if path.is_dir()]


def latest_bundle_dir(bundle_root: Path) -> Path:
    version_dirs = bundle_version_dirs(bundle_root)
    if not version_dirs:
        raise SystemExit(f"No bundle version directories found under: {bundle_root.resolve()}")
    return version_dirs[-1]


def search_dirs(bundle_root: Path, bundle_version: str) -> list[Path]:
    if bundle_version and bundle_version.lower() != "latest":
        target = bundle_root / bundle_version
        if not target.is_dir():
            raise SystemExit(f"Bundle version not found: {target.resolve()}")
        return [target]
    return [latest_bundle_dir(bundle_root)]


def scene_candidates(search_roots: Iterable[Path]) -> list[Path]:
    scenes: list[Path] = []
    for root in search_roots:
        scenes.extend(
            sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".scene")
        )
    return scenes


def resolve_bundle_input(bundle_arg: str, bundle_root: Path, bundle_version: str) -> Path:
    candidate_path = Path(bundle_arg)
    if candidate_path.is_file():
        return candidate_path.resolve()

    roots = search_dirs(bundle_root.resolve(), bundle_version)
    scenes = scene_candidates(roots)
    if not scenes:
        searched = ", ".join(str(root) for root in roots)
        raise SystemExit(f"No .scene bundles found under: {searched}")

    lowered = bundle_arg.lower()
    accepted_names = {lowered}
    if not lowered.endswith(".scene"):
        accepted_names.add(f"{lowered}.scene")

    exact_matches = [path for path in scenes if path.name.lower() in accepted_names]
    if exact_matches:
        return exact_matches[0].resolve()

    partial_matches = [path for path in scenes if lowered in path.stem.lower() or lowered in path.name.lower()]
    if len(partial_matches) == 1:
        return partial_matches[0].resolve()

    searched = ", ".join(str(root) for root in roots)
    if partial_matches:
        match_names = ", ".join(path.name for path in partial_matches)
        raise SystemExit(
            f"Ambiguous map name '{bundle_arg}'. Matches: {match_names}. "
            f"Searched under: {searched}"
        )

    available = ", ".join(path.name for path in scenes)
    raise SystemExit(
        f"Could not find map '{bundle_arg}' under: {searched}\n"
        f"Available scenes: {available}"
    )


def print_discoverable_maps(bundle_root: Path) -> None:
    version_dirs = bundle_version_dirs(bundle_root.resolve())
    if not version_dirs:
        raise SystemExit(f"No bundle version directories found under: {bundle_root.resolve()}")

    print(f"Bundle root: {bundle_root.resolve()}")
    for version_dir in version_dirs:
        print(f"\n[{version_dir.name}]")
        scenes = scene_candidates([version_dir])
        for scene in scenes:
            print(f"  {scene.name}")


def extract_manager_data(env):
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            data = obj.read()
            script = data.m_Script.read()
            script_name = getattr(script, "m_Name", None) or getattr(script, "name", None)
        except Exception:
            continue
        if script_name != "MGeoDataManager":
            continue
        return data
    return None


def build_direct_textasset_map(env) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for obj in env.objects:
        if obj.type.name != "TextAsset":
            continue
        data = obj.read()
        name = getattr(data, "m_Name", None) or getattr(data, "name", None)
        if name:
            mapping[str(name)] = obj
    return mapping


def try_format_json(text: str, pretty: bool) -> str:
    if not pretty:
        return text
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"


def export_from_manager(env, manager_data, out_dir: Path, pretty: bool) -> list[Path]:
    written: list[Path] = []
    direct_assets = build_direct_textasset_map(env)

    for field_name, default_filename in FIELD_TO_FILENAME.items():
        ref = getattr(manager_data, field_name, None)
        asset_obj = None
        if ref is not None:
            try:
                asset_obj = ref.read()
            except Exception:
                asset_obj = None

        if asset_obj is None and field_name == "txLaneMarkingSetPath":
            asset_obj = direct_assets.get("lane_marking_set") or direct_assets.get("lane_boundary_set")
            if asset_obj is not None and "lane_boundary_set" in direct_assets and "lane_marking_set" not in direct_assets:
                default_filename = "lane_boundary_set.json"

        if asset_obj is None and field_name == "txSurfaceData":
            asset_obj = direct_assets.get("surface_marking_set") or direct_assets.get("road_polygon_set")
            if asset_obj is not None and "road_polygon_set" in direct_assets and "surface_marking_set" not in direct_assets:
                default_filename = "road_polygon_set.json"

        if asset_obj is None:
            continue

        if hasattr(asset_obj, "m_Script"):
            raw = getattr(asset_obj, "m_Script", "")
            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
        else:
            text = read_textasset_string(asset_obj)
        if not text.strip():
            continue

        output_path = out_dir / default_filename
        output_path.write_text(try_format_json(text, pretty), encoding="utf-8")
        written.append(output_path)

    return written


def main() -> None:
    args = parse_args()

    if args.list-maps:
        print_discoverable_maps(args.bundle_root)
        return

    if not args.bundle:
        raise SystemExit("bundle argument is required unless --list-maps is used")

    bundle_path = resolve_bundle_input(args.bundle, args.bundle_root, args.bundle_version)
    env = UnityPy.load(str(bundle_path))
    manager_data = extract_manager_data(env)

    if manager_data is None:
        raise SystemExit(f"MGeoDataManager not found in bundle: {bundle_path}")

    dataset_name = args.dataset_name or getattr(manager_data, "createMapName", "") or bundle_path.stem
    output_root = Path("MORAI-ADModule/mgeo/lib/mgeo_data") if args.admodule else args.out_dir
    out_dir = output_root.resolve() / normalize_dataset_name(str(dataset_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    written = export_from_manager(env, manager_data, out_dir, args.pretty)

    print(f"Bundle: {bundle_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {out_dir}")
    if written:
        print("Written files:")
        for path in written:
            print(f"  {path}")
    else:
        print("No non-empty MGeo TextAssets were exported.")


if __name__ == "__main__":
    main()
