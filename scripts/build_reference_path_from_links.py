from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_selected_link_ids(path: Path, selection_key: str = "selected_link_ids") -> tuple[list[str], list[dict]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            if selection_key in data:
                return list(data.get(selection_key, [])), list(data.get("partial_links", []))
            if selection_key != "selected_link_ids":
                return [], list(data.get("partial_links", []))
            return list(data.get("selected_link_ids", [])), list(data.get("partial_links", []))
        if isinstance(data, list):
            return list(data), []
        raise ValueError("selected link json must be a list or contain selected_link_ids")

    with path.open("r", encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    return ids, []


def load_link_map(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {item["idx"]: item for item in raw}


def is_lane_change_link(link_id: str) -> bool:
    return "-" in link_id


def apply_partial_points(points: list[list[float]], start_ratio: float, end_ratio: float) -> list[list[float]]:
    if not points:
        return []
    start_ratio = max(0.0, min(1.0, float(start_ratio)))
    end_ratio = max(0.0, min(1.0, float(end_ratio)))
    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio

    start_idx = int(math.floor((len(points) - 1) * start_ratio))
    end_idx = int(math.ceil((len(points) - 1) * end_ratio))
    return points[start_idx : end_idx + 1]


def build_partial_lookup(partial_links: list[dict]) -> dict[str, tuple[float, float]]:
    lookup: dict[str, tuple[float, float]] = {}
    for item in partial_links:
        link_id = item.get("link_id")
        if not link_id:
            continue
        lookup[link_id] = (
            float(item.get("start_ratio", 0.0)),
            float(item.get("end_ratio", 1.0)),
        )
    return lookup


def order_links_by_connectivity(link_ids: list[str], link_map: dict[str, dict]) -> list[str]:
    selected = [link_id for link_id in link_ids if link_id in link_map]
    if len(selected) < 2:
        return selected

    from_nodes = {link_id: link_map[link_id]["from_node_idx"] for link_id in selected}
    to_nodes = {link_id: link_map[link_id]["to_node_idx"] for link_id in selected}

    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for link_id in selected:
        outgoing.setdefault(from_nodes[link_id], []).append(link_id)
        incoming.setdefault(to_nodes[link_id], []).append(link_id)

    start_candidates = [
        link_id
        for link_id in selected
        if from_nodes[link_id] not in incoming
    ]
    start_link = start_candidates[0] if len(start_candidates) == 1 else selected[0]

    ordered = [start_link]
    visited = {start_link}
    current = start_link

    while True:
        next_candidates = [
            link_id
            for link_id in outgoing.get(to_nodes[current], [])
            if link_id not in visited
        ]
        if not next_candidates:
            break
        next_link = next_candidates[0]
        ordered.append(next_link)
        visited.add(next_link)
        current = next_link

    remaining = [link_id for link_id in selected if link_id not in visited]
    ordered.extend(remaining)
    return ordered


def build_outgoing_map(link_ids: list[str], link_map: dict[str, dict]) -> dict[str, list[str]]:
    outgoing: dict[str, list[str]] = {link_id: [] for link_id in link_ids}
    from_node_to_links: dict[str, list[str]] = {}
    for link_id in link_ids:
        from_node_to_links.setdefault(link_map[link_id]["from_node_idx"], []).append(link_id)

    for link_id in link_ids:
        to_node = link_map[link_id]["to_node_idx"]
        outgoing[link_id] = list(from_node_to_links.get(to_node, []))
    return outgoing


def build_incoming_map(link_ids: list[str], outgoing_map: dict[str, list[str]]) -> dict[str, list[str]]:
    incoming: dict[str, list[str]] = {link_id: [] for link_id in link_ids}
    for link_id, next_ids in outgoing_map.items():
        for next_id in next_ids:
            incoming.setdefault(next_id, []).append(link_id)
    return incoming


def lateral_neighbors(link_id: str, link_map: dict[str, dict], selected: set[str]) -> list[str]:
    item = link_map[link_id]
    neighbors: list[str] = []
    for key in ("left_lane_change_dst_link_idx", "right_lane_change_dst_link_idx"):
        neighbor = item.get(key)
        if neighbor and neighbor in selected:
            neighbors.append(neighbor)
    return neighbors


def ordered_lateral_bundle(seed_link_id: str, link_map: dict[str, dict], selected: set[str]) -> list[str]:
    bundle = {seed_link_id}
    stack = [seed_link_id]
    while stack:
        current = stack.pop()
        for neighbor in lateral_neighbors(current, link_map, selected):
            if neighbor not in bundle:
                bundle.add(neighbor)
                stack.append(neighbor)

    leftmost = None
    for link_id in bundle:
        left_neighbor = link_map[link_id].get("left_lane_change_dst_link_idx")
        if not left_neighbor or left_neighbor not in bundle:
            leftmost = link_id
            break
    if leftmost is None:
        leftmost = seed_link_id

    ordered: list[str] = []
    visited: set[str] = set()
    current = leftmost
    while current and current not in visited:
        ordered.append(current)
        visited.add(current)
        right_neighbor = link_map[current].get("right_lane_change_dst_link_idx")
        if right_neighbor in bundle:
            current = right_neighbor
        else:
            current = None

    if len(visited) != len(bundle):
        for link_id in sorted(bundle):
            if link_id not in visited:
                ordered.append(link_id)
    return ordered


def choose_bundle_representative(
    bundle: list[str],
    downstream_score_fn,
) -> str:
    if len(bundle) == 1:
        return bundle[0]

    center = (len(bundle) - 1) / 2.0
    candidates = sorted(
        bundle,
        key=lambda link_id: (
            abs(bundle.index(link_id) - center),
            -downstream_score_fn(link_id),
            link_id,
        ),
    )
    return candidates[0]


def build_centerline_link_sequence(link_ids: list[str], link_map: dict[str, dict]) -> list[str]:
    selected = set(link_ids)
    outgoing_map = build_outgoing_map(link_ids, link_map)
    incoming_map = build_incoming_map(link_ids, outgoing_map)
    bundle_cache: dict[str, tuple[str, ...]] = {}
    rep_cache: dict[str, str] = {}

    def bundle_for(link_id: str) -> tuple[str, ...]:
        cached = bundle_cache.get(link_id)
        if cached is not None:
            return cached
        bundle = tuple(ordered_lateral_bundle(link_id, link_map, selected))
        for item in bundle:
            bundle_cache[item] = bundle
        return bundle

    def downstream_score(link_id: str) -> float:
        total = 0.0
        stack = [link_id]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            total += float(len(link_map[current].get("points", [])))
            for next_id in outgoing_map.get(current, []):
                if next_id not in visited:
                    stack.append(next_id)
        return total

    def representative_for(link_id: str) -> str:
        cached = rep_cache.get(link_id)
        if cached is not None:
            return cached
        bundle = list(bundle_for(link_id))
        representative = choose_bundle_representative(bundle, downstream_score)
        for item in bundle:
            rep_cache[item] = representative
        return representative

    start_links = [link_id for link_id in link_ids if not incoming_map.get(link_id)]
    if not start_links:
        start_links = [link_ids[0]]

    start_reps = []
    for link_id in start_links:
        rep = representative_for(link_id)
        if rep not in start_reps:
            start_reps.append(rep)
    current = max(start_reps, key=downstream_score)

    ordered: list[str] = []
    visited: set[str] = set()
    while current not in visited:
        ordered.append(current)
        visited.add(current)
        next_candidates: list[str] = []
        for next_link in outgoing_map.get(current, []):
            rep = representative_for(next_link)
            if rep != current and rep not in next_candidates:
                next_candidates.append(rep)
        if not next_candidates:
            break
        current = max(next_candidates, key=downstream_score)

    return ordered


def flatten_points(
    ordered_link_ids: list[str],
    link_map: dict[str, dict],
    partial_lookup: dict[str, tuple[float, float]],
) -> list[tuple[float, float]]:
    path_points: list[tuple[float, float]] = []

    for index, link_id in enumerate(ordered_link_ids):
        points = list(link_map[link_id].get("points", []))
        if link_id in partial_lookup:
            start_ratio, end_ratio = partial_lookup[link_id]
            points = apply_partial_points(points, start_ratio, end_ratio)
        if not points:
            continue

        xy = [(float(point[0]), float(point[1])) for point in points]
        if index > 0 and path_points and xy:
            if path_points[-1] == xy[0]:
                xy = xy[1:]
        path_points.extend(xy)

    deduped: list[tuple[float, float]] = []
    for point in path_points:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    return deduped


def compute_yaw_deg(points: list[tuple[float, float]], index: int) -> float:
    if len(points) < 2:
        return 0.0
    if index < len(points) - 1:
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
    else:
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def write_reference_csv(path: Path, points: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "yaw_deg"])
        for index, (x, y) in enumerate(points):
            writer.writerow([x, y, compute_yaw_deg(points, index)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reference_path CSV from selected MGeo links."
    )
    parser.add_argument(
        "--selected-links",
        default="output/selected_links.json",
        help="Path to selected link ids (.json or .txt).",
    )
    parser.add_argument(
        "--selection-key",
        default="selected_link_ids",
        help="JSON key to read from when --selected-links is a dict json.",
    )
    parser.add_argument(
        "--link-set",
        default="MORAI-ADModule/mgeo/lib/mgeo_data/R_KR_PG_K-City/link_set.json",
        help="Path to the MGeo link_set.json file.",
    )
    parser.add_argument(
        "--output-csv",
        default="morai_rl/data/reference_path_from_links.csv",
        help="Where to save the generated reference path CSV.",
    )
    parser.add_argument(
        "--keep-lane-change-links",
        action="store_true",
        help="Include link ids that contain '-' instead of filtering them out.",
    )
    parser.add_argument(
        "--centerline-only",
        action="store_true",
        help="Build a single representative lane sequence from selected parallel links.",
    )
    args = parser.parse_args()

    selected_path = Path(args.selected_links)
    link_set_path = Path(args.link_set)
    output_csv_path = Path(args.output_csv)

    selected_link_ids, partial_links = load_selected_link_ids(selected_path, selection_key=args.selection_key)
    if not selected_link_ids:
        raise ValueError(
            f"no link ids found in {selected_path} for selection key '{args.selection_key}'"
        )
    if not args.keep_lane_change_links:
        selected_link_ids = [
            link_id for link_id in selected_link_ids if not is_lane_change_link(link_id)
        ]

    link_map = load_link_map(link_set_path)
    ordered_link_ids = order_links_by_connectivity(selected_link_ids, link_map)
    if args.centerline_only:
        ordered_link_ids = build_centerline_link_sequence(ordered_link_ids, link_map)
    partial_lookup = build_partial_lookup(partial_links)
    points = flatten_points(ordered_link_ids, link_map, partial_lookup)

    if len(points) < 2:
        raise ValueError("reference path needs at least two points after link processing")

    write_reference_csv(output_csv_path, points)
    print(f"selected_links={len(selected_link_ids)}")
    print(f"ordered_links={len(ordered_link_ids)}")
    print(f"path_points={len(points)}")
    print(f"output_csv={output_csv_path.resolve()}")


if __name__ == "__main__":
    main()
