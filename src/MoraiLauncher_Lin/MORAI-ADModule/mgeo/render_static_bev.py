#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


DEFAULT_LINK_WIDTH_M = 3.8
DEFAULT_LANE_MARKING_WIDTH_M = 0.15


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_selected_link_ids(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"selected link file not found: {path}")

    if path.lower().endswith(".json"):
        data = load_json(path)
        if isinstance(data, list):
            return {str(item).strip() for item in data if str(item).strip()}
        if isinstance(data, dict):
            selected = data.get("selected_link_ids", [])
            return {str(item).strip() for item in selected if str(item).strip()}
        raise ValueError("selected link json must be a list or contain selected_link_ids")

    with open(path, "r", encoding="utf-8") as file:
        raw = file.read()

    tokens = []
    for line in raw.splitlines():
        tokens.extend(line.replace(",", " ").split())

    return {token.strip() for token in tokens if token.strip()}


def load_selection_config(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"selected link file not found: {path}")

    if path.lower().endswith(".json"):
        data = load_json(path)
        if isinstance(data, list):
            return {
                "selected_link_ids": {str(item).strip() for item in data if str(item).strip()},
                "fill_groups": [],
                "partial_links": {},
            }
        if isinstance(data, dict):
            return {
                "selected_link_ids": {str(item).strip() for item in data.get("selected_link_ids", []) if str(item).strip()},
                "fill_groups": [
                    {
                        "name": str(group.get("name") or f"group_{index + 1:02d}"),
                        "link_ids": {str(item).strip() for item in group.get("link_ids", []) if str(item).strip()},
                    }
                    for index, group in enumerate(data.get("fill_groups", []))
                ],
                "partial_links": {
                    str(item.get("link_id")).strip(): {
                        "start_ratio": float(item.get("start_ratio", 0.0)),
                        "end_ratio": float(item.get("end_ratio", 1.0)),
                    }
                    for item in data.get("partial_links", [])
                    if str(item.get("link_id", "")).strip()
                },
            }
        raise ValueError("selected link json must be a list or contain selected_link_ids/fill_groups/partial_links")

    return {
        "selected_link_ids": load_selected_link_ids(path),
        "fill_groups": [],
        "partial_links": {},
    }


def points_from_item(item):
    if "points" in item and item["points"]:
        return item["points"]
    if "point" in item and item["point"]:
        return [item["point"]]
    return []


def update_bounds(bounds, points):
    for point in points:
        x = float(point[0])
        y = float(point[1])
        bounds["min_x"] = min(bounds["min_x"], x)
        bounds["min_y"] = min(bounds["min_y"], y)
        bounds["max_x"] = max(bounds["max_x"], x)
        bounds["max_y"] = max(bounds["max_y"], y)


def compute_bounds(feature_sets, margin_m):
    bounds = {
        "min_x": float("inf"),
        "min_y": float("inf"),
        "max_x": float("-inf"),
        "max_y": float("-inf"),
    }

    for items in feature_sets:
        for item in items:
            update_bounds(bounds, points_from_item(item))

    if not math.isfinite(bounds["min_x"]):
        raise ValueError("Could not compute bounds from the provided MGeo features.")

    bounds["min_x"] -= margin_m
    bounds["min_y"] -= margin_m
    bounds["max_x"] += margin_m
    bounds["max_y"] += margin_m
    return bounds


def world_to_pixel_float(x, y, bounds, resolution):
    px = (float(x) - bounds["min_x"]) / resolution
    py = (bounds["max_y"] - float(y)) / resolution
    return px, py


def meters_to_pixels(length_m, resolution, minimum=1):
    return max(minimum, int(round(float(length_m) / resolution)))


def create_empty_layer(image_size):
    return Image.new("L", image_size, color=0)


def build_preview(bev_layers, channel_names):
    height = bev_layers.shape[1]
    width = bev_layers.shape[2]
    preview = np.zeros((height, width, 3), dtype=np.uint8)

    if "drivable_area" in channel_names:
        drivable_index = channel_names.index("drivable_area")
        drivable = bev_layers[drivable_index] > 0
        preview[drivable] = np.array([55, 55, 55], dtype=np.uint8)

    if "lane_marking" in channel_names:
        lane_index = channel_names.index("lane_marking")
        lane_marking = bev_layers[lane_index] > 0
        preview[lane_marking] = np.array([245, 215, 90], dtype=np.uint8)

    return preview


def distance_2d(p0, p1):
    return math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))


def resample_polyline(points, spacing_m=0.25):
    """
    polyline을 일정 간격으로 재샘플링한다.
    z는 선형 보간한다.
    """
    if len(points) < 2:
        return points

    pts = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in points]

    cumulative = [0.0]
    for i in range(1, len(pts)):
        seg_len = distance_2d(pts[i - 1], pts[i])
        cumulative.append(cumulative[-1] + seg_len)

    total_length = cumulative[-1]
    if total_length <= 1e-6:
        return [list(pts[0]), list(pts[-1])]

    sample_distances = np.arange(0.0, total_length, spacing_m).tolist()
    if sample_distances[-1] < total_length:
        sample_distances.append(total_length)

    result = []
    seg_idx = 0

    for d in sample_distances:
        while seg_idx < len(cumulative) - 2 and cumulative[seg_idx + 1] < d:
            seg_idx += 1

        d0 = cumulative[seg_idx]
        d1 = cumulative[seg_idx + 1]
        p0 = pts[seg_idx]
        p1 = pts[seg_idx + 1]

        if d1 - d0 < 1e-9:
            t = 0.0
        else:
            t = (d - d0) / (d1 - d0)

        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        z = p0[2] + t * (p1[2] - p0[2])
        result.append([x, y, z])

    return result


def interpolate_point(p0, p1, t):
    z0 = float(p0[2]) if len(p0) > 2 else 0.0
    z1 = float(p1[2]) if len(p1) > 2 else 0.0
    return [
        float(p0[0]) + t * (float(p1[0]) - float(p0[0])),
        float(p0[1]) + t * (float(p1[1]) - float(p0[1])),
        z0 + t * (z1 - z0),
    ]


def trim_polyline_by_ratio(points, start_ratio, end_ratio):
    if len(points) < 2:
        return points

    start_ratio = max(0.0, min(1.0, float(start_ratio)))
    end_ratio = max(0.0, min(1.0, float(end_ratio)))
    if end_ratio <= start_ratio:
        return []
    if start_ratio <= 0.0 and end_ratio >= 1.0:
        return [list(point) for point in points]

    cumulative = [0.0]
    total_length = 0.0
    for index in range(1, len(points)):
        seg_len = distance_2d(points[index - 1], points[index])
        total_length += seg_len
        cumulative.append(total_length)

    if total_length <= 1e-9:
        return []

    start_distance = start_ratio * total_length
    end_distance = end_ratio * total_length
    trimmed = []

    for index in range(1, len(points)):
        seg_start = cumulative[index - 1]
        seg_end = cumulative[index]
        if seg_end <= start_distance or seg_start >= end_distance:
            continue

        p0 = points[index - 1]
        p1 = points[index]
        seg_len = seg_end - seg_start
        if seg_len <= 1e-9:
            continue

        local_start = max(start_distance, seg_start)
        local_end = min(end_distance, seg_end)
        t0 = (local_start - seg_start) / seg_len
        t1 = (local_end - seg_start) / seg_len
        start_point = interpolate_point(p0, p1, t0)
        end_point = interpolate_point(p0, p1, t1)

        if not trimmed or distance_2d(trimmed[-1], start_point) > 1e-9:
            trimmed.append(start_point)
        if distance_2d(trimmed[-1], end_point) > 1e-9:
            trimmed.append(end_point)

    return trimmed


def make_linestring(points):
    """
    shapely는 2D만 사용하므로 x, y만 넣는다.
    """
    if len(points) < 2:
        return None

    coords = [(float(p[0]), float(p[1])) for p in points]
    if len(set(coords)) < 2:
        return None

    return LineString(coords)


def safe_buffer_line(line, half_width_m):
    """
    centerline을 실제 polygon처럼 확장한다.
    cap_style=2(flat), join_style=1(round)
    """
    if line is None or line.is_empty:
        return None

    if half_width_m <= 0.0:
        return None

    poly = line.buffer(
        half_width_m,
        cap_style=2,
        join_style=1,
        resolution=16,
    )

    if poly.is_empty:
        return None

    poly = poly.buffer(0)
    if poly.is_empty:
        return None

    return poly


def extract_link_width_m(link):
    width_start = float(link.get("width_start") or 0.0)
    width_end = float(link.get("width_end") or 0.0)
    width = max(width_start, width_end, DEFAULT_LINK_WIDTH_M)
    return width


def extract_lane_marking_width_m(lane_marking):
    return float(lane_marking.get("lane_width") or DEFAULT_LANE_MARKING_WIDTH_M)


def collect_link_polygons(link_set, resample_spacing_m=0.25, width_margin_m=0.35):
    polygons = []

    for link in link_set:
        raw_points = link.get("points", [])
        if len(raw_points) < 2:
            continue

        sampled = resample_polyline(raw_points, spacing_m=resample_spacing_m)
        line = make_linestring(sampled)
        if line is None:
            continue

        width_m = extract_link_width_m(link) + 2.0 * width_margin_m
        poly = safe_buffer_line(line, half_width_m=width_m / 2.0)
        if poly is not None:
            polygons.append(poly)

    return polygons


def collect_lane_marking_polygons(lane_marking_set, resample_spacing_m=0.15, extra_width_m=0.0):
    polygons = []

    for lane_marking in lane_marking_set:
        raw_points = lane_marking.get("points", [])
        if len(raw_points) < 2:
            continue

        sampled = resample_polyline(raw_points, spacing_m=resample_spacing_m)
        line = make_linestring(sampled)
        if line is None:
            continue

        width_m = extract_lane_marking_width_m(lane_marking) + extra_width_m
        poly = safe_buffer_line(line, half_width_m=width_m / 2.0)
        if poly is not None:
            polygons.append(poly)

    return polygons


def filter_link_set(link_set, selected_link_ids):
    if not selected_link_ids:
        return link_set
    return [link for link in link_set if link.get("idx") in selected_link_ids]


def apply_partial_links(link_set, partial_links):
    if not partial_links:
        return [dict(link) for link in link_set]

    effective_links = []
    for link in link_set:
        link_copy = dict(link)
        link_id = link.get("idx")
        spec = partial_links.get(link_id)
        if spec is not None:
            trimmed_points = trim_polyline_by_ratio(
                link.get("points", []),
                spec.get("start_ratio", 0.0),
                spec.get("end_ratio", 1.0),
            )
            if len(trimmed_points) < 2:
                continue
            link_copy["points"] = trimmed_points
        effective_links.append(link_copy)
    return effective_links


def collect_related_lane_marking_ids(link_set):
    lane_marking_ids = set()
    for link in link_set:
        lane_marking_ids.update(link.get("lane_mark_left", []) or [])
        lane_marking_ids.update(link.get("lane_mark_right", []) or [])
    return lane_marking_ids


def filter_lane_marking_set(lane_marking_set, lane_marking_ids):
    if not lane_marking_ids:
        return []
    return [lane_marking for lane_marking in lane_marking_set if lane_marking.get("idx") in lane_marking_ids]


def geometric_closing(geom, bridge_distance_m):
    if geom is None or geom.is_empty:
        return geom
    if bridge_distance_m <= 0.0:
        return geom

    closed = geom.buffer(bridge_distance_m, join_style=1).buffer(-bridge_distance_m, join_style=1)
    if closed.is_empty:
        return geom
    return closed.buffer(0)


def build_fill_group_geometries(
    fill_groups,
    effective_link_lookup,
    selected_link_ids,
    link_resample_spacing_m,
    link_width_margin_m,
    fill_group_bridge_distance_m,
):
    geometries = []

    for group in fill_groups:
        group_link_ids = set(group.get("link_ids", set())) & set(selected_link_ids)
        if not group_link_ids:
            continue

        group_links = [
            effective_link_lookup[link_id]
            for link_id in sorted(group_link_ids)
            if link_id in effective_link_lookup
        ]
        if not group_links:
            continue

        group_polygons = collect_link_polygons(
            link_set=group_links,
            resample_spacing_m=link_resample_spacing_m,
            width_margin_m=link_width_margin_m,
        )
        if not group_polygons:
            continue

        group_union = unary_union(group_polygons).buffer(0)
        group_filled = geometric_closing(group_union, fill_group_bridge_distance_m)
        if group_filled is not None and not group_filled.is_empty:
            geometries.append(
                {
                    "name": group.get("name"),
                    "link_ids": sorted(group_link_ids),
                    "geometry": group_filled,
                }
            )

    return geometries


def flatten_polygons(geom):
    """
    Polygon/MultiPolygon/GeometryCollection을 Polygon 리스트로 평탄화한다.
    """
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]

    if isinstance(geom, GeometryCollection):
        result = []
        for g in geom.geoms:
            result.extend(flatten_polygons(g))
        return result

    return []


def rasterize_polygon_to_layer(layer, polygon, bounds, resolution):
    if polygon is None or polygon.is_empty:
        return

    draw = ImageDraw.Draw(layer)

    exterior = [
        world_to_pixel_float(x, y, bounds, resolution)
        for x, y in polygon.exterior.coords
    ]
    exterior = [(round(x), round(y)) for x, y in exterior]
    if len(exterior) >= 3:
        draw.polygon(exterior, fill=255)

    for interior in polygon.interiors:
        hole = [
            world_to_pixel_float(x, y, bounds, resolution)
            for x, y in interior.coords
        ]
        hole = [(round(x), round(y)) for x, y in hole]
        if len(hole) >= 3:
            draw.polygon(hole, fill=0)


def rasterize_geometry(layer, geom, bounds, resolution):
    for poly in flatten_polygons(geom):
        rasterize_polygon_to_layer(layer, poly, bounds, resolution)


def morphological_closing_pil(layer, kernel_px=3, iterations=1):
    """
    Pillow의 MaxFilter -> MinFilter로 간단한 closing 수행
    """
    if kernel_px < 3:
        return layer

    if kernel_px % 2 == 0:
        kernel_px += 1

    result = layer
    for _ in range(iterations):
        result = result.filter(ImageFilter.MaxFilter(size=kernel_px))
        result = result.filter(ImageFilter.MinFilter(size=kernel_px))
    return result


def downsample_binary_layer(layer, out_size):
    """
    supersampled layer를 부드럽게 줄인 뒤 binary로 threshold.
    """
    resized = layer.resize(out_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.uint8)
    arr = (arr >= 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def render_static_bev(
    mgeo_dir,
    output_dir,
    resolution,
    margin_m,
    selected_link_ids=None,
    fill_groups=None,
    partial_links=None,
    include_lane_marking=True,
    supersample=4,
    link_resample_spacing_m=0.25,
    lane_resample_spacing_m=0.15,
    link_width_margin_m=0.35,
    fill_group_bridge_distance_m=2.0,
    drivable_closing_kernel_px=3,
    drivable_closing_iterations=1,
):
    global_info = load_json(os.path.join(mgeo_dir, "global_info.json"))
    link_set_all = load_json(os.path.join(mgeo_dir, "link_set.json"))
    lane_marking_set_all = []
    if include_lane_marking:
        lane_marking_set_all = load_json(os.path.join(mgeo_dir, "lane_marking_set.json"))

    link_set = filter_link_set(link_set_all, selected_link_ids)
    link_set = apply_partial_links(link_set, partial_links or {})
    if not link_set:
        raise ValueError("No links matched the selected link ids.")
    effective_link_lookup = {link.get("idx"): link for link in link_set}

    lane_marking_set = []
    if include_lane_marking:
        lane_marking_ids = collect_related_lane_marking_ids(link_set)
        lane_marking_set = filter_lane_marking_set(lane_marking_set_all, lane_marking_ids)

    bounds = compute_bounds(
        [
            link_set,
            lane_marking_set,
        ],
        margin_m,
    )

    if supersample < 1:
        supersample = 1

    render_resolution = resolution / supersample

    width_px_ss = int(math.ceil((bounds["max_x"] - bounds["min_x"]) / render_resolution)) + 1
    height_px_ss = int(math.ceil((bounds["max_y"] - bounds["min_y"]) / render_resolution)) + 1
    image_size_ss = (width_px_ss, height_px_ss)

    width_px = int(math.ceil((bounds["max_x"] - bounds["min_x"]) / resolution)) + 1
    height_px = int(math.ceil((bounds["max_y"] - bounds["min_y"]) / resolution)) + 1
    image_size = (width_px, height_px)

    layers_ss = {
        "drivable_area": create_empty_layer(image_size_ss),
    }
    if include_lane_marking:
        layers_ss["lane_marking"] = create_empty_layer(image_size_ss)

    # 1) 링크를 polygon으로 생성 후 union
    link_polygons = collect_link_polygons(
        link_set=link_set,
        resample_spacing_m=link_resample_spacing_m,
        width_margin_m=link_width_margin_m,
    )
    drivable_union = unary_union(link_polygons).buffer(0) if link_polygons else None

    fill_group_geometries = build_fill_group_geometries(
        fill_groups=fill_groups or [],
        effective_link_lookup=effective_link_lookup,
        selected_link_ids=selected_link_ids or set(),
        link_resample_spacing_m=link_resample_spacing_m,
        link_width_margin_m=link_width_margin_m,
        fill_group_bridge_distance_m=fill_group_bridge_distance_m,
    )
    if fill_group_geometries:
        combined_geometries = [drivable_union] if drivable_union is not None and not drivable_union.is_empty else []
        combined_geometries.extend(item["geometry"] for item in fill_group_geometries)
        drivable_union = unary_union(combined_geometries).buffer(0)

    # 2) lane marking도 polygon으로 생성 후 union
    lane_union = None
    if include_lane_marking:
        lane_polygons = collect_lane_marking_polygons(
            lane_marking_set=lane_marking_set,
            resample_spacing_m=lane_resample_spacing_m,
            extra_width_m=0.0,
        )
        lane_union = unary_union(lane_polygons).buffer(0) if lane_polygons else None

    # 3) supersampled rasterization
    if drivable_union is not None and not drivable_union.is_empty:
        rasterize_geometry(
            layers_ss["drivable_area"],
            drivable_union,
            bounds,
            render_resolution,
        )

    if include_lane_marking and lane_union is not None and not lane_union.is_empty:
        rasterize_geometry(
            layers_ss["lane_marking"],
            lane_union,
            bounds,
            render_resolution,
        )

    # 4) drivable closing
    if drivable_closing_kernel_px >= 3 and drivable_closing_iterations > 0:
        layers_ss["drivable_area"] = morphological_closing_pil(
            layers_ss["drivable_area"],
            kernel_px=drivable_closing_kernel_px,
            iterations=drivable_closing_iterations,
        )

    # 5) downsample
    layers = {
        name: downsample_binary_layer(layer, image_size)
        for name, layer in layers_ss.items()
    }

    channel_names = list(layers.keys())
    bev_layers = np.stack(
        [
            (np.array(layers[name], dtype=np.uint8) > 0).astype(np.uint8)
            for name in channel_names
        ],
        axis=0,
    )

    preview = build_preview(bev_layers, channel_names)

    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "static_bev.npz"),
        bev=bev_layers,
        channel_names=np.array(channel_names),
    )
    Image.fromarray(preview).save(os.path.join(output_dir, "static_bev_preview.png"))

    metadata = {
        "map_name": os.path.basename(os.path.normpath(mgeo_dir)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": {
            "width_px": width_px,
            "height_px": height_px,
        },
        "supersampled_image_size": {
            "width_px": width_px_ss,
            "height_px": height_px_ss,
        },
        "resolution_m_per_px": resolution,
        "render_resolution_m_per_px": render_resolution,
        "supersample_factor": supersample,
        "margin_m": margin_m,
        "channels": channel_names,
        "pixel_transform": {
            "description": "col = (x - min_x) / resolution, row = (max_y - y) / resolution",
            "min_x": bounds["min_x"],
            "max_y": bounds["max_y"],
        },
        "world_bounds": bounds,
        "global_coordinate_system": global_info.get("global_coordinate_system"),
        "local_origin_in_global": global_info.get("local_origin_in_global"),
        "selected_link_count": len(link_set),
        "selected_lane_marking_count": len(lane_marking_set),
        "selected_link_ids": [link.get("idx") for link in link_set],
        "partial_link_count": len(partial_links or {}),
        "partial_links_applied": [
            {
                "link_id": link_id,
                "start_ratio": float(spec["start_ratio"]),
                "end_ratio": float(spec["end_ratio"]),
            }
            for link_id, spec in sorted((partial_links or {}).items())
            if link_id in effective_link_lookup
        ],
        "fill_group_count": len(fill_groups or []),
        "fill_groups_applied": [
            {
                "name": item["name"],
                "link_count": len(item["link_ids"]),
                "link_ids": item["link_ids"],
            }
            for item in fill_group_geometries
        ],
        "rendering_options": {
            "include_lane_marking": include_lane_marking,
            "link_resample_spacing_m": link_resample_spacing_m,
            "lane_resample_spacing_m": lane_resample_spacing_m,
            "link_width_margin_m": link_width_margin_m,
            "fill_group_bridge_distance_m": fill_group_bridge_distance_m,
            "drivable_closing_kernel_px": drivable_closing_kernel_px,
            "drivable_closing_iterations": drivable_closing_iterations,
        },
    }

    with open(os.path.join(output_dir, "static_bev_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a static semantic BEV raster from an MGeo map folder."
    )
    parser.add_argument(
        "--mgeo-dir",
        required=True,
        help="Path to the MGeo folder that contains global_info.json and related files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where static_bev.npz, static_bev_preview.png, and metadata will be saved.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.2,
        help="Final output meters per pixel.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=20.0,
        help="Padding margin in meters added around the map bounds.",
    )
    parser.add_argument(
        "--selected-link-file",
        help="Optional txt/json file containing selected link ids. If omitted, all links are used.",
    )
    parser.add_argument(
        "--exclude-lane-marking",
        action="store_true",
        help="Render only drivable_area and omit the lane_marking channel.",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=4,
        help="Internal supersampling factor before downsampling.",
    )
    parser.add_argument(
        "--link-resample-spacing",
        type=float,
        default=0.25,
        help="Resampling interval in meters for link centerlines.",
    )
    parser.add_argument(
        "--lane-resample-spacing",
        type=float,
        default=0.35,
        help="Resampling interval in meters for lane marking polylines.",
    )
    parser.add_argument(
        "--link-width-margin",
        type=float,
        default=0.15,
        help="Extra width margin added to both sides of link polygons.",
    )
    parser.add_argument(
        "--fill-group-bridge-distance",
        type=float,
        default=2.0,
        help="Bridge distance in meters used to selectively close gaps inside fill groups.",
    )
    parser.add_argument(
        "--drivable-closing-kernel",
        type=int,
        default=3,
        help="Kernel size in pixels for morphological closing on supersampled drivable layer.",
    )
    parser.add_argument(
        "--drivable-closing-iterations",
        type=int,
        default=1,
        help="Number of morphological closing iterations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_link_ids = None
    fill_groups = []
    partial_links = {}
    if args.selected_link_file:
        selection_config = load_selection_config(args.selected_link_file)
        selected_link_ids = selection_config["selected_link_ids"]
        fill_groups = selection_config["fill_groups"]
        partial_links = selection_config["partial_links"]

    metadata = render_static_bev(
        mgeo_dir=os.path.abspath(args.mgeo_dir),
        output_dir=os.path.abspath(args.output_dir),
        resolution=args.resolution,
        margin_m=args.margin,
        selected_link_ids=selected_link_ids,
        fill_groups=fill_groups,
        partial_links=partial_links,
        include_lane_marking=not args.exclude_lane_marking,
        supersample=args.supersample,
        link_resample_spacing_m=args.link_resample_spacing,
        lane_resample_spacing_m=args.lane_resample_spacing,
        link_width_margin_m=args.link_width_margin,
        fill_group_bridge_distance_m=args.fill_group_bridge_distance,
        drivable_closing_kernel_px=args.drivable_closing_kernel,
        drivable_closing_iterations=args.drivable_closing_iterations,
    )

    print("Saved static BEV to:", os.path.abspath(args.output_dir))
    print("Image size:", metadata["image_size"])
    print("Channels:", ", ".join(metadata["channels"]))
    print("Selected links:", metadata["selected_link_count"])


if __name__ == "__main__":
    main()
