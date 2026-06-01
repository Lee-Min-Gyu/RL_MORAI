from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import UnityPy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Directly scrape Mesh/GameObject/Transform data from a Unity scene bundle."
    )
    parser.add_argument("bundle", type=Path, help="Path to the scene bundle.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scene_mesh_dump"),
        help="Directory where extracted scene mesh data will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Override output folder name. Defaults to the bundle stem.",
    )
    parser.add_argument(
        "--name-keyword",
        action="append",
        dest="name_keywords",
        default=[],
        help="Only keep GameObjects whose names contain one of these keywords. Can be repeated.",
    )
    return parser.parse_args()


def safe_name(obj: Any) -> str:
    return (
        getattr(obj, "m_Name", None)
        or getattr(obj, "name", None)
        or getattr(obj, "__class__", type(obj)).__name__
    )


def normalize_dataset_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "_")


def pptr_path_id(ptr: Any) -> int | None:
    return getattr(ptr, "m_PathID", None)


def pptr_file_id(ptr: Any) -> int | None:
    return getattr(ptr, "m_FileID", None)


def read_pptr(ptr: Any) -> Any | None:
    if ptr is None:
        return None
    try:
        return ptr.read()
    except Exception:
        return None


def vector3_to_dict(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    x = getattr(value, "x", None)
    y = getattr(value, "y", None)
    z = getattr(value, "z", None)
    if x is None and y is None and z is None:
        return None
    return {"x": float(x or 0.0), "y": float(y or 0.0), "z": float(z or 0.0)}


def quaternion_to_dict(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    x = getattr(value, "x", None)
    y = getattr(value, "y", None)
    z = getattr(value, "z", None)
    w = getattr(value, "w", None)
    if x is None and y is None and z is None and w is None:
        return None
    return {
        "x": float(x or 0.0),
        "y": float(y or 0.0),
        "z": float(z or 0.0),
        "w": float(w or 1.0),
    }


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_len__": len(value), "preview_hex": value[:128].hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: make_json_safe(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def matches_keywords(name: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def export_mesh_payload(mesh_obj: Any, mesh_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    name = safe_name(mesh_obj)
    payload["name"] = name

    export_func = getattr(mesh_obj, "export", None)
    if callable(export_func):
        try:
            exported = export_func()
        except Exception as exc:
            payload["export_error"] = str(exc)
        else:
            if isinstance(exported, str) and exported.strip():
                export_path = mesh_dir / f"{name or 'mesh'}.obj"
                export_path.write_text(exported, encoding="utf-8")
                payload["obj_path"] = str(export_path)
            elif isinstance(exported, bytes) and exported:
                export_path = mesh_dir / f"{name or 'mesh'}.bin"
                export_path.write_bytes(exported)
                payload["bin_path"] = str(export_path)

    for attr_name in (
        "m_SubMeshes",
        "m_IsReadable",
        "m_KeepVertices",
        "m_KeepIndices",
        "m_MeshCompression",
    ):
        if hasattr(mesh_obj, attr_name):
            payload[attr_name] = make_json_safe(getattr(mesh_obj, attr_name))

    return payload


def main() -> None:
    args = parse_args()
    bundle_path = args.bundle.resolve()
    dataset_name = args.dataset_name or bundle_path.stem
    output_root = args.out_dir.resolve() / normalize_dataset_name(dataset_name)
    mesh_dir = output_root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    env = UnityPy.load(str(bundle_path))

    transforms: dict[int, dict[str, Any]] = {}
    mesh_components: dict[int, dict[str, Any]] = {}
    collider_components: dict[int, dict[str, Any]] = {}
    mesh_payloads: dict[int, dict[str, Any]] = {}
    game_objects: list[dict[str, Any]] = []

    for obj in env.objects:
        type_name = obj.type.name

        if type_name == "Transform":
            try:
                data = obj.read()
            except Exception:
                continue
            transforms[obj.path_id] = {
                "path_id": obj.path_id,
                "game_object_path_id": pptr_path_id(getattr(data, "m_GameObject", None)),
                "local_position": vector3_to_dict(getattr(data, "m_LocalPosition", None)),
                "local_rotation": quaternion_to_dict(getattr(data, "m_LocalRotation", None)),
                "local_scale": vector3_to_dict(getattr(data, "m_LocalScale", None)),
                "father_path_id": pptr_path_id(getattr(data, "m_Father", None)),
                "children_path_ids": [
                    pptr_path_id(child) for child in getattr(data, "m_Children", []) if pptr_path_id(child)
                ],
            }
            continue

        if type_name in ("MeshFilter", "MeshCollider"):
            try:
                data = obj.read()
            except Exception:
                continue
            mesh_ref = getattr(data, "m_Mesh", None)
            mesh_data = read_pptr(mesh_ref)
            target = mesh_components if type_name == "MeshFilter" else collider_components
            target[obj.path_id] = {
                "path_id": obj.path_id,
                "game_object_path_id": pptr_path_id(getattr(data, "m_GameObject", None)),
                "mesh_path_id": pptr_path_id(mesh_ref),
                "mesh_file_id": pptr_file_id(mesh_ref),
                "mesh_name": safe_name(mesh_data) if mesh_data is not None else None,
                "component_type": type_name,
            }
            continue

        if type_name == "Mesh":
            try:
                data = obj.read()
            except Exception:
                continue

            mesh_payload = {
                "path_id": obj.path_id,
                "type": type_name,
                "name": safe_name(data),
            }
            mesh_payload.update(export_mesh_payload(data, mesh_dir))

            try:
                mesh_payload["typetree"] = make_json_safe(obj.read_typetree())
            except Exception as exc:
                mesh_payload["typetree_error"] = str(exc)

            mesh_payloads[obj.path_id] = mesh_payload

            mesh_json_path = mesh_dir / f"{obj.path_id}_{mesh_payload['name'] or 'mesh'}.json"
            mesh_json_path.write_text(
                json.dumps(mesh_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

    for obj in env.objects:
        if obj.type.name != "GameObject":
            continue
        try:
            data = obj.read()
        except Exception:
            continue

        name = safe_name(data)
        if not matches_keywords(name, args.name_keywords):
            continue

        component_refs = getattr(data, "m_Component", [])
        component_path_ids: list[int] = []
        mesh_filter_refs: list[dict[str, Any]] = []
        mesh_collider_refs: list[dict[str, Any]] = []
        transform_ref: dict[str, Any] | None = None

        for item in component_refs:
            component_ptr = getattr(item, "component", None) or item
            component = read_pptr(component_ptr)
            component_path_id = pptr_path_id(component_ptr)
            if component_path_id:
                component_path_ids.append(component_path_id)

            if component_path_id in mesh_components:
                mesh_filter_refs.append(mesh_components[component_path_id])
            elif component_path_id in collider_components:
                mesh_collider_refs.append(collider_components[component_path_id])
            elif component is not None and component.__class__.__name__ == "Transform":
                transform_ref = transforms.get(component_path_id)

        game_objects.append(
            {
                "path_id": obj.path_id,
                "name": name,
                "component_path_ids": component_path_ids,
                "transform": transform_ref,
                "mesh_filters": mesh_filter_refs,
                "mesh_colliders": mesh_collider_refs,
            }
        )

    summary = {
        "bundle": str(bundle_path),
        "dataset_name": dataset_name,
        "counts": {
            "game_objects": len(game_objects),
            "transforms": len(transforms),
            "mesh_filters": len(mesh_components),
            "mesh_colliders": len(collider_components),
            "meshes": len(mesh_payloads),
        },
    }

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "scene_graph.json").write_text(
        json.dumps(make_json_safe(game_objects), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "mesh_filters.json").write_text(
        json.dumps(make_json_safe(list(mesh_components.values())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "mesh_colliders.json").write_text(
        json.dumps(make_json_safe(list(collider_components.values())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Bundle: {bundle_path}")
    print(f"Output: {output_root}")
    print(json.dumps(summary["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
