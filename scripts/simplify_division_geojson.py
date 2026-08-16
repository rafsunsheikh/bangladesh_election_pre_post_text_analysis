#!/usr/bin/env python3
"""Shrink the division-boundary GeoJSON used as the map basemap.

The source file carries 463 administrative sub-features at full survey precision
(7.3 MB), but the maps only read `NAME_1` to colour and label 6 division
outlines at zoom 7. Dissolving by division and simplifying the geometry cuts the
file by ~96% with no visible difference, which is what makes a single-file
standalone dashboard practical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def round_coords(obj, ndigits: int):
    if isinstance(obj, (int, float)):
        return round(obj, ndigits)
    if isinstance(obj, (list, tuple)):
        return [round_coords(x, ndigits) for x in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/in_use/bangladesh_map_json_files/bangladesh.geojson")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/in_use/bangladesh_map_json_files/bangladesh_divisions_simplified.geojson"),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.002,
        help="Douglas-Peucker tolerance in degrees (0.002 is roughly 220 m).",
    )
    parser.add_argument("--precision", type=int, default=4, help="Decimal places kept per coordinate.")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("type") != "FeatureCollection":
        raise ValueError(f"{args.input} is not a GeoJSON FeatureCollection.")

    by_division: dict[str, list] = {}
    for feature in source["features"]:
        name = feature.get("properties", {}).get("NAME_1")
        if name is None:
            continue
        by_division.setdefault(name, []).append(shape(feature["geometry"]))

    features = []
    for name in sorted(by_division):
        merged = unary_union(by_division[name])
        if args.tolerance > 0:
            merged = merged.simplify(args.tolerance, preserve_topology=True)
        geom = round_coords(mapping(merged), args.precision)
        features.append({"type": "Feature", "properties": {"NAME_1": name}, "geometry": geom})

    out = {"type": "FeatureCollection", "features": features}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    before = args.input.stat().st_size
    after = args.output.stat().st_size
    print(f"Saved: {args.output}")
    print(f"  divisions: {len(features)} ({', '.join(sorted(by_division))})")
    print(f"  {before/1024/1024:.2f} MB -> {after/1024:.1f} KB ({100*(1-after/before):.1f}% smaller)")


if __name__ == "__main__":
    main()
