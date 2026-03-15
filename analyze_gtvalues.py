#!/usr/bin/env python3

"""Compare SISIFOS gtValues.txt files using estimator-focused motion metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def _parse_gtvalues(path: Path) -> dict[str, list[str]]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    data: dict[str, list[str]] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line:
            idx += 1
            continue
        if line.endswith("="):
            key = line[:-1].strip()
            idx += 1
            values: list[str] = []
            while idx < len(lines) and not lines[idx].endswith("="):
                if lines[idx]:
                    values.append(lines[idx])
                idx += 1
            data[key] = values
        else:
            idx += 1
    return data


def _parse_vectors(lines: Iterable[str]) -> list[list[float]]:
    return [[float(x) for x in line.split()] for line in lines]


def _quat_to_rot(q: list[float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[c][r] for c in range(3)] for r in range(3)]


def _matvec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vec[col] for col in range(3)) for row in range(3)]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[idx]


def _resolve_gtvalues(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_dir():
        candidate = path / "gtValues.txt"
        if not candidate.exists():
            raise FileNotFoundError(f"No gtValues.txt found in directory: {path}")
        return candidate
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _discover_sweep_gtvalues(root: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for traj_json in sorted(root.glob("Config_*_trajectory.json")):
        config_prefix = traj_json.stem.removesuffix("_trajectory")
        payload = json.loads(traj_json.read_text())
        lookat_mode = str(payload.get("camera_lookat_mode", config_prefix))

        config_dirs = sorted(path for path in root.glob(f"{config_prefix}_*") if path.is_dir())
        if not config_dirs:
            continue

        for config_dir in config_dirs:
            agent_dirs = sorted(path for path in config_dir.glob("Agent_*") if path.is_dir())
            if not agent_dirs:
                continue
            multi_agent = len(agent_dirs) > 1
            for agent_dir in agent_dirs:
                gt_path = agent_dir / "gtValues.txt"
                if gt_path.exists():
                    label = lookat_mode if not multi_agent else f"{lookat_mode}:{agent_dir.name}"
                    entries.append((label, gt_path))

    return entries


def _expand_inputs(paths: list[str]) -> list[tuple[str, Path]]:
    expanded: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir() and not (path / "gtValues.txt").exists():
            discovered = _discover_sweep_gtvalues(path)
            if discovered:
                for label, gt_path in discovered:
                    if gt_path not in seen:
                        expanded.append((label, gt_path))
                        seen.add(gt_path)
                continue

        gt_path = _resolve_gtvalues(raw_path)
        if gt_path not in seen:
            default_label = gt_path.parent.name if gt_path.name == "gtValues.txt" else gt_path.stem
            expanded.append((default_label, gt_path))
            seen.add(gt_path)

    return expanded


def _subset_indices(timestamps: list[float], end_time_s: float) -> list[int]:
    return [idx for idx, ts in enumerate(timestamps) if abs(ts - round(ts)) < 1e-9 and 0.0 <= ts < end_time_s]


def _window_summary(
    q_gs: list[list[float]],
    omega_gi_g: list[list[float]],
    omega_si_s: list[list[float]],
    r_sg_g: list[list[float]],
    v_sg_g: list[list[float]],
    start_idx: int,
    end_idx: int,
) -> dict[str, float]:
    gyro_fraction: list[float] = []
    omega_gs_norms: list[float] = []
    ranges: list[float] = []
    speeds: list[float] = []
    radial_fractions: list[float] = []

    for idx in range(start_idx, end_idx):
        r_sg = r_sg_g[idx]
        v_sg = v_sg_g[idx]
        w_gi = omega_gi_g[idx]
        w_si = omega_si_s[idx]
        w_gi_s = _matvec(_transpose(_quat_to_rot(q_gs[idx])), w_gi)
        w_gs_s = [w_gi_s[k] - w_si[k] for k in range(3)]
        gyro_fraction.append(_norm(w_si) / (_norm(w_gi) + 1e-12))
        omega_gs_norms.append(_norm(w_gs_s))
        ranges.append(_norm(r_sg))
        speeds.append(_norm(v_sg))
        u = [x / (_norm(r_sg) + 1e-12) for x in r_sg]
        radial_fractions.append(abs(sum(u[k] * v_sg[k] for k in range(3))) / (_norm(v_sg) + 1e-12))

    return {
        "gyro_fraction_mean": _mean(gyro_fraction),
        "omega_GS_S_mean": _mean(omega_gs_norms),
        "range_mean": _mean(ranges),
        "vel_mean": _mean(speeds),
        "radial_frac_mean": _mean(radial_fractions),
    }


def analyze_gtvalues(path: Path, subset_end_time_s: float) -> dict[str, object]:
    gt = _parse_gtvalues(path)
    timestamps = [float(x) for x in gt["timestamps"]]
    q_gs = _parse_vectors(gt["q_GS"])
    omega_gi_g = _parse_vectors(gt["omega_GI_G"])
    omega_si_s = _parse_vectors(gt["omega_SI_S"])
    r_sg_g = _parse_vectors(gt["r_SG_G"])
    v_sg_g = _parse_vectors(gt["v_SG_G"])

    omega_gs_s: list[list[float]] = []
    gyro_fraction: list[float] = []
    vo_proxy_fraction: list[float] = []
    ranges: list[float] = []
    speeds: list[float] = []
    radial_fractions: list[float] = []

    for q, w_gi, w_si, r_sg, v_sg in zip(q_gs, omega_gi_g, omega_si_s, r_sg_g, v_sg_g):
        w_gi_s = _matvec(_transpose(_quat_to_rot(q)), w_gi)
        w_gs_s = [w_gi_s[k] - w_si[k] for k in range(3)]
        omega_gs_s.append(w_gs_s)

        norm_w_gi = _norm(w_gi) + 1e-12
        gyro_fraction.append(_norm(w_si) / norm_w_gi)
        vo_proxy_fraction.append(_norm(w_gs_s) / norm_w_gi)

        norm_r = _norm(r_sg)
        norm_v = _norm(v_sg)
        ranges.append(norm_r)
        speeds.append(norm_v)
        u = [x / (norm_r + 1e-12) for x in r_sg]
        radial_fractions.append(abs(sum(u[k] * v_sg[k] for k in range(3))) / (norm_v + 1e-12))

    out: dict[str, object] = {
        "path": str(path),
        "n": len(timestamps),
        "dt": timestamps[1] - timestamps[0] if len(timestamps) > 1 else None,
        "omega_GI_G_mean": _mean([_norm(x) for x in omega_gi_g]),
        "omega_GI_G_p95": _p95([_norm(x) for x in omega_gi_g]),
        "omega_SI_S_mean": _mean([_norm(x) for x in omega_si_s]),
        "omega_SI_S_p95": _p95([_norm(x) for x in omega_si_s]),
        "omega_GS_S_mean": _mean([_norm(x) for x in omega_gs_s]),
        "omega_GS_S_p95": _p95([_norm(x) for x in omega_gs_s]),
        "gyro_fraction_mean": _mean(gyro_fraction),
        "vo_proxy_fraction_mean": _mean(vo_proxy_fraction),
        "range_mean": _mean(ranges),
        "range_min": min(ranges),
        "range_max": max(ranges),
        "vel_mean": _mean(speeds),
        "vel_p95": _p95(speeds),
        "radial_frac_mean": _mean(radial_fractions),
        "radial_frac_p95": _p95(radial_fractions),
    }

    subset_idx = _subset_indices(timestamps, subset_end_time_s)
    if subset_idx:
        q_subset = [q_gs[idx] for idx in subset_idx]
        w_gi_subset = [omega_gi_g[idx] for idx in subset_idx]
        w_si_subset = [omega_si_s[idx] for idx in subset_idx]
        r_subset = [r_sg_g[idx] for idx in subset_idx]
        v_subset = [v_sg_g[idx] for idx in subset_idx]
        w_gs_subset: list[list[float]] = []
        gyro_subset: list[float] = []
        vo_subset: list[float] = []
        radial_subset: list[float] = []

        for q, w_gi, w_si, r_sg, v_sg in zip(q_subset, w_gi_subset, w_si_subset, r_subset, v_subset):
            w_gi_s = _matvec(_transpose(_quat_to_rot(q)), w_gi)
            w_gs_s = [w_gi_s[k] - w_si[k] for k in range(3)]
            w_gs_subset.append(w_gs_s)
            gyro_subset.append(_norm(w_si) / (_norm(w_gi) + 1e-12))
            vo_subset.append(_norm(w_gs_s) / (_norm(w_gi) + 1e-12))
            u = [x / (_norm(r_sg) + 1e-12) for x in r_sg]
            radial_subset.append(abs(sum(u[k] * v_sg[k] for k in range(3))) / (_norm(v_sg) + 1e-12))

        out["subset_0_10s_1Hz"] = {
            "n": len(subset_idx),
            "omega_GI_G_mean": _mean([_norm(x) for x in w_gi_subset]),
            "omega_SI_S_mean": _mean([_norm(x) for x in w_si_subset]),
            "omega_GS_S_mean": _mean([_norm(x) for x in w_gs_subset]),
            "gyro_fraction_mean": _mean(gyro_subset),
            "vo_proxy_fraction_mean": _mean(vo_subset),
            "range_mean": _mean([_norm(x) for x in r_subset]),
            "range_min": min(_norm(x) for x in r_subset),
            "range_max": max(_norm(x) for x in r_subset),
            "vel_mean": _mean([_norm(x) for x in v_subset]),
            "radial_frac_mean": _mean(radial_subset),
        }

    n_samples = len(timestamps)
    if n_samples >= 20:
        out["windows"] = {
            "first_10pct": _window_summary(q_gs, omega_gi_g, omega_si_s, r_sg_g, v_sg_g, 0, n_samples // 10),
            "middle_50pct": _window_summary(
                q_gs, omega_gi_g, omega_si_s, r_sg_g, v_sg_g, n_samples // 4, 3 * n_samples // 4
            ),
            "last_25pct": _window_summary(q_gs, omega_gi_g, omega_si_s, r_sg_g, v_sg_g, 3 * n_samples // 4, n_samples),
        }

    return out


def _print_table(results: dict[str, dict[str, object]]) -> None:
    headers = [
        "label",
        "n",
        "dt",
        "omega",
        "gyro",
        "vo_proxy",
        "range[min,max]",
        "vel",
        "radial",
    ]
    rows = []
    for label, metrics in results.items():
        rows.append(
            [
                label,
                str(metrics["n"]),
                f"{metrics['dt']:.3f}" if metrics["dt"] is not None else "n/a",
                f"{metrics['omega_GI_G_mean']:.4f}",
                f"{metrics['omega_SI_S_mean']:.4f}",
                f"{metrics['omega_GS_S_mean']:.4f}",
                f"{metrics['range_min']:.1f},{metrics['range_max']:.1f}",
                f"{metrics['vel_mean']:.4f}",
                f"{metrics['radial_frac_mean']:.3f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    fmt = "  ".join(f"{{:{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gtvalues",
        nargs="+",
        help="One or more gtValues.txt files, agent directories, or sweep root directories",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label override in the form label=path. Can be repeated.",
    )
    parser.add_argument(
        "--subset-end-time",
        type=float,
        default=10.0,
        help="End time in seconds for the 1 Hz early-window summary (default: 10.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output instead of the compact table",
    )
    args = parser.parse_args()

    label_overrides: dict[Path, str] = {}
    for item in args.label:
        if "=" not in item:
            raise ValueError(f"Invalid --label value: {item!r}. Expected label=path.")
        label, raw_path = item.split("=", 1)
        label_overrides[_resolve_gtvalues(raw_path)] = label

    results: dict[str, dict[str, object]] = {}
    for default_label, path in _expand_inputs(args.gtvalues):
        label = label_overrides.get(path, default_label)
        results[label] = analyze_gtvalues(path, subset_end_time_s=args.subset_end_time)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
