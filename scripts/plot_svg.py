#!/usr/bin/env python3
"""Render docs/fem_breakdown.svg from results/fem_400.csv and results/fem_800.csv."""
from __future__ import annotations

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def render(csv400: pathlib.Path, csv800: pathlib.Path | None, out_path: pathlib.Path) -> None:
    r400 = _load(csv400)
    r800 = _load(csv800) if csv800 and csv800.is_file() else r400

    w, h = 980, 390
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" rx="12" fill="#0d1117"/>',
        '<text x="32" y="36" fill="#f0f6fc" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700">'
        '2D P1 FEM Solver: L3-Resident (nx=400, 159k DOFs) vs DRAM-Bound (nx=800, 638k DOFs)</text>',
        '<text x="32" y="58" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="12">'
        'Intel i7-14650HX (WSL2, 6c/12t, gcc 16.2.0) · 8-color lock-free CSR assembly + Conjugate Gradient</text>',
    ]

    # Left panel: CSR SpMV Effective Bandwidth vs Threads (400 vs 800)
    x0, y0, pw, ph = 32, 84, 468, 276
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="8" fill="#161b22" stroke="#30363d"/>')
    lines.append(
        f'<text x="{x0 + 16}" y="{y0 + 26}" fill="#e6edf3" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" font-weight="600">CSR SpMV Effective Bandwidth (GB/s)</text>'
    )
    gx, gy, gw, gh = x0 + 56, y0 + 50, pw - 90, ph - 98
    max_y = 175.0
    xs = [1, 2, 4, 6, 12]
    x_pos = {t: gx + i * (gw / (len(xs) - 1)) for i, t in enumerate(xs)}

    def sy(val: float) -> float:
        return gy + gh - (val / max_y) * gh

    for tick in (0, 50, 100, 150):
        ty = sy(float(tick))
        lines.append(f'<line x1="{gx}" y1="{ty:.1f}" x2="{gx + gw}" y2="{ty:.1f}" stroke="#21262d"/>')
        lines.append(
            f'<text x="{gx - 8}" y="{ty + 4:.1f}" text-anchor="end" fill="#8b949e" '
            f'font-family="Consolas, monospace" font-size="11">{tick}</text>'
        )
    for t in xs:
        lines.append(
            f'<text x="{x_pos[t]:.1f}" y="{gy + gh + 18}" text-anchor="middle" fill="#8b949e" '
            f'font-family="Consolas, monospace" font-size="11">{t}t</text>'
        )

    # hpc-kernels axpy DRAM ceiling reference line (58.35 GB/s)
    y_ceil = sy(58.35)
    lines.append(
        f'<line x1="{gx}" y1="{y_ceil:.1f}" x2="{gx + gw}" y2="{y_ceil:.1f}" '
        f'stroke="#f85149" stroke-dasharray="5,5" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{gx + 4}" y="{y_ceil - 6:.1f}" text-anchor="start" fill="#f85149" '
        f'font-family="Consolas, monospace" font-size="10">axpy DRAM wall (58.4 GB/s)</text>'
    )

    for lbl, rows_src, col, dy in [
        ("nx=400 (~16 MB, in L3)", r400, "#56d364", -10),
        ("nx=800 (~64 MB, DRAM)", r800, "#58a6ff", 16),
    ]:
        pts = [(int(r["threads"]), float(r["spmv_gbps"])) for r in rows_src if r["mode"] == "omp_colored"]
        poly = " ".join(f"{x_pos[t]:.1f},{sy(v):.1f}" for t, v in pts)
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.5" points="{poly}"/>')
        for t, v in pts:
            lines.append(f'<circle cx="{x_pos[t]:.1f}" cy="{sy(v):.1f}" r="4" fill="{col}"/>')
        lt, lv = pts[-1]
        lines.append(
            f'<text x="{x_pos[lt] - 6:.1f}" y="{sy(lv) + dy:.1f}" text-anchor="end" fill="{col}" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="700">{lbl}: {lv:.1f} GB/s</text>'
        )

    # Right panel: Stage Breakdown at nx=800 (Assembly vs SpMV vs Vector Ops)
    rx0, ry0, rpw, rph = 520, 84, 428, 276
    lines.append(f'<rect x="{rx0}" y="{ry0}" width="{rpw}" height="{rph}" rx="8" fill="#161b22" stroke="#30363d"/>')
    lines.append(
        f'<text x="{rx0 + 16}" y="{ry0 + 26}" fill="#e6edf3" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" font-weight="600">nx=800 Stage Breakdown (seconds)</text>'
    )
    for lx, col, lbl in [
        (rx0 + 16, "#f0883e", "Assembly"),
        (rx0 + 104, "#58a6ff", "CSR SpMV"),
        (rx0 + 198, "#a371f7", "CG Vector/Dot"),
    ]:
        lines.append(f'<rect x="{lx}" y="{ry0 + 38}" width="10" height="10" rx="2" fill="{col}"/>')
        lines.append(f'<text x="{lx + 15}" y="{ry0 + 47}" fill="#c9d1d9" font-family="Segoe UI, Arial, sans-serif" font-size="11">{lbl}</text>')

    picked = [r for r in r800 if (r["mode"] == "serial") or (r["mode"] == "omp_colored" and int(r["threads"]) in (2, 4, 6, 12))]
    base_tot = float(picked[0]["total_s"])
    bx0, bw_max = rx0 + 118, rpw - 224
    for idx, r in enumerate(picked):
        by = ry0 + 70 + idx * 38
        lbl = "serial (1t)" if r["mode"] == "serial" else f"colored ({r['threads']}t)"
        tot = float(r["total_s"])
        sp = base_tot / tot
        lines.append(
            f'<text x="{bx0 - 8}" y="{by + 13}" text-anchor="end" fill="#c9d1d9" '
            f'font-family="Consolas, monospace" font-size="11">{lbl}</text>'
        )
        cx = bx0
        for key, col in [("asm_s", "#f0883e"), ("spmv_s", "#58a6ff"), ("vec_s", "#a371f7")]:
            sw = (float(r[key]) / base_tot) * bw_max
            if sw > 0.6:
                lines.append(f'<rect x="{cx:.1f}" y="{by + 2}" width="{sw:.1f}" height="15" fill="{col}"/>')
                cx += sw
        lines.append(
            f'<text x="{cx + 6:.1f}" y="{by + 13}" fill="#e6edf3" font-family="Consolas, monospace" '
            f'font-size="11">{tot:.2f}s ({sp:.2f}×)</text>'
        )

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    c400 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "fem_400.csv"
    c800 = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "fem_800.csv"
    svg_out = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "docs" / "fem_breakdown.svg"
    render(c400, c800, svg_out)
