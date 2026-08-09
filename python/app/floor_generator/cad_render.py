"""
CAD-style architectural floor plan renderer.

Takes the dict returned by python/app/ml/predict.py's generate_floor_plan()
and renders it as a black-and-white architectural line drawing instead of
flat colored rectangles: a merged wall network (thick exterior / thin
interior, shared walls drawn exactly once), door openings with swing-arc
symbols at each rule-engine adjacency, room labels, per-room dimension
strings, overall building dimension lines, and simple furniture/fixture
glyphs per room type.

Wall/door/dimension engine adapted from an initial prototype; furniture
icons adapted from a second prototype built in parallel with a different
emphasis - see the project's commit history for both originals.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
SOLVER_DIR = ROOT / "python" / "app" / "training"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from geometry._layout._solver import _build_attach_map  # noqa: E402

# --------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------

EXT_WALL_T = 0.20    # exterior wall thickness (m)
INT_WALL_T = 0.10    # interior wall thickness (m)
DOOR_W = 0.90        # standard door width (m)
WALL_SNAP = 0.08     # tolerance for clustering coincident wall lines (m)
MERGE_TOL = 0.06     # tolerance for merging touching intervals along a wall line (m)
CORNER_CLEAR = 0.28  # keep interior doors this far from a wall corner/junction (m)
EXT_CORNER_CLEAR = 0.5  # keep the front door this far from a building corner (m)

TYPE_LABELS = {
    0: "BEDROOM",
    1: "BATHROOM",
    2: "KITCHEN",
    3: "LIVING ROOM",
    4: "BALCONY",
    5: "STORAGE",
    7: "ENTRY",
}

FURN_FILL = "#f4f4f4"
FURN_COLOR = "#555555"
M_TO_FT = 3.2808399


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------

def merge_intervals(intervals, tol):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(iv) for iv in merged]


def cluster_coords(values, tol):
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(sum(c) / len(c), c) for c in clusters]


def feet_inches(m):
    total_in = m * M_TO_FT * 12.0
    feet = int(total_in // 12)
    inches = round(total_in - feet * 12)
    if inches >= 12:
        inches -= 12
        feet += 1
    return f"{feet}'-{inches}\""


def fit_fontsize(text, box_w_m, box_h_m, pts_per_m, max_fs, min_fs,
                  frac_w=0.86, char_w_factor=0.58, height_frac=1.0):
    if not text:
        return min_fs
    fs_w = (box_w_m * frac_w * pts_per_m) / max(1, len(text)) / char_w_factor
    fs_h = box_h_m * height_frac * pts_per_m
    return max(min_fs, min(max_fs, fs_w, fs_h))


# --------------------------------------------------------------------------
# Load real-world room data from a generate_floor_plan() result
# --------------------------------------------------------------------------

def _real_rooms_from_result(result, use_raw=False):
    """Convert a generate_floor_plan() result dict into real-world-meter
    room dicts plus the rule-engine adjacency map, without touching the
    solver's normalized [0,1] representation anywhere else in the project.

    plot_width_m/plot_depth_m (used elsewhere, e.g. DXF export) reflect the
    required *area* for this room list, not necessarily the exact footprint
    the solver actually filled - the solved layout can legitimately not
    reach every edge of that nominal rectangle. Drawing the full nominal
    rectangle then leaves a visually confusing dead-space margin, so for
    this rendering we crop to the tight bounding box of the rooms that were
    actually placed and scale that box to real-world size using the same
    plot_width_m/plot_depth_m as the conversion factor - room-to-room
    proportions are unaffected, only the empty margin is trimmed.
    """
    plot_W = float(result["plot_width_m"])
    plot_D = float(result["plot_depth_m"])
    boxes = result["raw_boxes"] if use_raw else result["solved_boxes"]
    rooms = result["rooms"]

    xs0 = [float(b[0] - b[2] / 2) for b in boxes]
    xs1 = [float(b[0] + b[2] / 2) for b in boxes]
    ys0 = [float(b[1] - b[3] / 2) for b in boxes]
    ys1 = [float(b[1] + b[3] / 2) for b in boxes]
    bbox_x0, bbox_x1 = min(xs0), max(xs1)
    bbox_y0, bbox_y1 = min(ys0), max(ys1)
    bbox_w = max(bbox_x1 - bbox_x0, 1e-6)
    bbox_h = max(bbox_y1 - bbox_y0, 1e-6)

    W = bbox_w * plot_W
    D = bbox_h * plot_D

    real_rooms = {}
    for i, r in enumerate(rooms):
        cx = (float(boxes[i][0]) - bbox_x0) / bbox_w * W
        cy = (float(boxes[i][1]) - bbox_y0) / bbox_h * D
        w = float(boxes[i][2]) / bbox_w * W
        h = float(boxes[i][3]) / bbox_h * D
        real_rooms[i] = {
            "index": i,
            "type": r["type"],
            "type_id": r["type_id"],
            "room_index": r["room_index"],
            "cx": cx, "cy": cy, "w": w, "h": h,
            "x1": cx - w / 2, "x2": cx + w / 2,
            "y1": cy - h / 2, "y2": cy + h / 2,
        }

    import torch
    room_types = torch.tensor([r["type_id"] for r in rooms], dtype=torch.long)
    ids = list(range(len(rooms)))
    attach_map = _build_attach_map(boxes, room_types, ids)

    return W, D, real_rooms, attach_map


# --------------------------------------------------------------------------
# Wall network construction
# --------------------------------------------------------------------------

class WallSegment:
    __slots__ = ("orientation", "coord", "start", "end", "thickness", "cuts")

    def __init__(self, orientation, coord, start, end, thickness):
        self.orientation = orientation  # 'v' or 'h'
        self.coord = coord
        self.start = start
        self.end = end
        self.thickness = thickness
        self.cuts = []  # list of (gap_start, gap_end, inward_dx, inward_dy)


def build_interior_walls(real_rooms, W, D):
    v_edges = []  # (x, y1, y2)
    h_edges = []  # (y, x1, x2)
    for r in real_rooms.values():
        if r["type_id"] == 7:
            continue  # front_door is a door marker, not a walled room
        for x in (r["x1"], r["x2"]):
            if WALL_SNAP < x < W - WALL_SNAP:
                v_edges.append((x, r["y1"], r["y2"]))
        for y in (r["y1"], r["y2"]):
            if WALL_SNAP < y < D - WALL_SNAP:
                h_edges.append((y, r["x1"], r["x2"]))

    segments = []

    for rep, _ in cluster_coords([e[0] for e in v_edges], WALL_SNAP):
        ivs = [(y1, y2) for (x, y1, y2) in v_edges if abs(x - rep) <= WALL_SNAP]
        for s, e in merge_intervals(ivs, MERGE_TOL):
            if e - s > 1e-6:
                segments.append(WallSegment("v", rep, s, e, INT_WALL_T))

    for rep, _ in cluster_coords([e[0] for e in h_edges], WALL_SNAP):
        ivs = [(x1, x2) for (y, x1, x2) in h_edges if abs(y - rep) <= WALL_SNAP]
        for s, e in merge_intervals(ivs, MERGE_TOL):
            if e - s > 1e-6:
                segments.append(WallSegment("h", rep, s, e, INT_WALL_T))

    return segments


def build_exterior_walls(W, D):
    return [
        WallSegment("v", 0.0, 0.0, D, EXT_WALL_T),
        WallSegment("v", W, 0.0, D, EXT_WALL_T),
        WallSegment("h", 0.0, 0.0, W, EXT_WALL_T),
        WallSegment("h", D, 0.0, W, EXT_WALL_T),
    ]


def find_segment(segments, orientation, coord, at):
    best = None
    best_d = None
    for seg in segments:
        if seg.orientation != orientation:
            continue
        if abs(seg.coord - coord) > WALL_SNAP:
            continue
        if seg.start - 1e-6 <= at <= seg.end + 1e-6:
            d = abs(seg.coord - coord)
            if best is None or d < best_d:
                best, best_d = seg, d
    return best


def shared_edge(a, b):
    candidates = []
    for xa, xb in ((a["x2"], b["x1"]), (a["x1"], b["x2"])):
        gap = abs(xa - xb)
        ov_s, ov_e = max(a["y1"], b["y1"]), min(a["y2"], b["y2"])
        if gap <= WALL_SNAP and ov_e - ov_s > 1e-3:
            candidates.append(("v", (xa + xb) / 2, ov_s, ov_e, gap))
    for ya, yb in ((a["y2"], b["y1"]), (a["y1"], b["y2"])):
        gap = abs(ya - yb)
        ov_s, ov_e = max(a["x1"], b["x1"]), min(a["x2"], b["x2"])
        if gap <= WALL_SNAP and ov_e - ov_s > 1e-3:
            candidates.append(("h", (ya + yb) / 2, ov_s, ov_e, gap))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[4])
    o, c, s, e, _ = candidates[0]
    return o, c, s, e


def add_door_cut(seg, gap_center, gap_half, inward_pt):
    gs = max(seg.start + 0.02, gap_center - gap_half)
    ge = min(seg.end - 0.02, gap_center + gap_half)
    if ge - gs < 0.3:
        return
    if seg.orientation == "v":
        idx = 1.0 if inward_pt[0] >= seg.coord else -1.0
        seg.cuts.append((gs, ge, idx, 0.0))
    else:
        idy = 1.0 if inward_pt[1] >= seg.coord else -1.0
        seg.cuts.append((gs, ge, 0.0, idy))


def place_interior_doors(segments, real_rooms, attach_map):
    for src_idx, tgt_idx in attach_map.items():
        a = real_rooms.get(src_idx)
        b = real_rooms.get(tgt_idx)
        if a is None or b is None:
            continue
        if a["type_id"] == 7 or b["type_id"] == 7:
            continue  # front door handled separately (exterior wall)
        edge = shared_edge(a, b)
        if edge is None:
            continue
        o, c, s, e = edge
        seg = find_segment(segments, o, c, (s + e) / 2)
        if seg is None:
            continue
        lo, hi = seg.start + CORNER_CLEAR, seg.end - CORNER_CLEAR
        if hi <= lo:
            lo, hi = seg.start, seg.end
        mid = min(max((s + e) / 2, lo), hi)
        gap_half = min(DOOR_W, (seg.end - seg.start) * 0.7) / 2
        add_door_cut(seg, mid, gap_half, (b["cx"], b["cy"]))


def place_front_door(ext_segments, real_rooms, W, D):
    for r in real_rooms.values():
        if r["type_id"] != 7:
            continue
        cx, cy = r["cx"], r["cy"]
        dists = {"left": cx, "right": W - cx, "bottom": cy, "top": D - cy}
        side = min(dists, key=dists.get)
        center_pt = (W / 2, D / 2)
        gap_half = max(DOOR_W, r["w"] if side in ("bottom", "top") else r["h"]) / 2
        gap_half = min(gap_half, DOOR_W)
        if side in ("left", "right"):
            coord = 0.0 if side == "left" else W
            seg = next(s for s in ext_segments if s.orientation == "v" and abs(s.coord - coord) < 1e-6)
            lo, hi = seg.start + EXT_CORNER_CLEAR, seg.end - EXT_CORNER_CLEAR
            mid = min(max(cy, lo), hi)
            add_door_cut(seg, mid, gap_half, center_pt)
        else:
            coord = 0.0 if side == "bottom" else D
            seg = next(s for s in ext_segments if s.orientation == "h" and abs(s.coord - coord) < 1e-6)
            lo, hi = seg.start + EXT_CORNER_CLEAR, seg.end - EXT_CORNER_CLEAR
            mid = min(max(cx, lo), hi)
            add_door_cut(seg, mid, gap_half, center_pt)


# --------------------------------------------------------------------------
# Wall + door drawing
# --------------------------------------------------------------------------

def draw_wall_segment(ax, seg):
    t = seg.thickness
    cuts = sorted(seg.cuts)
    pieces = []
    cur = seg.start
    for gs, ge, _, _ in cuts:
        if gs > cur:
            pieces.append((cur, gs))
        cur = max(cur, ge)
    if cur < seg.end:
        pieces.append((cur, seg.end))
    if not pieces:
        pieces = [(seg.start, seg.end)]

    for s, e in pieces:
        pad_s = t / 2 if abs(s - seg.start) < 1e-6 else 0.0
        pad_e = t / 2 if abs(e - seg.end) < 1e-6 else 0.0
        s2, e2 = s - pad_s, e + pad_e
        if e2 - s2 <= 1e-6:
            continue
        if seg.orientation == "v":
            xy = (seg.coord - t / 2, s2)
            w, h = t, e2 - s2
        else:
            xy = (s2, seg.coord - t / 2)
            w, h = e2 - s2, t
        ax.add_patch(Rectangle(xy, w, h, facecolor="black", edgecolor="black",
                                linewidth=0, zorder=5))


def draw_door_symbol(ax, seg):
    for gs, ge, idx, idy in seg.cuts:
        width = ge - gs
        if width <= 0:
            continue
        if seg.orientation == "v":
            hinge = (seg.coord, gs)
            d = (0.0, 1.0)
            n = (idx, 0.0)
        else:
            hinge = (gs, seg.coord)
            d = (1.0, 0.0)
            n = (0.0, idy)

        leaf_end = (hinge[0] + n[0] * width, hinge[1] + n[1] * width)
        ax.add_line(Line2D([hinge[0], leaf_end[0]], [hinge[1], leaf_end[1]],
                            color="black", linewidth=1.1, zorder=6))

        ang_d = math.degrees(math.atan2(d[1], d[0]))
        ang_n = math.degrees(math.atan2(n[1], n[0]))
        diff = (ang_d - ang_n) % 360
        if abs(diff - 90) < 1:
            t1, t2 = ang_n, ang_n + 90
        else:
            t1, t2 = ang_n - 90, ang_n
        arc = Arc(hinge, width * 2, width * 2, angle=0, theta1=t1, theta2=t2,
                  color="black", linewidth=0.9, zorder=6)
        ax.add_patch(arc)


# --------------------------------------------------------------------------
# Furniture icons (per-room, real-world meters)
# --------------------------------------------------------------------------

def draw_bed(ax, x1, y1, x2, y2, margin):
    rw, rh = x2 - x1, y2 - y1
    if rw < 1.4 or rh < 1.4:
        return
    against_top = rh >= rw
    bw = min(1.6, rw - 2 * margin)
    bh = min(2.05, rh - 2 * margin)
    if bw <= 0.4 or bh <= 0.4:
        return
    bx1 = x1 + (rw - bw) / 2
    by1 = y1 + margin if against_top else y1 + (rh - bh) / 2
    bx2 = bx1 + bw

    ax.add_patch(Rectangle((bx1, by1), bw, bh, facecolor=FURN_FILL,
                            edgecolor="black", lw=1.0, zorder=3))
    ax.plot([bx1, bx2], [by1, by1], color="black", lw=2.2, zorder=3.1)
    pw, ph = bw * 0.40, bh * 0.15
    gap = bw * 0.08
    px = bx1 + (bw - 2 * pw - gap) / 2
    py = by1 + bh * 0.07
    ax.add_patch(Rectangle((px, py), pw, ph, facecolor="white",
                            edgecolor="black", lw=0.7, zorder=3.2))
    ax.add_patch(Rectangle((px + pw + gap, py), pw, ph, facecolor="white",
                            edgecolor="black", lw=0.7, zorder=3.2))
    fold_y = by1 + bh * 0.60
    ax.plot([bx1, bx2], [fold_y, fold_y], color="black", lw=0.7, zorder=3.2)
    available_right = (x2 - margin) - bx2
    ns = min(0.45, available_right - 0.1)
    if ns > 0.2:
        nx = bx2 + 0.08
        ax.add_patch(Rectangle((nx, by1), ns, ns, facecolor="white",
                                edgecolor="black", lw=0.7, zorder=3.1))


def draw_living(ax, x1, y1, x2, y2, margin):
    rw, rh = x2 - x1, y2 - y1
    if rw < 2.0 or rh < 2.0:
        return
    depth = min(0.72, rh * 0.28)
    arm1_w = min(rw * 0.55, 3.2)
    sx1 = x1 + margin
    sy2 = y2 - margin
    sy1 = sy2 - depth
    sx2 = sx1 + arm1_w
    ax.add_patch(Rectangle((sx1, sy1), arm1_w, depth, facecolor=FURN_FILL,
                            edgecolor="black", lw=1.0, zorder=3))
    for i in range(1, 3):
        cx = sx1 + arm1_w * i / 3
        ax.plot([cx, cx], [sy1, sy2], color="black", lw=0.6, zorder=3.1)

    arm2_h = min(rh * 0.42, 2.0)
    a2x1 = x1 + margin
    a2x2 = a2x1 + depth
    a2y2 = sy1
    a2y1 = a2y2 - arm2_h
    if a2y1 > y1 + margin and a2x2 < sx2:
        ax.add_patch(Rectangle((a2x1, a2y1), depth, arm2_h, facecolor=FURN_FILL,
                                edgecolor="black", lw=1.0, zorder=3))

    tw, th = min(1.0, rw * 0.22), min(0.55, rh * 0.18)
    tx = sx1 + depth + 0.35
    ty = sy1 - th - 0.35
    if tx + tw < x2 - margin and ty > y1 + margin + arm2_h * 0.2:
        ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="white",
                                edgecolor="black", lw=1.0, zorder=3))


def draw_kitchen(ax, x1, y1, x2, y2, margin):
    rw, rh = x2 - x1, y2 - y1
    if rw < 1.1 or rh < 1.1:
        return
    depth = min(0.62, rh * 0.35, rw * 0.35)
    if depth < 0.28:
        return
    cx1, cx2 = x1 + margin, x2 - margin
    cy2 = y2 - margin
    cy1 = cy2 - depth
    ax.add_patch(Rectangle((cx1, cy1), cx2 - cx1, depth, facecolor=FURN_FILL,
                            edgecolor="black", lw=1.0, zorder=3))
    cw = cx2 - cx1
    r = min(depth, cw) * 0.085
    stove_cx = cx1 + cw * 0.22
    stove_cy = (cy1 + cy2) / 2
    if r > 0.02:
        for dx in (-r * 1.5, r * 1.5):
            for dy in (-r * 1.5, r * 1.5):
                ax.add_patch(Circle((stove_cx + dx, stove_cy + dy), r,
                                     facecolor="white", edgecolor="black",
                                     lw=0.7, zorder=3.2))
    sink_w, sink_h = cw * 0.28, depth * 0.55
    sink_x = cx1 + cw * 0.68 - sink_w / 2
    sink_y = (cy1 + cy2) / 2 - sink_h / 2
    if sink_w > 0.1 and sink_h > 0.1:
        ax.add_patch(FancyBboxPatch((sink_x, sink_y), sink_w, sink_h,
                                     boxstyle=f"round,pad=0,rounding_size={min(sink_w, sink_h) * 0.2}",
                                     facecolor="white", edgecolor="black",
                                     lw=0.7, zorder=3.2))
        ax.add_patch(Circle((sink_x + sink_w * 0.5, sink_y + sink_h + 0.03), 0.02,
                             facecolor="black", edgecolor="black", zorder=3.3))


def draw_bathroom_fixtures(ax, x1, y1, x2, y2, margin):
    rw, rh = x2 - x1, y2 - y1
    if rw < 0.9 or rh < 0.9:
        return
    tw, td = min(0.42, rw * 0.4), min(0.62, rh * 0.45)
    tox = x2 - margin - tw
    toy = y1 + margin
    tank_h = td * 0.28
    ax.add_patch(Rectangle((tox, toy), tw, tank_h, facecolor=FURN_FILL,
                            edgecolor="black", lw=0.9, zorder=3))
    bowl_cy = toy + tank_h + (td - tank_h) * 0.5
    ax.add_patch(Ellipse((tox + tw / 2, bowl_cy), tw * 0.82, (td - tank_h) * 0.95,
                          facecolor="white", edgecolor="black", lw=0.9, zorder=3.1))

    sw, sd = min(0.5, rw * 0.4), min(0.34, rh * 0.3)
    if rw - tw - 2 * margin - 0.15 > sw:
        sx = x1 + margin
        sy = y1 + margin
        ax.add_patch(FancyBboxPatch((sx, sy), sw, sd,
                                     boxstyle=f"round,pad=0,rounding_size={min(sw, sd) * 0.3}",
                                     facecolor="white", edgecolor="black",
                                     lw=0.9, zorder=3.1))
        ax.add_patch(Circle((sx + sw / 2, sy + sd / 2), min(sw, sd) * 0.12,
                             facecolor="black", zorder=3.2))


def draw_storage_shelf(ax, x1, y1, x2, y2, margin):
    rw, rh = x2 - x1, y2 - y1
    if rw < 0.6 or rh < 0.6:
        return
    depth = min(0.35, rw * 0.3)
    sx1, sx2 = x1 + margin, x2 - margin
    sy1 = y1 + margin
    sy2 = sy1 + depth
    if sy2 >= y2 - margin:
        return
    ax.add_patch(Rectangle((sx1, sy1), sx2 - sx1, depth, facecolor="white",
                            edgecolor="black", lw=0.8, zorder=3))
    for i in range(1, 4):
        xx = sx1 + (sx2 - sx1) * i / 4
        ax.plot([xx, xx], [sy1, sy2], color="black", lw=0.5, zorder=3.1)


def draw_furniture(ax, room):
    t = room["type_id"]
    x1, y1, x2, y2 = room["x1"], room["y1"], room["x2"], room["y2"]
    margin = INT_WALL_T / 2 + 0.12
    if t == 0:
        draw_bed(ax, x1, y1, x2, y2, margin)
    elif t == 1:
        draw_bathroom_fixtures(ax, x1, y1, x2, y2, margin)
    elif t == 2:
        draw_kitchen(ax, x1, y1, x2, y2, margin)
    elif t == 3:
        draw_living(ax, x1, y1, x2, y2, margin)
    elif t == 5:
        draw_storage_shelf(ax, x1, y1, x2, y2, margin)
    # balcony (4): left plain (railing would double up with the exterior wall
    # already drawn there for a small room, and reads as clutter at this scale)


# --------------------------------------------------------------------------
# Labels + dimensions
# --------------------------------------------------------------------------

def draw_room_labels(ax, real_rooms, pts_per_m):
    bedroom_ids = [r["index"] for r in real_rooms.values() if r["type_id"] == 0]
    master_id = None
    if bedroom_ids:
        master_id = max(bedroom_ids, key=lambda i: real_rooms[i]["w"] * real_rooms[i]["h"])

    counts = {}
    for r in real_rooms.values():
        counts[r["type_id"]] = counts.get(r["type_id"], 0) + 1

    for r in real_rooms.values():
        if r["type_id"] == 7:
            continue

        if r["index"] == master_id:
            label = "MASTER BEDROOM"
        else:
            label = TYPE_LABELS.get(r["type_id"], r["type"].upper())
            if counts.get(r["type_id"], 1) > 1:
                label = f"{label} {r['room_index'] + 1}"

        dim_text = f"{feet_inches(r['w'])} x {feet_inches(r['h'])}"
        w_m, h_m = r["w"], r["h"]
        cx, cy = r["cx"], r["cy"]

        label_fs = fit_fontsize(label, w_m, h_m, pts_per_m, max_fs=12.5, min_fs=5.0, height_frac=0.30)
        dim_fs = fit_fontsize(dim_text, w_m, h_m, pts_per_m, max_fs=8.5, min_fs=4.2, height_frac=0.16)

        draw_label = label_fs >= 4.6
        draw_dims = dim_fs >= 4.0 and h_m > 0.9 and w_m > 0.9

        halo = dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88)

        if draw_label and draw_dims:
            label_h_m = label_fs * 1.15 / pts_per_m
            dim_h_m = dim_fs * 1.15 / pts_per_m
            line_gap_m = 2.0 / pts_per_m
            label_y = cy + label_h_m / 2 + line_gap_m / 2
            dim_y = cy - dim_h_m / 2 - line_gap_m / 2
            ax.text(cx, label_y, label, ha="center", va="center",
                    fontsize=label_fs, fontweight="bold", color="black",
                    family="sans-serif", zorder=8, bbox=halo)
            ax.text(cx, dim_y, dim_text, ha="center", va="center",
                    fontsize=dim_fs, color="#333333", family="sans-serif",
                    zorder=8, bbox=halo)
        elif draw_label:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=label_fs, fontweight="bold", color="black",
                    family="sans-serif", zorder=8, bbox=halo)


def draw_dim_line(ax, p0, p1, offset_vec, text, tick_size, fontsize):
    ox, oy = offset_vec
    q0 = (p0[0] + ox, p0[1] + oy)
    q1 = (p1[0] + ox, p1[1] + oy)

    ext_color = "#444444"
    ax.add_line(Line2D([p0[0], q0[0] + ox * 0.15], [p0[1], q0[1] + oy * 0.15],
                        color=ext_color, linewidth=0.6, zorder=3))
    ax.add_line(Line2D([p1[0], q1[0] + ox * 0.15], [p1[1], q1[1] + oy * 0.15],
                        color=ext_color, linewidth=0.6, zorder=3))
    ax.add_line(Line2D([q0[0], q1[0]], [q0[1], q1[1]], color="black", linewidth=0.8, zorder=3))

    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    for (qx, qy) in (q0, q1):
        ax.add_line(Line2D(
            [qx - (ux - px) * tick_size / 2, qx + (ux - px) * tick_size / 2],
            [qy - (uy - py) * tick_size / 2, qy + (uy - py) * tick_size / 2],
            color="black", linewidth=1.2, zorder=4))

    mx, my = (q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2
    label_ox = px * tick_size * 1.6
    label_oy = py * tick_size * 1.6
    rotation = 0 if abs(ux) >= abs(uy) else 90
    ax.text(mx + label_ox, my + label_oy, text, ha="center", va="center",
            fontsize=fontsize, rotation=rotation, color="black",
            family="sans-serif", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def draw_building_dimensions(ax, W, D, pts_per_m):
    tick = max(0.08, min(min(W, D) * 0.02, 0.22))
    fs = max(7.5, min(11, pts_per_m * 0.09))
    off_w = min(0.9, max(0.5, D * 0.07))
    off_d = min(0.9, max(0.5, W * 0.07))

    draw_dim_line(ax, (0, 0), (W, 0), (0, -off_w), f"{W:.2f} m", tick, fs)
    draw_dim_line(ax, (0, 0), (0, D), (-off_d, 0), f"{D:.2f} m", tick, fs)


# --------------------------------------------------------------------------
# Main render
# --------------------------------------------------------------------------

def render_cad(result, output_path, use_raw=False, title="FLOOR PLAN"):
    """Render a generate_floor_plan() result as a CAD-style architectural
    drawing and save it to output_path. Set use_raw=True to render the raw
    (pre-solver) model prediction instead of the solved layout."""
    W, D, real_rooms, attach_map = _real_rooms_from_result(result, use_raw=use_raw)

    interior_walls = build_interior_walls(real_rooms, W, D)
    exterior_walls = build_exterior_walls(W, D)

    place_interior_doors(interior_walls, real_rooms, attach_map)
    place_front_door(exterior_walls, real_rooms, W, D)

    pad_left, pad_bottom = 1.5, 1.5
    pad_right, pad_top = 0.7, 0.7
    total_w = W + pad_left + pad_right
    total_h = D + pad_bottom + pad_top
    target_max_in = 12.5
    scale = target_max_in / max(total_w, total_h)
    fig_w = total_w * scale
    fig_h = total_h * scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    pts_per_m = scale * 72.0

    ax.set_xlim(-pad_left, W + pad_right)
    ax.set_ylim(-pad_bottom, D + pad_top)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.axis("off")

    for r in real_rooms.values():
        if r["type_id"] != 7:
            draw_furniture(ax, r)

    for seg in interior_walls:
        draw_wall_segment(ax, seg)
    for seg in exterior_walls:
        draw_wall_segment(ax, seg)
    for seg in interior_walls + exterior_walls:
        draw_door_symbol(ax, seg)

    draw_room_labels(ax, real_rooms, pts_per_m)
    draw_building_dimensions(ax, W, D, pts_per_m)

    ax.set_title(title, fontsize=14, fontweight="bold", family="sans-serif", pad=14)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    ML_DIR = ROOT / "python" / "app" / "ml"
    if str(ML_DIR) not in sys.path:
        sys.path.insert(0, str(ML_DIR))
    from predict import generate_floor_plan  # noqa: E402

    demo_requirements = {
        "bedrooms": 3,
        "bathrooms": 2,
        "kitchens": 1,
        "living_rooms": 1,
        "balconies": 2,
        "storages": 1,
    }

    print("=" * 70)
    print("AI FLOOR PLANNER - CAD-STYLE RENDER")
    print("=" * 70)

    result = generate_floor_plan(demo_requirements)
    out = ROOT / "dataset" / "generated_plans" / "demo_plan_cad.png"
    render_cad(result, out)

    print(f"\nRooms: {len(result['rooms'])}")
    print(f"Plot : {result['plot_width_m']:.2f} m x {result['plot_depth_m']:.2f} m")
    print(f"Saved: {out}")
    print("\nDone.")
